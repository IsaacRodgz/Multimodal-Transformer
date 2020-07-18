import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.sklearns import F1
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna
from optuna.integration import PyTorchLightningPruningCallback

import math
from argparse import ArgumentParser
from src.dataset import *
from src.utils import *
from src.eval_metrics import *
from numpy import load
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

pl.seed_everything(1111)


class MMTransformerModel(pl.LightningModule):
    def __init__(self, hyp_params, trial=None):
        """
        Construct a MulT model.
        """
        super(MMTransformerModel, self).__init__()
        
        self.save_hyperparameters()
        self.hyp_params = hyp_params
        self.trial = trial
        
        if trial is None:
            attn_dropout = hyp_params.attn_dropout
            attn_dropout_v = hyp_params.attn_dropout_v
            relu_dropout = hyp_params.relu_dropout
            res_dropout = hyp_params.res_dropout
            out_dropout = hyp_params.out_dropout
            embed_dropout = hyp_params.embed_dropout
        else:
            '''
            attn_dropout = trial.suggest_uniform("attn_dropout", 0.0, 0.5)
            attn_dropout_v = trial.suggest_uniform("attn_dropout_v", 0.0, 0.5)
            relu_dropout = trial.suggest_uniform("relu_dropout", 0.0, 0.5)
            res_dropout = trial.suggest_uniform("res_dropout", 0.0, 0.5)
            out_dropout = trial.suggest_uniform("out_dropout", 0.0, 0.5)
            embed_dropout = trial.suggest_uniform("embed_dropout", 0.0, 0.5)
            '''
            attn_dropout = hyp_params.attn_dropout
            attn_dropout_v = hyp_params.attn_dropout_v
            relu_dropout = hyp_params.relu_dropout
            res_dropout = hyp_params.res_dropout
            out_dropout = hyp_params.out_dropout
            embed_dropout = hyp_params.embed_dropout
        
        self.orig_d_l, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_v
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.vonly = True
        self.lonly = True
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = attn_dropout
        self.attn_dropout_v = attn_dropout_v
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout
        self.attn_mask = hyp_params.attn_mask

        combined_dim = self.d_l + self.d_v

        self.partial_mode = self.lonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = self.d_l   # assuming d_l == d_v
        else:
            combined_dim = (self.d_l + self.d_v)
        
        output_dim = hyp_params.output_dim

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['v', 'lv']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self, x_l, x_v):
        """
        text, and vision should have dimension [batch_size, seq_len, n_features]
        """
        
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = x_v.transpose(1, 2)
        
        #print("Input size: ")
        #print("L: ", x_l.size())
        #print("V: ", x_v.size())
       
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        
        #print("Projected size: ")
        #print("L: ", x_l.size())
        #print("V: ", x_v.size())

        if self.lonly:
            # V --> L
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = self.trans_l_mem(h_l_with_vs)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.vonly:
            # L --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_vs = self.trans_v_mem(h_v_with_ls)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
            
        #print("MM size: ")
        #print("L: ", h_l_with_vs.size())
        #print("V: ", h_v_with_ls.size())
            
        #print("Last size: ")
        #print("L: ", last_h_l.size())
        #print("V: ", last_h_v.size())
        
        if self.partial_mode == 2:
            last_hs = torch.cat([last_h_l, last_h_v], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output, last_hs
    
    def loss_function(self, outputs, targets):
        return F.binary_cross_entropy_with_logits(outputs, targets, weight=self.hyp_params.label_weights.cuda())

    
    def configure_optimizers(self):
        if self.trial is None:
            lr = self.hyp_params.lr
            gamma = self.hyp_params.gamma
            when = self.hyp_params.when
        else:
            lr = self.trial.suggest_loguniform('learning_rate', 1e-6, 0.1)
            gamma = self.trial.suggest_uniform("sch_gamma", 0.1, 0.9)
            when = self.trial.suggest_int("when", 1, 30)

        optimizer = getattr(optim, self.hyp_params.optim)(self.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=when, factor=gamma, verbose=True)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        targets = batch["label"]
        images = batch['image']

        outputs, hiddens = self(
            x_l=input_ids,
            x_v=images
        )
        
        loss = self.loss_function(outputs, targets)
        
        if self.hyp_params.gpus > 1:
            loss = loss.unsqueeze(0)
        
        return {'loss': loss}
    
    def train_dataloader(self):
        data_path = os.path.join(args.data_path, dataset) + '/data_bert_'+str(self.hyp_params.max_token_length)+'/partition.json'
        
        with open(data_path) as json_file:
            labels = json.load(json_file)['train']
        
        train_data = MMIMDbDataset(self.hyp_params.data_path,
                                   self.hyp_params.dataset,
                                   self.hyp_params.max_token_length,
                                   labels
                                  )
        
        train_loader = DataLoader(train_data,
                                  batch_size=self.hyp_params.batch_size,
                                  shuffle=True,
                                  num_workers=self.hyp_params.num_workers
                                 )
        
        return train_loader
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        targets = batch["label"]
        images = batch['image']
        
        outputs, hiddens = self(
            x_l=input_ids,
            x_v=images
        )
                
        loss = self.loss_function(outputs, targets)
        
        outputs = (outputs > 0.5).float()

        _, f1_micro, f1_macro, f1_weighted, f1_samples = metrics(outputs, targets)
        f1_micro = torch.from_numpy(np.array(f1_micro)).to(self.device)
        f1_macro = torch.from_numpy(np.array(f1_macro)).to(self.device)
        f1_weighted = torch.from_numpy(np.array(f1_weighted)).to(self.device)
        f1_samples = torch.from_numpy(np.array(f1_samples)).to(self.device)
        
        if self.hyp_params.gpus > 1:
            loss = loss.unsqueeze(0)
        
        return {'val_loss': loss,
                'val_f1_micro': f1_micro,
                'val_f1_macro': f1_macro,
                'val_f1_weighted': f1_weighted,
                'val_f1_samples': f1_samples}
    
    def validation_epoch_end(self, outputs):
        val_loss_mean = 0
        val_f1_micro_mean = 0
        val_f1_macro_mean = 0
        
        for x in outputs:
        
            val_loss = x['val_loss']
            val_f1_micro = x['val_f1_micro']
            val_f1_macro = x['val_f1_macro']
            
            if self.hyp_params.gpus > 1:
                val_loss = torch.mean(val_loss)
                val_f1_micro = torch.mean(val_f1_micro)
                val_f1_macro = torch.mean(val_f1_macro)
            
            val_loss_mean += val_loss
            val_f1_micro_mean += val_f1_micro
            val_f1_macro_mean += val_f1_macro
            
        val_loss_mean /= len(outputs)
        val_f1_micro_mean /= len(outputs)
        val_f1_macro_mean /= len(outputs)

        #avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #avg_f1_micro = torch.stack([x['val_f1_micro'] for x in outputs]).mean()
        #avg_f1_macro = torch.stack([x['val_f1_macro'] for x in outputs]).mean()
        #avg_f1_weighted = torch.stack([x['val_f1_weighted'] for x in outputs]).mean()
        #avg_f1_samples = torch.stack([x['val_f1_samples'] for x in outputs]).mean()

        self.logger.experiment.add_scalar('f1-micro/val', val_f1_micro_mean, self.current_epoch+1)
        self.logger.experiment.add_scalar('f1-macro/val', val_f1_macro_mean, self.current_epoch+1)
        self.logger.experiment.add_scalar('loss/val', val_loss_mean, self.current_epoch+1)
        #self.logger.experiment.add_scalar('f1-weighted/val', avg_f1_weighted, self.current_epoch+1)
        #self.logger.experiment.add_scalar('f1-samples/val', avg_f1_samples, self.current_epoch+1)
        
        return {'val_loss': val_loss_mean, 'val_f1_macro': val_f1_macro_mean}
    
    def val_dataloader(self):
        data_path = os.path.join(args.data_path, dataset) + '/data_bert_'+str(self.hyp_params.max_token_length)+'/partition.json'
        
        with open(data_path) as json_file:
            labels = json.load(json_file)['dev']
        
        val_data = MMIMDbDataset(self.hyp_params.data_path,
                                   self.hyp_params.dataset,
                                   self.hyp_params.max_token_length,
                                   labels
                                  )
        
        val_loader = DataLoader(val_data,
                                batch_size=self.hyp_params.batch_size,
                                num_workers=self.hyp_params.num_workers
                               )
        
        return val_loader
    
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        targets = batch["label"]
        images = batch['image']
        
        outputs, hiddens = self(
            x_l=input_ids,
            x_v=images
        )
        
        loss = self.loss_function(outputs, targets)
        outputs = (outputs > 0.5).float()
        _, f1_micro, f1_macro, f1_weighted, f1_samples = metrics(outputs, targets)
        f1_micro = torch.from_numpy(np.array(f1_micro)).to(self.device)
        f1_macro = torch.from_numpy(np.array(f1_macro)).to(self.device)
        f1_weighted = torch.from_numpy(np.array(f1_weighted)).to(self.device)
        f1_samples = torch.from_numpy(np.array(f1_samples)).to(self.device)
        f1_per_class = torch.from_numpy(report_per_class(outputs, targets)).to(self.device)
        
        return {'test_loss': loss,
                'test_f1_micro': f1_micro,
                'test_f1_macro': f1_macro,
                'test_f1_weighted': f1_weighted,
                'test_f1_samples': f1_samples,
                'test_f1_per_class': f1_per_class}
    
    def test_epoch_end(self, outputs):
        test_loss_mean = 0
        test_f1_micro_mean = 0
        test_f1_macro_mean = 0
        test_f1_per_class_mean = 0
        
        for x in outputs:
        
            test_loss = x['test_loss']
            test_f1_micro = x['test_f1_micro']
            test_f1_macro = x['test_f1_macro']
            test_f1_per_class = x['test_f1_per_class']
            
            if self.hyp_params.gpus > 1:
                test_loss = torch.mean(test_loss)
                test_f1_micro = torch.mean(test_f1_micro)
                test_f1_macro = torch.mean(test_f1_macro)
                test_f1_per_class = torch.mean(test_f1_per_class)
            
            test_loss_mean += test_loss
            test_f1_micro_mean += test_f1_micro
            test_f1_macro_mean += test_f1_macro
            test_f1_per_class_mean += test_f1_per_class
            
        test_loss_mean /= len(outputs)
        test_f1_micro_mean /= len(outputs)
        test_f1_macro_mean /= len(outputs)
        test_f1_per_class_mean /= len(outputs)
            
        print("f1-score per class: ", test_f1_per_class_mean)
        
        #avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        #avg_f1_micro = torch.stack([x['test_f1_micro'] for x in outputs]).mean()
        #avg_f1_macro = torch.stack([x['test_f1_macro'] for x in outputs]).mean()
        #avg_f1_weighted = torch.stack([x['test_f1_weighted'] for x in outputs]).mean()
        #avg_f1_samples = torch.stack([x['test_f1_samples'] for x in outputs]).mean()
 
        return {'test_loss': test_loss_mean,
                'test_f1_micro': test_f1_micro_mean,
                'test_f1_macro': test_f1_macro_mean
               }
    
    def test_dataloader(self):
        data_path = os.path.join(args.data_path, dataset) + '/data_bert_'+str(self.hyp_params.max_token_length)+'/partition.json'
        
        with open(data_path) as json_file:
            labels = json.load(json_file)['test']
        
        test_data = MMIMDbDataset(self.hyp_params.data_path,
                                   self.hyp_params.dataset,
                                   self.hyp_params.max_token_length,
                                   labels
                                  )
        
        test_loader = DataLoader(test_data,
                                 batch_size=self.hyp_params.batch_size,
                                 num_workers=self.hyp_params.num_workers
                                )
        
        return test_loader
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
        parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
        parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
        # Dropouts
        parser.add_argument('--attn_dropout', type=float, default=0.1,
                            help='attention dropout')
        parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                            help='attention dropout (for visual)')
        parser.add_argument('--relu_dropout', type=float, default=0.1,
                            help='relu dropout')
        parser.add_argument('--embed_dropout', type=float, default=0.25,
                            help='embedding dropout')
        parser.add_argument('--res_dropout', type=float, default=0.1,
                            help='residual block dropout')
        parser.add_argument('--out_dropout', type=float, default=0.0,
                            help='output layer dropout')
        parser.add_argument('--bert_model', type=str, default="bert-base-cased",
                    help='pretrained bert model to use')
        parser.add_argument('--cnn_model', type=str, default="vgg16",
                            help='pretrained CNN to use for image feature extraction')
        parser.add_argument('--image_feature_size', type=int, default=4096,
                            help='image feature size extracted from pretrained CNN (default: 4096)')
        parser.add_argument('--nlevels', type=int, default=5,
                            help='number of layers in the network (default: 5)')
        parser.add_argument('--num_heads', type=int, default=5,
                            help='number of heads for the transformer network (default: 5)')
        parser.add_argument('--attn_mask', action='store_false',
                            help='use attention mask for Transformer (default: true)')
        parser.add_argument('--gamma', type=float, default=0.5,
                            help='Scheduler factor (default: 0.5)')
        return parser


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def objective(trial):
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join('checkpoints', "trial_{}".format(trial.number)), monitor="val_f1_macro"
    )

    metrics_callback = MetricsCallback()
    run_name = create_run_name(args)
    logger = TensorBoardLogger(save_dir='runs_pl_temp/', name=run_name)

    trainer = pl.Trainer(gpus=args.gpus,
                         distributed_backend='dp',
                         logger=logger,
                         max_epochs=args.max_epochs,
                         gradient_clip_val=trial.suggest_uniform("clip", 0.1, 0.9),
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_f1_macro"),
                         callbacks=[metrics_callback],
                        )
    
    mlp = MMTransformerModel(args, trial)
    trainer.fit(mlp)

    return metrics_callback.metrics[-1]["val_f1_macro"].item()

    
if __name__ == '__main__':
    
    parser = ArgumentParser(description='mmimdb model')

    # add PROGRAM level args
    parser.add_argument('--model', type=str, default='GMU',
                    help='name of the model to use (GMU, Concatenate)')
    parser.add_argument('--dataset', type=str, default='mmimdb',
                        help='dataset to use (default: mmimdb)')
    parser.add_argument('--data_path', type=str, default='data',
                        help='path for storing the dataset')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--chk', type=int, default=3,
                        help='Number of top models to consider for checkpoint (default: 3)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='do not use cuda')
    parser.add_argument('--name', type=str, default='model',
                    help='name of the trial (default: "model")')
    parser.add_argument('--search', action='store_true',
                        help='Hyperparameter search')
    parser.add_argument('--test', action='store_true',
                        help='Load pretrained model for test evaluation')
    
    # add MODEL level args
    parser = MMTransformerModel.add_model_specific_args(parser)
    
    # add TRAINER level args
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size (default: 8)')
    parser.add_argument('--max_token_length', type=int, default=50,
                    help='max number of tokens per sentence (default: 50)')
    parser.add_argument('--lr', type=float, default=2e-5,
                    help='initial learning rate (default: 2e-5)')
    parser.add_argument('--optim', type=str, default='AdamW',
                    help='optimizer to use (default: AdamW)')
    parser.add_argument('--when', type=int, default=2,
                    help='when to decay learning rate (default: 2)')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--gradient_clip_val', type=float, default=0.8)
    
    args = parser.parse_args()
    dataset = str.lower(args.dataset.strip())
    
    use_cuda = False

    output_dim_dict = {
        'meme_dataset': 2,
        'mmimdb': 23
    }

    criterion_dict = {
        'meme_dataset': 'CrossEntropyLoss',
        'mmimdb': 'BCEWithLogitsLoss'
    }

    '''
    torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available():
        if args.no_cuda:
            print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
        else:
            torch.cuda.manual_seed(args.seed)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            use_cuda = True
    '''

    args.orig_d_l, args.orig_d_v = 768, args.image_feature_size
    args.l_len, args.v_len = args.max_token_length, 1
    args.layers = args.nlevels
    args.use_cuda = use_cuda
    args.dataset = dataset
    args.model = args.model.strip()
    args.output_dim = output_dim_dict.get(dataset)
    args.label_weights = torch.load('class_weights/class_weights_1.pt').cuda()
    
    if args.search:
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=100, timeout=None)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    
    else:
        if args.test:
            num = 44
            model = MMTransformerModel.load_from_checkpoint(f'pre_trained_models/MLPGenreClassifier_{num}.ckpt')
            
            trainer = pl.Trainer(gpus=args.gpus)
            
            trainer.test(model)
            
        else:
            checkpoint_callback = ModelCheckpoint(
                filepath='pre_trained_models/'+args.name+'_{epoch}_{val_f1_macro:.4f}',
                save_top_k=args.chk,
                verbose=True,
                monitor='val_f1_macro',
                mode='max',
                prefix=''
            )
            
            run_name = create_run_name(args)
            logger = TensorBoardLogger(save_dir='runs_pl/', name=run_name)
            
            #trainer = pl.Trainer(gpus=2, distributed_backend='dp', logger=logger, max_epochs=args.max_epochs)
            trainer = pl.Trainer(gpus=args.gpus,
                                 distributed_backend='dp',
                                 max_epochs=args.max_epochs,
                                 gradient_clip_val=args.gradient_clip_val,
                                 logger=logger,
                                 checkpoint_callback=checkpoint_callback
                                )

            mlp = MMTransformerModel(args)

            trainer.fit(mlp)

            trainer.test(ckpt_path=None)