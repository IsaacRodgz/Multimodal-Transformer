import torch
import argparse
from torch.utils.data import DataLoader
from src.utils import *
from src import train

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"


parser = argparse.ArgumentParser(description='Multimodal classification on Images and Text')

# Fixed
parser.add_argument('--model', type=str, default='MMTransformer',
                    help='name of the model to use (MMTransformer, etc.)')

# Tasks
parser.add_argument('--dataset', type=str, default='meme_dataset',
                    help='dataset to use (default: meme_dataset)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')
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

# Architecture
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

# Tuning
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size (default: 8)')
parser.add_argument('--max_token_length', type=int, default=50,
                    help='max number of tokens per sentence (default: 50)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='initial learning rate (default: 2e-5)')
parser.add_argument('--optim', type=str, default='AdamW',
                    help='optimizer to use (default: AdamW)')
parser.add_argument('--num_epochs', type=int, default=3,
                    help='number of epochs (default: 3)')
parser.add_argument('--when', type=int, default=2,
                    help='when to decay learning rate (default: 2)')

# Logistics
parser.add_argument('--log_interval', type=int, default=100,
                    help='frequency of result logging (default: 100)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='model',
                    help='name of the trial (default: "model")')

args = parser.parse_args()

valid_partial_mode = args.lonly + args.vonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v}only.")

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
print(dataset)

use_cuda = False

output_dim_dict = {
    'meme_dataset': 2,
    'mmimdb': 23
}

criterion_dict = {
    'meme_dataset': 'CrossEntropyLoss',
    'mmimdb': 'BCEWithLogitsLoss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset
#
####################################################################

print("Start loading the data....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'dev')
test_data = get_data(args, dataset, 'test')

'''
def my_collate(batch):
    max_len = max([item['input_ids'].shape[0] for item in batch])
    txt_batch = [item['input_ids'] for item in batch]
    img_batch = [item['image'] for item in batch]
    label_batch = [item['label'] for item in batch]
    
    return {'input_ids': txt_batch, 'image': img_batch, 'label': label_batch}
'''

train_loader = DataLoader(train_data,
                        batch_size=args.batch_size,
                        shuffle=True,
                        #collate_fn=my_collate,
                        num_workers=32)
valid_loader = DataLoader(valid_data,
                        batch_size=args.batch_size,
                        shuffle=True,
                        #collate_fn=my_collate,
                        num_workers=32)
if test_data is None:
    test_loader = None
else:
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=32)

print('Finish loading the data....')

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_v = 768, hyp_params.image_feature_size
hyp_params.l_len, hyp_params.v_len = hyp_params.max_token_length, 1
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.n_train, hyp_params.n_valid = len(train_data), len(valid_data)
if test_data is None:
    pass
else:
    hyp_params.n_test = len(test_data)
hyp_params.model = args.model.strip()
hyp_params.output_dim = output_dim_dict.get(dataset)
hyp_params.criterion = criterion_dict.get(dataset)

if __name__ == '__main__':
    test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)