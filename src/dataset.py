import os
import torch
import json
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
import json

class MemeDataset(Dataset):
    """Hateful memes dataset from Facebook challenge"""

    def __init__(self, root_dir, dataset, split, transform=None):
        """
        Args:
            jsonl_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Metadata
        self.full_data_path = os.path.join(root_dir, dataset) + f'/{split}.jsonl'
        self.data_dict = pd.read_json(self.full_data_path, lines=True)
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform

        # BERT tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir+'/'+self.dataset+'/'+self.data_dict.iloc[idx,1]
        image = Image.open(img_name).convert('RGB')
        label = self.data_dict.iloc[idx,2]

        text = self.data_dict.iloc[idx,3]
        text_encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        if self.transform:
            image = self.transform(image)

        sample = {'image': image,
                  'input_ids': text_encoded['input_ids'].flatten(),
                  'attention_mask': text_encoded['attention_mask'].flatten(),
                  "label": int(label)}

        return sample
    
    
class MMIMDbDataset(Dataset):
    """Multimodal IMDb dataset (http://lisi1.unal.edu.co/mmimdb)"""

    def __init__(self, root_dir, dataset, idxs):
        """
        Args:
            root_dir (string): Directory with all data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.full_data_path = os.path.join(root_dir, dataset)+'/data_bert_120/features'
        self.idxs = idxs
        
    def __len__(self):
        
        return len(self.idxs)

    def __getitem__(self, idx):
        data = torch.load(self.full_data_path+'/'+self.idxs[idx]+'.pt')
        
        text_feature = data['txt']
        visual_feature = data['img']
        labels = data['labels']

        sample = {'image': visual_feature,
                  'input_ids': text_feature,
                  "label": labels.type(torch.FloatTensor)}

        return sample