import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import math

from utils.desolve import eval_file
from data_prepare import train_data_prepare


class AnnoData(Dataset):
    '''
    to load train dataset or dev dataset
    '''
    def __init__(self, file_dir, name_to_com, max_length=160):
        super().__init__()
        self.file_dir = file_dir
        self.max_length = max_length
        self.name_to_com = name_to_com
        self.json_data = eval_file(file_dir)
        self.data = train_data_prepare(self.json_data, name_to_com, TOKENIZER, max_length)
        self._len = self.data['ids'].shape[0]
        self.ids = self.data['ids']
        self.mask_mat = self.data['mask_mat']
        self.ent_mask = self.data['ent_mask']
        self.kb_ids = self.data['kb_ids']
        self.labels = self.data['labels']
        self.result_file = None
    
    def __len__(self):
        return self._len
    
    def __getitem__(self,index):
        offset = np.argwhere(self.ent_mask[index]==1)[0]
        sample = {
            'ids': self.ids[index],
            'mask_mat': self.mask_mat[index],
            'ent_mask': self.ent_mask[index],
            'kb_ids': self.kb_ids[index],
            'offset': offset,
            'labels': self.labels[index]
        }
        return sample

    def write_result(self, ):
        pass

    def evaluate(self):
        pass
