import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import math
import datetime
import json

from utils.desolve import eval_file
from .data_prepare import train_data_prepare

TOKENIZER = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

class BaseData(Dataset):
    def __init__ (self):
        self.result_json = {}
        self.result_json["team_name"] = "ddl自动机"
        self.result_json["submit_result"] = []
        self.raw_data = None
        self.threshold = 0.5
        self.need_post = False
        self.result_dir = None
        self.result_file_name = None
        self.gt_file_name = None  # filename of ground truth file

    def get_batch_result(self, batch_input, batch_output):
        for index, pred in zip(batch_input['index'], batch_output):
            result_item = {}
            text_id, text, mention, offset, kb_id = self.raw_data[index]
            result_item["text_id"] = text_id
            result_item["text"] = text
            result_item["mention_result"] = []
            mention_item = {}
            mention_item["mention"] = mention
            mention_item["offset"] = offset
            pred = pred.item()
            mention_item["kb_id"] = kb_id if pred >= self.threshold else -1
            mention_item["confidence"] = max(pred, 1 - pred)
            result_item["mention_result"].append(mention_item)
            self.result_json["submit_result"].append(result_item)

    def result_post_proc(self):
        pass

    def write_json(self, epoch):
        if self.need_post:
            self.result_post_proc()
        # if self.result_file_name is None:
        # curr_time = datetime.datetime.now()
        # self.result_file_name = self.result_dir + '/' +  curr_time.strftime("%m-%d") + '.json'
        filename = self.result_dir + '/' + str(epoch) + '.json'
        json.dump(
            self.result_json,
            open(filename, 'w', encoding='utf8'), 
            ensure_ascii=False, 
            indent=True)
        self.result_json['submit_result'] = []

    def evaluate(self, epoch):
        ''' only used for dev dataset '''
        filename = self.result_dir + '/' + str(epoch) + '.json'
        print('evaluate '+filename)
        re_file = eval_file(filename)
        gt_set = eval_file(self.gt_file_name)
        re_set = re_file["submit_result"]
        match_cnt = 0
        M = 0
        M_ = 0
        for re, gt in zip(re_set, gt_set):
            re_results = re["mention_result"]
            gt_results = gt["lab_result"]
            M += len(gt_results)
            M_ += len(re_results)
            flag = True
            for re_item in re_results:
                for gt_item in gt_results:
                    if re_item["mention"] == gt_item["mention"] \
                    and re_item["offset"] == gt_item["offset"] \
                    and re_item["kb_id"] == gt_item["kb_id"]:
                        # del
                        flag = False
                        match_cnt += 1
            #if flag:
            #    print(re)
            #    print(gt)
        R = match_cnt / M
        P = match_cnt / M_
        F1 = 2 * P * R / (P + R)
        return R, P, F1



class AnnoData(BaseData):
    '''
    to load train dataset or dev dataset
    '''
    def __init__(self, file_dir, name_to_com, id_to_com, max_length=160, threshold=0.5, result_dir=None):
        super().__init__()
        self.gt_file_name = file_dir
        self.max_length = max_length
        self.name_to_com = name_to_com
        self.id_to_com = id_to_com
        self.threshold = threshold
        self.result_dir = result_dir
        json_data = eval_file(file_dir)
        self.raw_data, self.data = train_data_prepare(json_data, name_to_com, TOKENIZER, max_length)
        self._len = self.data['ids'].shape[0]
        self.ids = torch.tensor(self.data['ids'], dtype=torch.long)
        self.mask_mat = torch.tensor(self.data['mask_mat'], dtype=torch.long)
        self.ent_mask = torch.tensor(self.data['ent_mask'], dtype=torch.long)
        self.kb_ids = torch.tensor(self.data['kb_ids'], dtype=torch.long)
        self.labels = torch.tensor(self.data['labels'], dtype=torch.uint8)
        self.offsets = torch.tensor(self.data['offsets'], dtype=torch.long)
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, index):
        # offset = np.argwhere(self.ent_mask[index]==1)[0]
        sample = {
            'index': index,
            'ids': self.ids[index],
            'mask_mat': self.mask_mat[index],
            'ent_mask': self.ent_mask[index],
            'kb_ids': self.kb_ids[index],
            'offset': self.offsets[index],
            'labels': self.labels[index]
        }
        return sample

class TestData(BaseData):
    '''
    for inference and test
    '''
    def __init__(self, file_dir, name_to_com, id_to_com, max_length=160):
        super().__init__()
        # self.gt_file_name = file_dir
        # self.max_length = max_length
        # self.name_to_com = name_to_com
        # self.id_to_com = id_to_com
        # json_data = eval_file(file_dir)
        # self.raw_data, self.data = train_data_prepare(json_data, name_to_com, TOKENIZER, max_length)
        # self._len = self.data['ids'].shape[0]
        # self.ids = torch.tensor(self.data['ids'], dtype=torch.long)
        # self.mask_mat = torch.tensor(self.data['mask_mat'], dtype=torch.long)
        # self.ent_mask = torch.tensor(self.data['ent_mask'], dtype=torch.long)
        # self.kb_ids = torch.tensor(self.data['kb_ids'], dtype=torch.long)
        # self.labels = torch.tensor(self.data['labels'], dtype=torch.uint8)
        # self.offsets = torch.tensor(self.data['offsets'], dtype=torch.long)
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, index):
        # offset = np.argwhere(self.ent_mask[index]==1)[0]
        # sample = {
        #     'index': index,
        #     'ids': self.ids[index],
        #     'mask_mat': self.mask_mat[index],
        #     'ent_mask': self.ent_mask[index],
        #     'kb_ids': self.kb_ids[index],
        #     'offset': self.offsets[index],
        #     'labels': self.labels[index]
        # }
        return sample
