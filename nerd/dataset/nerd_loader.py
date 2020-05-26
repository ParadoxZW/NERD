import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import math
import datetime
import json
import pickle

from utils.desolve import eval_file, desolve_str_to_list
from .data_prepare import train_data_prepare, test_data_prepare

TOKENIZER = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

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
        self.is_test = False

    def get_batch_result(self, batch_input, batch_output):
        if self.is_test:
            preds = np.asarray(batch_output.cpu()).reshape(1, -1)
            self.pred_list.append(preds)
        for index, pred in zip(batch_input['index'], batch_output):
            result_item = {}
            text_id, text, mention, offset, kb_id = self.raw_data[index]
            result_item["text_id"] = text_id
            result_item["text"] = text
            result_item["mention_result"] = []
            mention_item = {}
            mention_item["mention"] = mention
            mention_item["offset"] = offset
            pred = torch.sigmoid(pred).item()
            mention_item["kb_id"] = kb_id if pred >= self.threshold else -1
            mention_item["confidence"] = max(pred, 1 - pred)
            result_item["mention_result"].append(mention_item)
            self.result_json["submit_result"].append(result_item)

    def result_post_proc(self):
        # only use for inference
        # merge multi-mention for same text sample
        # `results/runtime/infer.json` is just a intermediate result file.
        filename = self.result_dir + '/infer.json'
        with open(filename, 'r', encoding='utf8') as js_file:
            js = json.load(js_file)
            res = js['submit_result']
            new_js = {"team_name": "ddl自动机",}
            news = []
            pn = None
            i = None
            for item in res:
                if item['text_id'] == i:
                    pn['mention_result'] += item['mention_result']
                else:
                    i = item['text_id']
                    pn = deepcopy(item)
                    news.append(pn)
            new_js['submit_result'] = news
            return new_js # merged json result file


    def write_json(self, epoch):
        # save json file in local
        # the file is just a intermediate file,
        # which needs post-process
        filename = self.result_dir + '/' + str(epoch) + '.json'
        json.dump(
            self.result_json,
            open(filename, 'w', encoding='utf8'), 
            ensure_ascii=False, 
            indent=True)
        self.result_json['submit_result'] = []
        if self.is_test:
            preds = np.hstack(self.pred_list).reshape(len(self.raw_data))
            curr_time = datetime.datetime.now()
            ens_file_name = '../results/ensemble/' + self.version + '.pkl'
            result_pred = [
                {'question': self.raw_data[ix], 'confidence': preds[ix]} for ix in range(len(self.raw_data))
            ]
            pickle.dump(result_pred, open(ens_file_name, 'wb+'), protocol=-1)

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
            # if flag and epoch == 4:
                # print(re)
                # print(gt)
                # print()
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
    to load test data
    '''
    def __init__(self, file_dir, name_to_com, id_to_com, max_length=160, threshold=0.5, result_dir=None, version=None):
        super().__init__()
        self.is_test = True
        self.version = version
        self.pred_list = []
        self.test_file_name = file_dir
        self.max_length = max_length
        self.name_to_com = name_to_com
        self.id_to_com = id_to_com
        self.threshold = threshold
        self.result_dir = result_dir
        querys = desolve_str_to_list(file_dir)
        self.raw_data, self.data = test_data_prepare(querys, id_to_com, TOKENIZER, max_length)
        self._len = self.data['ids'].shape[0]
        print('data size:', self._len)
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


class InferData(BaseData):
    '''
    when deploy, preocess the inference request.
    '''
    def __init__(self, querys, name_to_com, id_to_com, max_length=160, threshold=0.5, result_dir=None):
        super().__init__()
        self.pred_list = []
        self.max_length = max_length
        self.name_to_com = name_to_com
        self.id_to_com = id_to_com
        self.threshold = threshold
        self.result_dir = result_dir
        self.raw_data, self.data = test_data_prepare(querys, id_to_com, TOKENIZER, max_length)
        self._len = self.data['ids'].shape[0]
        print('data size:', self._len)
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
