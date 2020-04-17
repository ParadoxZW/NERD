import numpy as np
import torch
from torch import nn

from .head import NerdHead
from .ops import init_weights
from transformers import BertModel

class NerdEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
    
    def init(self):
        self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def forward(self, sent, sent_mask, token_type=None):
        if token_type is None:
            output = self.bert(sent, attention_mask=sent_mask)
        else:
            output = self.bert(sent, attention_mask=sent_mask, token_type_ids=token_type)
        return output


class NerdModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.encoder = NerdEncoder()
        self.cls_head = NerdHead(config)
        # self.cls_head.apply(init_weights)

    def init(self):
        self.cls_head.apply(init_weights)
        self.encoder.init()

    def forward(self, sent, sent_mask, entity_id, entity_pos, entity_mask, token_type=None):
        embs, cls_pooled = self.encoder(sent, sent_mask, token_type)
        output = self.cls_head(
            embs, 
            cls_pooled, 
            sent_mask, 
            entity_id, 
            entity_pos, 
            entity_mask
        )
        return output
