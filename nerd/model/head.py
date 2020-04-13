import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .ops import GeLU 
from .fc import MLP

BertLayerNorm = torch.nn.LayerNorm

class EntityFeats(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.first_ent_emb = config.first_ent_emb
        self.sum_ent_emb = config.sum_ent_emb
        self.sum_sent_emb = config.sum_sent_emb
        self.cls_pooled = config.cls_pooled
        self.hidden_size = config.hidden_size
        num = 1 if self.cls_pooled else 0
        if self.first_ent_emb:
            num += 1
            self.first_proj = MLP(
                self.hidden_size, 
                self.hidden_size * 4,
                self.hidden_size,
                0.1,
                eval(config.activation)())
        if self.sum_ent_emb:
            num += 1
            self.sum_proj = MLP(
                self.hidden_size, 
                self.hidden_size * 4,
                self.hidden_size,
                0.1,
                eval(config.activation)())
        if self.sum_sent_emb:
            num += 1
            self.sent_proj = MLP(
                self.hidden_size, 
                self.hidden_size * 4,
                self.hidden_size,
                0.1,
                eval(config.activation)())
        self.mix = nn.Linear(self.hidden_size * num, self.hidden_size)
        self.bn = BertLayerNorm(self.hidden_size)
        # self.config = config
        
    def forward(self, 
            embs, 
            cls_pooled, 
            sent_mask,  
            entity_pos, 
            entity_mask):
        feats = []
        masked_sent = embs * sent_mask.unsqueeze(-1).float()
        if self.first_ent_emb:
            x = torch.gather(masked_sent, 1, entity_pos.view(-1, 1, 1).expand_as(masked_sent[:,0:1,:]))
            ent_emb_first = self.first_proj(
                x.view(
                    -1, self.hidden_size
                )
            )
            feats.append(ent_emb_first)
        if self.sum_ent_emb:
            ent_emb_sum = self.sum_proj(
                torch.mean(masked_sent * entity_mask.unsqueeze(-1), -2).view(
                    -1, self.hidden_size
                )
            )
            feats.append(ent_emb_sum)
        if self.sum_sent_emb:
            sent_emb_sum = self.sent_proj(
                torch.mean(masked_sent, -2).view(
                    -1, self.hidden_size
                )
            )
            feats.append(sent_emb_sum)
        if self.cls_pooled:
            feats.append(cls_pooled)
        ent_emb = torch.cat(feats, -1)
        ent_emb = self.bn(self.mix(ent_emb))
        return ent_emb


class SimiFuse(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bias_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.fc = MLP(
            config.hidden_size, 
            config.hidden_size * 2,
            config.hidden_size,
            0.1,
            eval(config.activation)()
        )
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = eval(config.activation)()

    def forward(self, feats, entity_id):
        tar_b = self.bias_embeddings(entity_id).view(
            -1, 
            self.config.hidden_size
        )
        x = tar_b * feats
        x = self.fc(x) + self.act(self.linear(feats))
        return x
 

class NerdSimilarity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.simi = SimiFuse(config)
        self.bn = BertLayerNorm(config.hidden_size)
        # self.fc = nn.Linear(config.hidden_size, 1)
        self.fc = MLP(
            config.hidden_size, 
            config.hidden_size * 2,
            1,
            0.1,
            eval(config.activation)()
        )

    def forward(self, feats, entity_id):
        output = self.simi(feats, entity_id)
        output = self.bn(output)
        output = self.fc(output)
        return output


class NerdHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.extractor = EntityFeats(config)
        self.simi = NerdSimilarity(config)

    def forward(self, 
            embs, 
            cls_pooled, 
            sent_mask, 
            entity_id, 
            entity_pos, 
            entity_mask):
        assert len(embs.shape) == 3
        assert len(cls_pooled.shape) == 2
        feats = self.extractor(
            embs, 
            cls_pooled, 
            sent_mask,  
            entity_pos, 
            entity_mask)
        output = self.simi(feats, entity_id)
        return output