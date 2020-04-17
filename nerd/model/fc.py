import numpy as np
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., act=lambda x: x):
        super(MLP, self).__init__()

        self.linear_in = nn.Linear(in_size, mid_size)
        self.act = act
        self.linear_out = nn.Linear(mid_size, out_size)

    def forward(self, x):
        out = self.linear_out(
            self.act(self.linear_in(x))
        )
        if x.shape[-1] == out.shape[-1]:
            return x + out
        else:
            return out