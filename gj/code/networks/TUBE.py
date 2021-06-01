import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

from dataset import START, PAD

class TUBEPosBias(nn.Module):
    def __init__(self, q_channels, k_channels, head_num=8, dropout=0.1):
        super(TUBEPosBias, self).__init__()

        self.q_channels = q_channels
        self.k_channels = k_channels
        self.head_dim = q_channels // head_num
        self.head_num = head_num

        self.q_linear = nn.Linear(q_channels, self.head_num * self.head_dim)
        self.k_linear = nn.Linear(k_channels, self.head_num * self.head_dim)

        self.temperature = 2 * (self.head_num * self.head_dim) ** 0.5

    def forward(self, q, k):
        # attn_bias [B, HEAD_NUM, Q_LEN, K_LEN]
        b, q_len, k_len = q.size(0), q.size(1), k.size(1)
        q = (
            self.q_linear(q)
            .view(b, q_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        ) # [B, HEAD_NUM, Q_LEN, HEAD_DIM]
        k = (
            self.k_linear(k)
            .view(b, k_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )


        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature

        return attn

