import math
import torch
import torch.nn as nn

from config import FEATURE_DIMENSION


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, FEATURE_DIMENSION)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, FEATURE_DIMENSION, 2).float() * -(math.log(10000.0) / FEATURE_DIMENSION))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加 batch 维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()
