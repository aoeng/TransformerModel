import torch.nn as nn

from config import DROPOUT_PROBABILITY, FEATURE_DIMENSION
from feed_forward_neural_network import FeedForwardNeuralNetwork
from self_attention_mechanism import SelfAttentionMechanism


# 定义Transformer模型的Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super(TransformerEncoderLayer, self).__init__()
        self.sam = SelfAttentionMechanism()
        self.ffn = FeedForwardNeuralNetwork()
        self.norm1 = nn.LayerNorm(FEATURE_DIMENSION)
        self.norm2 = nn.LayerNorm(FEATURE_DIMENSION)
        self.dropout = nn.Dropout(DROPOUT_PROBABILITY)

    def forward(self, src, mask=None):
        # 自注意力残差连接 + 层归一化
        src = self.norm1(src + self.dropout(self.sam(src, src, src, mask)))
        # 前馈残差连接 + 层归一化
        src = self.norm2(src + self.dropout(self.ffn(src)))
        return src
