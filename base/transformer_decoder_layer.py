import torch.nn as nn

from config import FEATURE_DIMENSION, DROPOUT_PROBABILITY
from feed_forward_neural_network import FeedForwardNeuralNetwork
from self_attention_mechanism import SelfAttentionMechanism


# 定义Transformer模型的Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self):
        super(TransformerDecoderLayer, self).__init__()
        self.sam1 = SelfAttentionMechanism()
        self.sam2 = SelfAttentionMechanism()
        self.ffn = FeedForwardNeuralNetwork()
        self.norm1 = nn.LayerNorm(FEATURE_DIMENSION)
        self.norm2 = nn.LayerNorm(FEATURE_DIMENSION)
        self.norm3 = nn.LayerNorm(FEATURE_DIMENSION)
        self.dropout = nn.Dropout(DROPOUT_PROBABILITY)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 有遮蔽的自注意力残差连接 + 层归一化
        tgt = self.norm1(tgt + self.dropout(self.sam1(tgt, tgt, tgt, tgt_mask)))
        # 编码器-解码器注意力残差连接 + 层归一化
        tgt = self.norm2(tgt + self.dropout(self.sam2(tgt, memory, memory, memory_mask)))
        # 前馈残差连接 + 层归一化
        tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))
        return tgt
