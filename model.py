import torch
from torch import nn

from base.positional_encoding import PositionalEncoding
from base.transformer_decoder_layer import TransformerDecoderLayer
from base.transformer_encoder_layer import TransformerEncoderLayer
from config import LAYER_COUNT, FEATURE_DIMENSION, VOCAB_SIZE


# 专家Transformer模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, FEATURE_DIMENSION)
        self.positional_encoding = PositionalEncoding()

        # 构建 Transformer 层，每层包括自注意力、多专家和前馈神经网络
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer()
            for _ in range(LAYER_COUNT)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer()
            for _ in range(LAYER_COUNT)
        ])

        self.output_layer = nn.Linear(FEATURE_DIMENSION, VOCAB_SIZE)

    def forward(self, input_text, target_text=None, input_mask=None, target_mask=None):
        # 嵌入和位置编码
        input_tensor = self.positional_encoding(self.embedding(input_text))
        if target_text is not None:
            target_tensor = self.positional_encoding(self.embedding(target_text))
        else:
            # 在推理阶段，初始化目标序列为开始标记（假设为[0]）
            target = torch.zeros_like(input_tensor[:, :1])  # 假设开始标记的索引为0
            target_tensor = self.positional_encoding(self.embedding(target))

        # 编码器部分
        memory = input_tensor
        for layer in self.encoder_layers:
            memory = layer(memory, input_mask)

        # 解码器部分
        for layer in self.decoder_layers:
            target_tensor = layer(target_tensor, memory, target_mask, input_mask)

        # 输出层
        output = self.output_layer(target_tensor)
        return output
