import torch
import torch.nn as nn


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_relative_position=32):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position

        # 相对位置嵌入矩阵（可学习参数）
        self.relative_embeddings = nn.Embedding(2 * max_relative_position + 1, d_model)

    def forward(self, length):
        # 计算相对位置
        positions = torch.arange(length).unsqueeze(0) - torch.arange(length).unsqueeze(1)
        positions = positions.clamp(-self.max_relative_position, self.max_relative_position)
        positions += self.max_relative_position  # 确保索引为正

        # 获取相对位置嵌入
        relative_position_encoding = self.relative_embeddings(positions)

        return relative_position_encoding
