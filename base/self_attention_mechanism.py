import math
import torch
import torch.nn as nn

from config import FEATURE_DIMENSION, HEAD_COUNT, EXPERT_COUNT


# 多头自注意力机制
class SelfAttentionMechanism(nn.Module):
    def __init__(self):
        super(SelfAttentionMechanism, self).__init__()
        self.d_k = FEATURE_DIMENSION // HEAD_COUNT

        # 查询、键、值的线性映射
        self.q_linear = nn.Linear(FEATURE_DIMENSION, FEATURE_DIMENSION)
        self.k_linear = nn.Linear(FEATURE_DIMENSION, FEATURE_DIMENSION)
        self.v_linear = nn.Linear(FEATURE_DIMENSION, FEATURE_DIMENSION)
        self.expert_heads = nn.ModuleList(
            [nn.Linear(FEATURE_DIMENSION, FEATURE_DIMENSION) for _ in range(EXPERT_COUNT)])
        self.out_linear = nn.Linear(FEATURE_DIMENSION, FEATURE_DIMENSION)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 计算 Q、K、V
        Q = self.q_linear(query).view(batch_size, -1, HEAD_COUNT, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, HEAD_COUNT, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, HEAD_COUNT, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重并加权 V
        attention = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attention, V).transpose(1, 2).contiguous().view(batch_size, -1, FEATURE_DIMENSION)

        # 在这里，我们引入了专家选择
        expert_outputs = [expert(output) for expert in self.expert_heads]
        expert_output = torch.stack(expert_outputs, dim=-1).mean(dim=-1)  # 将多个专家的输出取平均

        # 线性映射输出
        output = self.out_linear(expert_output)
        return output
