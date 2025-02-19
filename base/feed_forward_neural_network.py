import torch.nn as nn

from config import FEATURE_DIMENSION, FEEDFORWARD_DIMENSION, DROPOUT_PROBABILITY


# 前馈神经网络
class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(FEATURE_DIMENSION, FEEDFORWARD_DIMENSION),
            nn.ReLU(),
            nn.Dropout(DROPOUT_PROBABILITY),
            nn.Linear(FEEDFORWARD_DIMENSION, FEATURE_DIMENSION)
        )

    def forward(self, x):
        return self.ff(x)
