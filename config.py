from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv(dotenv_path='.env.poem')

MODEL_NAME = os.getenv('MODEL_NAME', 'local')  # 模型文件的名称
VOCAB_SIZE = os.getenv("VOCAB_SIZE", 5000)  # 控制输入的词汇表大小
FEATURE_DIMENSION = os.getenv("FEATURE_DIMENSION", 512)  # 输入和输出的向量的维度
FEEDFORWARD_DIMENSION = os.getenv("FEEDFORWARD_DIMENSION", 2048)  # 前馈神经网络维度(一般为向量维度的4倍)
HEAD_COUNT = os.getenv("HEAD_COUNT", 8)  # 多头注意力的数量
EXPERT_COUNT = os.getenv("EXPERT_COUNT", 4)  # 每层专家的数量
LAYER_COUNT = os.getenv("LAYER_COUNT", 6)  # 神经网络中层的数量
DROPOUT_PROBABILITY = os.getenv("DROPOUT_PROBABILITY", 0.1)  # 控制Dropout的概率 防止过拟合
MAX_RELATIVE_POSITION = os.getenv("MAX_RELATIVE_POSITION", 32)