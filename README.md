# 大语言模型最佳实践

大语言模型(Large Language Models，LLMs) 基于Transformer架构,
Transformer 是一种专为处理序列数据（如文本）而设计的神经网络架构,
其核心组件包括自注意力机制（Self-Attention Mechanism）和前馈神经网络（Feed-Forward Neural Network）

## 1. 理解原理
___
模型中定义了一组庞大的向量参数,准备一组数据进行训练,数据包含输入和输出,将输入和输出不断的传给模型,让模型去调整向量参数来完成训练

1. 文本预处理
   - Tokenization(分词): 将文本转换成Token数组
   - Embedding(词嵌入): 将Token转换成一个向量,用来表示词语之间的关系
   - Positional Encoding(位置编码): 保存词语在句子中的位置
2. 编码器
   - 编码器有多个层,层数越多深度越大
   - 通过多头注意力机制将输入数据`src`进行处理,并对处理的结果进行`dropout`
   - 将输入的`src`和处理过的`src`相加并且进行`LayerNorm`层归一化
   - 通过前馈神经网络对`src`进行处理,并对处理的结果进行`dropout`
   - 将处理之前的`src`和处理过的`src`相加并且进行`LayerNorm`层归一化
