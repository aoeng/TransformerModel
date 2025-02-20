from inference import Inference
from model import Model
from trainer import Trainer

train_dataset = MyDataset("train_data.txt")
val_dataset = MyDataset("val_data.txt")
tokenizer = Tokenizer()

model = Model()

# 训练
trainer = Trainer(model, train_dataset, val_dataset, tokenizer)
trainer.train(10)

# 推理
inference = Inference(model, tokenizer)
# 生成文本
input_text = "从前有一个勇敢的骑士，他"
output_text = inference.sample_top_p(input_text, p=0.9)
print(output_text)
