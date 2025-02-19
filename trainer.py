import os.path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from config import MODEL_NAME


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, tokenizer, batch_size=32, learning_rate=1e-4, device='cuda'):
        self.device = device  # 设备（如 'cuda' 或 'cpu'）
        self.model = model.to(self.device)  # 将模型移动到指定设备（如 GPU）
        self.tokenizer = tokenizer  # 分词器
        self.save_path = f'/builds/{MODEL_NAME}.pth'
        self.start_epoch = 0

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 创建训练数据加载器
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 创建验证数据加载器

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # 定义损失函数为交叉熵损失

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)  # 定义优化器为 Adam
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.95)  # 定义学习率调度器

        self.scaler = GradScaler()  # AMP混合精度

    def train_epoch(self):
        self.model.train()  # 设置模型为训练模式
        total_loss = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)  # 将输入和目标移动到指定设备
            self.optimizer.zero_grad()  # 清空梯度

            # 混合精度计算
            with autocast():
                output = self.model(inputs, targets)  # 前向传播
                loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))  # 计算损失

            self.scaler.scale(loss).backward()  # 反向传播
            self.scaler.step(self.optimizer)  # 更新模型参数
            self.scaler.update()

            total_loss += loss.item()  # 累加损失

        avg_loss = total_loss / len(self.train_loader)

        self.scheduler.step()  # 调整学习率

        return avg_loss  # 返回平均损失

    def evaluate(self):
        self.model.eval()  # 设置模型为评估模式
        total_loss = 0
        correct = 0
        total_samples = 0
        with torch.no_grad():  # 禁用梯度计算
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)  # 将输入和目标移动到指定设备
                outputs = self.model(inputs)  # 前向传播
                loss = self.criterion(outputs, targets)  # 计算损失
                total_loss += loss.item() * inputs.size(0)  # 累加损失

                # 计算准确率
                predictions = outputs.argmax(dim=1)  # 取最大值的索引作为预测类别
                correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / total_samples  # 计算平均损失
        accuracy = correct / total_samples  # 计算准确率
        return avg_loss, accuracy

    def save_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.save_path)

        print(f"模型参数和训练参数已保存至 {self.save_path}")

    def train(self, epochs=10, recover=False):
        if not recover:
            checkpoint = torch.load(self.save_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1

        for epoch in range(epochs):
            self.start_epoch += epoch
            train_loss = self.train_epoch()  # 训练一个轮次
            val_loss, accuracy = self.evaluate()  # 验证一个轮次
            self.save_model(self.start_epoch)  # 保存模型
            print(
                f'{self.start_epoch}/{self.start_epoch + epochs}: 训练损失 {train_loss:.4f}, 验证损失: {val_loss:.4f}, 准确率: {accuracy:.4f}')

        print("训练完成！")
