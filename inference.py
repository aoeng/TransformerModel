import torch
import torch.nn.functional as F


# 推理
class Inference:
    def __init__(self, model, tokenizer, device="cuda", max_length=512):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def greedy_decode(self, input_text, max_new_tokens=50):
        self.model.eval()
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                output = self.model(input_ids, input_ids)  # 输入和目标都是当前文本
                next_token_logits = output[:, -1, :]  # 取最后一个单词的预测分布
                next_token_id = torch.argmax(next_token_logits, dim=-1)  # 贪心搜索
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)

    def sample_top_k(self, input_text, k=10, max_new_tokens=50, temperature=1.0):
        self.model.eval()
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                output = self.model(input_ids, input_ids)
                next_token_logits = output[:, -1, :] / temperature
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                next_token_id = top_k_indices[torch.multinomial(F.softmax(top_k_logits, dim=-1), num_samples=1)]
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)

    def sample_top_p(self, input_text, p=0.9, max_new_tokens=50, temperature=1.0):
        self.model.eval()
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                output = self.model(input_ids, input_ids)
                next_token_logits = output[:, -1, :] / temperature

                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                cutoff = cumulative_probs > p
                sorted_logits[cutoff] = float("-inf")

                probs = F.softmax(sorted_logits, dim=-1)
                next_token_id = sorted_indices[torch.multinomial(probs, num_samples=1)]
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
