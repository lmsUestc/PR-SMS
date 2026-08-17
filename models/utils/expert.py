import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Expert(nn.Module):
    def __init__(self, num_experts=3, token_dim=768, num_classes=10, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.token_dim = token_dim
        self.temperature = temperature
        self.last_tem = temperature
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Gumbel-softmax 分类网络，用于产生 N 个概率
        self.selector = nn.Sequential(
            nn.Linear(num_experts * token_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )
        
        self.fc = nn.Linear(num_experts * token_dim, 500)
        self.classifier = nn.Linear(1000, num_classes)
        
    def get_temperature(self, epoch, tau_0=5.0, tau_min=0.5, total_epochs=50):
        # print('epoch:', epoch)
        return tau_min + 0.5 * (tau_0 - tau_min) * (1 + np.cos(np.pi * epoch / total_epochs))
    
    def get_fc(self, class_tokens, epoch):
        device = next(self.parameters()).device
        class_tokens = class_tokens.to(device)
        B, N, D = class_tokens.shape
        assert N == self.num_experts and D == self.token_dim, "输入维度不匹配"

        # 拼接所有 class token 并送入 selector
        # fused_token = class_tokens.sum(dim=1)  # [B, 768]
        flattened_tokens = class_tokens.reshape(class_tokens.size(0), -1)
        selector_logits = self.selector(flattened_tokens)  # [B, N]

        # 使用 Gumbel Softmax 得到概率分布
        if epoch != -1:
            self.temperature = self.get_temperature(epoch)
            self.last_tem = self.temperature
        else:
            self.temperature = self.last_tem
        
        gumbel_probs = F.gumbel_softmax(selector_logits, tau=self.temperature, hard=False, dim=1)  # [B, N]

        # 先扩展概率维度：[B, N, 1]
        weighted_tokens = class_tokens * gumbel_probs.unsqueeze(-1)  # [B, N, D]

        # 然后把 N 个 token 拼接成一个大向量：[B, N*D]
        fused_token = weighted_tokens.reshape(weighted_tokens.size(0), -1)  # [B, N*D]
        # print('fused:', fused_token.shape)
        fc = self.fc(fused_token)
        
        return fc

    def forward(self, class_tokens, best_token, epoch, returnt = 'out'):
        if returnt == 'out':
            device = next(self.parameters()).device
            class_tokens = class_tokens.to(device)
            B, N, D = class_tokens.shape
            assert N == self.num_experts and D == self.token_dim, "输入维度不匹配"

            # 拼接所有 class token 并送入 selector
            # fused_token = class_tokens.sum(dim=1)  # [B, 768]
            flattened_tokens = class_tokens.reshape(class_tokens.size(0), -1)
            selector_logits = self.selector(flattened_tokens)  # [B, N]

            # 使用 Gumbel Softmax 得到概率分布
            if epoch != -1:
                self.temperature = self.get_temperature(epoch)
                self.last_tem = self.temperature
            else:
                self.temperature = self.last_tem
        
            gumbel_probs = F.gumbel_softmax(selector_logits, tau=self.temperature, hard=False, dim=1)  # [B, N]

            # 先扩展概率维度：[B, N, 1]
            weighted_tokens = class_tokens * gumbel_probs.unsqueeze(-1)  # [B, N, D]

            # 然后把 N 个 token 拼接成一个大向量：[B, N*D]
            fused_token = weighted_tokens.reshape(weighted_tokens.size(0), -1)  # [B, N*D]
            # print('fused:', fused_token.shape)
            fc = self.fc(fused_token)
        elif returnt == 'fc':
            fc = class_tokens
        else:
            raise ValueError('Unknown returnt: {}'.format(returnt))
        
        alpha = torch.sigmoid(self.alpha)
        if isinstance(best_token, torch.Tensor):
            fused_feature = fc + alpha * (best_token - fc)
        else:
            fused_feature = alpha * fc + (1 - alpha) * fc  
            
        logits = self.classifier(torch.cat([fc, fused_feature], dim=1))
        return logits, fc

class ExpertSelector(nn.Module):
    def __init__(self, token_dim=500, max_experts=10, hidden_dim=256):
        super().__init__()
        print('max_experts:', max_experts)
        self.max_experts = max_experts
        self.token_dim = token_dim
        self.selector = nn.Sequential(
            nn.Linear(max_experts * token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_experts)
        )

    def forward(self, class_tokens, active_experts=None):
        device = next(self.parameters()).device
        class_tokens = class_tokens.to(device)
        B, N, D = class_tokens.shape
        assert N == active_experts and D == self.token_dim

        # Padding到 max_experts 个专家
        if N < self.max_experts:
            pad_len = self.max_experts - N
            padding = torch.zeros(B, pad_len, D, device=class_tokens.device, dtype=class_tokens.dtype)
            class_tokens = torch.cat([class_tokens, padding], dim=1)  # [B, max_experts, D]

        flattened = class_tokens.reshape(B, -1)  # [B, max_experts * D]
        logits = self.selector(flattened)  # [B, max_experts]

        weights = F.softmax(logits[:, :active_experts], dim=1)  # [B, N]
        return weights
    
# # 模拟数据
# B, N, D = 32, 3, 768
# tokens = torch.randn(B, N, D, requires_grad=True).cuda()
# student = torch.randn(B, D).cuda()
# selector = ExpertSelector(max_experts=5, token_dim=D).cuda()

# # 前向传播
# weights = selector(tokens, active_experts=N)  # [B, N]
# soft_teacher = torch.sum(tokens * weights.unsqueeze(-1), dim=1)  # [B, D]
# loss = F.mse_loss(soft_teacher, student)
# loss.backward()

# # 打印梯度
# for name, param in selector.named_parameters():
#     print(f"{name}: grad = {param.grad.norm() if param.grad is not None else None}")