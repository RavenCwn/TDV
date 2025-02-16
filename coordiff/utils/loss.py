import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2):
        """
        alpha: 正样本权重 (这里设置为0.9给予稀有类更多关注)
        gamma: 调节难易样本的关注度，越大越关注困难样本
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        # alpha_t = self.alpha * targets + (1-self.alpha) * (1-targets)
        focal_loss = (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
    

class BalancedLoss(nn.Module):
    def __init__(self, pos_weight=9.0, smooth_eps=0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.smooth_eps = smooth_eps
        
    def forward(self, logits, targets):
        # 标签平滑
        smooth_targets = targets.float() * (1 - self.smooth_eps) + self.smooth_eps / 2
        
        # 计算加权BCE损失
        loss = F.binary_cross_entropy_with_logits(
            logits[:, 1], smooth_targets,
            pos_weight=torch.tensor([self.pos_weight]).to(logits.device)
        )
        return loss