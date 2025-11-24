import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            if isinstance(weight, list):
                self.weight =  torch.tensor(weight)
            elif isinstance(weight, torch.Tensor):
                self.weight = weight.clone().detach()
            elif isinstance(weight, np.ndarray):
                self.weight = torch.from_numpy(weight)
        else:
            self.weight = None

    def forward(self, inputs, targets):
        """
        :param inputs: logits, shape [N, C, H, W]
        :param targets: one-hot labels, shape [N, C, H, W]
        """
        # softmax over channels
        probs = F.softmax(inputs, dim=1)
        log_probs = torch.log(probs + 1e-8)  # log_softmax, 同时防止 log(0)

        # focal loss core: -weight * (1 - p_t)^gamma * log(p_t)
        focal_term = (1 - probs) ** self.gamma

        # class weights weight: shape [C,] -> reshape for broadcasting
        if self.weight is not None:
            weight = self.weight.to(inputs.device).view(1, -1, 1, 1)
            log_probs = log_probs * weight
        loss = - targets * focal_term * log_probs
        loss = loss.sum(dim=1)  # sum over classes => [N, H, W]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # shape [N, H, W]


if __name__ == "__main__":
    # 假设 targets 是整数标签
    labels = torch.randint(0, 4, (1, 3, 3, 3)).float().cuda()
    print("lables shape : ",labels.shape)
    print("lables : ",labels)
    inputs = torch.exp(labels + torch.randn(labels.size(),dtype=torch.float32).cuda()).cuda()
    print("inputs shape : ",inputs.shape)
    print("inputs : ",inputs)


    loss_fn = FocalLoss(gamma=2.0, weight=[0.1, 0.8, 0.1]).cuda()
    loss = loss_fn(inputs, labels)
    print("focal loss : ",loss)
