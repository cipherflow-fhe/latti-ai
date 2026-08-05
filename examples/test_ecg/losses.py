import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Loss Functions Module for ECG Classification.

This module implements custom loss functions including Focal Loss for handling class imbalance,
and provides a factory function to build different loss types (CrossEntropy, Weighted CrossEntropy,
Focal Loss) based on configuration parameters.
"""
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def build_loss(loss_name='weighted_ce', class_weights=None, focal_gamma=2.0):
    loss_name = loss_name.lower()

    if loss_name == 'ce':
        return nn.CrossEntropyLoss()

    if loss_name == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=class_weights)

    if loss_name == 'focal':
        return FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction='mean')

    raise ValueError(f'Not supported loss_name: {loss_name}')
