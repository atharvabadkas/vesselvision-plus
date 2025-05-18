import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # Ensure features have shape [batch_size, n_views, feature_dim]
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        
        batch_size = features.shape[0]
        
        # Normalize feature vectors
        features = F.normalize(features, p=2, dim=2)
        
        # Compute logits
        anchor_dot_contrast = torch.bmm(features, features.transpose(1, 2)) / self.temperature
        
        # For stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=2, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create masks for positive and negative pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.transpose(0, 1)).float().to(features.device)
        
        # Subtract self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(features.device)
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss

# Enhanced Focal Loss implementation
class EnhancedFocalLoss(nn.Module):

    def __init__(self, gamma=2.5, alpha=0.25, reduction='mean'):
        super(EnhancedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Apply focal weighting
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha balancing
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss 