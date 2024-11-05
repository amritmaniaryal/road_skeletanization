"""Losses and evaluation metrics for road-skeleton segmentation."""

import torch
import torch.nn as nn


def dice_loss(pred, target, eps=1e-6):
    """Soft Dice loss. pred/target are probability maps in [0, 1]."""
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    intersection = (pred * target).sum(1)
    return (1.0 - (2.0 * intersection + eps) / (pred.sum(1) + target.sum(1) + eps)).mean()


class BCEWithDice(nn.Module):
    """Binary cross-entropy + soft Dice, the standard segmentation combo."""

    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice_weight * dice_loss(pred, target)


def mse(pred, target):
    return nn.functional.mse_loss(pred, target)


def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)


def iou(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = (pred + target).clamp(0, 1).sum()
    return (intersection + eps) / (union + eps)


def skeleton_metrics(pred, target, tol=2):
    """Skeleton-aware precision/recall.

    A predicted skeleton pixel counts as a true positive if a ground-truth
    skeleton pixel lies within a Chebyshev distance of `tol`. This tolerates
    the 1-2 px drift common in skeleton prediction while still punishing
    spurious pixels.

    Returns (precision, recall, f1).
    """
    import torch.nn.functional as F

    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    if target.sum() == 0 and pred.sum() == 0:
        return 1.0, 1.0, 1.0
    if target.sum() == 0 or pred.sum() == 0:
        return 0.0, 0.0, 0.0

    kernel_size = 2 * tol + 1
    pred_dilated = F.max_pool2d(pred, kernel_size=kernel_size, stride=1, padding=tol)
    target_dilated = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=tol)

    # fraction of predicted pixels that lie near a GT pixel
    tp = (pred * target_dilated).sum()
    precision = tp / pred.sum()
    # fraction of GT pixels that lie near a predicted pixel
    tp_gt = (target * pred_dilated).sum()
    recall = tp_gt / target.sum()
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(precision), float(recall), float(f1)
