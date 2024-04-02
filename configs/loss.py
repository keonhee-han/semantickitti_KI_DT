"""
Defines a CrossEntropyLoss class that inherits from torch.nn.Module.
Initializes the class with optional arguments for the ignore index (used to ignore specific labels during loss calculation) and class weights (for handling imbalanced datasets).
Implements the forward pass that calculates the cross-entropy loss for semantic segmentation.
Uses F.cross_entropy from torch.nn.functional for efficient cross-entropy calculation with optional weighting.
Ignores the loss for pixels with the specified ignore index (if set).
Calculates the mean loss across the non-ignored pixels.
"""

import torch
from torch.nn import functional as F
from functools import reduce


class CrossEntropyLoss(torch.nn.Module):
  """CrossEntropyLoss for semantic segmentation."""

  def __init__(self, ignore_index=255, weight=None):
    """
    Args:
      ignore_index (int, optional): Label value to be ignored during loss calculation. Defaults to 255.
      weight (torch.Tensor, optional): Class weights for imbalanced datasets. Defaults to None.
    """
    super(CrossEntropyLoss, self).__init__()
    self.ignore_index = ignore_index
    self.weight = weight

  def forward(self, outputs, targets):
    """
    Calculates the cross-entropy loss for semantic segmentation.

    Args:
      outputs (torch.Tensor): Model predictions with shape (B, C, H, W)
      targets (torch.Tensor): Ground truth labels with shape (B, C, H, W).
        where B is batch size, C is number of classes, H is height, and W is width.
    Returns:
      torch.Tensor: The calculated cross-entropy loss.
    """
    # targets = targets.argmax(dim=1)  # Convert one-hot to class indices
    loss = F.cross_entropy(outputs, targets.long(), reduction='none', weight=self.weight)
    # Ignore loss for ignored labels
    # if self.ignore_index is not None:
    #   mask = targets != self.ignore_index
    #   loss = loss[mask].mean()
    # else:
    #   loss = loss.mean()
    loss = loss.mean()

    ## Training & testing pixel accuracy of the trained MLP against the 2D pseudo-labels.
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == targets).sum().item()
    total = reduce(lambda x, y: x * y, targets.shape)
    accuracy = (correct / total) * 100
    # print(f"Pixel Accuracy: {accuracy}")

    return loss, accuracy