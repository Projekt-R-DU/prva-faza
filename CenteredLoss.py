import torch
import numpy as np


class CenterLoss:
    def __init__(self):
        self.has_params = False

    def centers(self, x, y):
        return np.sum(x * y, axis=0, keepdims=True) / np.sum(y, axis=0)

    def forward(self, x, y):
        """
    Args:
      x: ndarray of shape (N, num_classes).
      y: ndarray of shape (N, num_classes).
    Returns:
      Scalar, average loss over N examples.
    """
        return (1 / 2) * np.sum((x - self.centers(x, y)) ** 2)

    def backward_inputs(self, x, y):
        """
    Args:
      x: ndarray of shape (N, num_classes).
      y: ndarray of shape (N, num_classes).
    Returns:

    """
        return x - self.centers(x, y)
