import numpy as np


class CenterLoss:
    def __init__(self, x, y):
        self.has_params = False
        self.centers = self.center(x, y)
        self.x = x

    def center(self, x, y):
        return np.sum(x * y, axis=0, keepdims=True) / np.sum(y, axis=0, keepdims=True)

    def forward(self):
        """
    Args:
      x: ndarray of shape (N, num_classes).
      y: ndarray of shape (N, num_classes).
    Returns:
      Scalar, average loss over N examples.
    """
        return (1 / 2) * np.sum((self.x - self.centers) ** 2)

    def backward_inputs(self):
        """
    Args:
      x: ndarray of shape (N, num_classes).
      y: ndarray of shape (N, num_classes).
    Returns:
        ndarray of shape (N, num_classes).
    """
        return self.x - self.centers
