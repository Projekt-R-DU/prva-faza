import numpy as np

"""
    Implementation: Make class instance before training or every time before starting a new epoch.        
"""

class CenterLoss:
    def __init__(self, x, y):
        self.has_params = False
        self.centers = self.center(x, y)

    def center(self, x, y):
        return np.sum(x * y, axis=0, keepdims=True) / np.sum(y, axis=0, keepdims=True)

    def forward(self, x):
        """
    Args:
      x: ndarray of shape (N, num_classes).
      y: ndarray of shape (N, num_classes).
    Returns:
      Scalar, average loss over N examples.
    """
        return (1 / 2) * np.sum((x - self.centers) ** 2)

    def backward_inputs(self, x):
        """
    Args:
      x: ndarray of shape (N, num_classes).
      y: ndarray of shape (N, num_classes).
    Returns:
        ndarray of shape (N, num_classes).
    """
        return x - self.centers
