class NegativeLogLikelihood():
    def __init__(self):
        self.has_params = False

    def softmax(self, x):
        x_exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x_exp_shifted / np.sum(x_exp_shifted, axis=1, keepdims=True)

    def forward(self, x, y):
        """
    Args:
      x: ndarray of shape (N, num_classes).
      y: ndarray of shape (N, num_classes).
    Returns:
      Scalar, average loss over N examples.
    """
        probs = self.softmax(x)
        return -np.mean(np.log(np.sum(probs * y, axis=1)))

    def backward_inputs(self, x, y):
        """
    Args:
      x: ndarray of shape (N, num_classes).
      y: ndarray of shape (N, num_classes).
    Returns:
      ndarray of shape (N, num_classes).
    """
        probs = self.softmax(x)
        return (1/probs.shape[0]) * (probs - y)