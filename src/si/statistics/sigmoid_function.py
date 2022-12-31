import numpy as np

def sigmoid_function(X: np.ndarray) -> np.ndarray:
    """
    It calculates the sigmoid function of the given input.
    :param X: np.ndarray, the input of the sigmoid function
    :return: np.ndarray, the sigmoid function of the given input
    """
    return 1 / (1 + np.exp(-X))

