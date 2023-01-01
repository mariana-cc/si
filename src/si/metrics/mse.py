import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the Mean Squared Error (MSE) of the model on the given dataset.
    MSE = sum[(y_true - y_pred)^2] / 2N , N = total number of samples

    :param y_true: np.ndarray, true labels (y values) of the dataset
    :param y_pred: np.ndarray, predicted labels (y values) of the dataset
    :return: float, the MSE value of the model
    """
    return np.sum((y_true - y_pred) ** 2) / (len(y_true) * 2)
    # N =  total number of samples = len(y_true) or y_true.shape[0]

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    It returns the derivative of the mean squared error for the y_pred variable.
    :param y_true: np.ndarray, true labels (y values) of the dataset
    :param y_pred: np.ndarray, predicted labels (y values) of the dataset
    :return: np.ndarray, the derivative of the MSE of the model
    """
    return -2 * (y_true - y_pred) / (len(y_true) * 2)

if __name__ == "__main__":
    true_values = np.array([1, 0, 0, 1, 0, 1])
    pred_values = np.array([1, 0, 1, 0, 1, 1])
    # MSE
    print(f"The MSE value for the model is: {mse(true_values, pred_values):.4f}")
    # derivative of MSE
    print(f"The derivative of the MSE value for the model is: {mse_derivative(true_values, pred_values)}")
