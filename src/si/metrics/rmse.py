import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It computes the Root-Mean-Square Error (RMSE) of the model on the given dataset.
    RMSE: sqrt((SUM[(ti - pi)^2]) / N).
    :param y_true: np.ndarray, true values of labels of the dataset
    :param y_pred: np.ndarray, predicted labels by a classifier
    :return: float, RMSE value
    """
    N = y_true.shape[0] # N represents the number of samples
    return np.sqrt(np.sum((y_true-y_pred)**2) / N) # RMSE formula

if __name__ == "__main__":
    true_values = np.array([1, 0, 0, 1, 0, 1])
    pred_values = np.array([1, 0, 1, 0, 1, 1])
    print(f"The RMSE value for the model is: {rmse(true_values, pred_values):.4f}")
