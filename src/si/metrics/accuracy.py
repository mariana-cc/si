import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It calculates the error of the model on the given dataset, following the accuracy formula:
    (VN + VP) / (VN + VP + FP + FN)
    :param y_true: np.ndarray, true labels (y values) of the dataset
    :param y_pred: np.ndarray, predicted labels (y values) of the dataset
    :return: float, accuracy of the model (error between the true labels and the predicted labels.
    """
    return np.sum(y_true == y_pred) / len(y_true)


if __name__ == "__main__":
    true_values = np.array([1, 0, 0, 1, 0, 1])
    pred_values = np.array([1, 0, 1, 0, 1, 1])
    print(f"The accuracy score for the model is: {accuracy(true_values, pred_values) * 100:.2f}%")