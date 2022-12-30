import numpy as np

from si.data.dataset_module import Dataset

def sigmoid_function(X: np.ndarray) -> np.ndarray:
    '''
    Método que calcula a função sigmoide para X de input.
    :param X: np.ndarray, valores de input
    :return: np.ndarray, função sigmoid dado X (A probabilidade dos valores serem iguais a 1)
    '''
    return 1 / (1 + np.exp(-X))

