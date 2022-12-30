import numpy as np

def euclidean_distance(x : np.ndarray,y :np.ndarray) -> np.ndarray:
    """
    It computes the Euclidean distances between a given point X and all points in a array of vectors y.
    Formulation: distance_y1n = np.sqrt((x1 - y11)^2 + (x2 - y12)^2 + ... + (xn - y1n)^2)
    :param x: np.ndarray, one dimension array (1 sample / 1 row)
    :param y: np.ndarray, two dimensions array (n sample / n row)
    :return: np.ndarray, a array with the distances calculated
    """
    return np.sqrt(((x-y) ** 2).sum(axis=1))
    #each point of x is compared to each point in y
    #axis = 1 because its made by one column at a time