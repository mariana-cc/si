from si.data.dataset_module import Dataset
import numpy as np
from typing import Tuple
from scipy import stats

def f_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scoring function that computes one-way ANOVA test for the given dataset, obtaining the F-scores for each feature.
    :param dataset: Dataset, a labeled dataset object
    :return: F : np.array, F scores
            p : np.array, p-values
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    F, p = stats.f_oneway(*groups)
    return F,p


if __name__ == "__main__":
    new_dataset = Dataset(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [4, 3, 7]]), np.array([1, 1, 0, 0]))
    print(f_classification(new_dataset))