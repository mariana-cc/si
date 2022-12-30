import numpy as np
from si.data.dataset_module import Dataset
from typing import Tuple
def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int=42) -> Tuple[Dataset, Dataset]:
    """
    It splits the dataset in training and testing sets of data.
    :param dataset: Dataset, dataset to split
    :param test_size: float, propotion of the dataset to include in the test split
    :param random_state: int, seed of the random number generator to use in the split
    :return: Tuple[Dataset, Dataset]
        train: Dataset, training dataset
        test: Dataset, testing dataset
    """
    # set random state
    np.random.seed(random_state)

    # get dataset size
    n_samples = dataset.shape()[0]

    # get number of samples in the test set
    n_test = int(n_samples * test_size)

    # get the dataset permutations
    permutations = np.random.permutation(n_samples)

    # get samples in the test set
    test_idxs = permutations[:n_test] # data until the number of samples defined is reached

    # get samples in the train set
    train_idxs = permutations[n_test:] # remaining data goes to the train set

    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


if __name__ == "__main__":
    new_dataset = Dataset.from_random(n_samples=5, n_features=5, label=None)
    train_ds, test_ds = train_test_split(dataset=new_dataset, test_size=0.3, random_state=0)
    print(train_ds.X.shape, train_ds.y.shape, test_ds.X.shape, test_ds.y.shape)
