import itertools
import numpy as np
from si.data.dataset_module import Dataset

class KMer:
    """
    A sequence descriptor that returns the k-mer composition of the sequence.
    """
    def __init__(self, k: int = 2):
        """
        Initializes the k-mer object.
        :param k: list of str, the k-mers
        """
        # Parameters
        self.k = k
        # Attributes
        self.k_mers = None

    def fit(self, dataset: Dataset):
        """
        It  fits the descriptor to the dataset.
        :param dataset: Dataset, dataset to fit the descriptor to
        :return: self, fitted descriptor
        """
        # generate the k-mers
        self.k_mers = [''.join(k_mer) for k_mer in itertools.product('ACTG', repeat=self.k)]
        return self

    def _get_sequence_k_mer_composition(self, sequence: str) -> np.ndarray:
        """
        It calculates the k-mer composition of the sequence.
        :param sequence: str, the sequence to calculate the k-mer composition for
        :return: list of float, the k-mer composition of the sequence
        """
        # calculate the k-mer composition
        counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i:i + self.k]
            counts[k_mer] += 1

        # normalize the counts
        return np.array([counts[k_mer] / len(sequence) for k_mer in self.k_mers])

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset.
        :param dataset: Dataset, dataset to transform
        :return: Dataset, transformed dataset
        """
        # calculate the k-mer composition
        sequences_k_mer_composition = [self._get_sequence_k_mer_composition(sequence)
                                       for sequence in dataset.X[:, 0]]
        sequences_k_mer_composition = np.array(sequences_k_mer_composition)

        # create a new dataset
        return Dataset(X=sequences_k_mer_composition, y=dataset.y, features=self.k_mers, label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits the descriptor to the dataset and transforms the dataset.
        :param dataset: Dataset, dataset to fit the descriptor to and transform
        :return: Dataset, transformed dataset
        """
        return self.fit(dataset).transform(dataset)


if __name__ == '__main__':
    from si.data.dataset_module import Dataset

    dataset_ = Dataset(X=np.array([['ACTGTTTAGCGGA', 'ACTGTTTAGCGGA']]),
                       y=np.array([1, 0]),
                       features=['sequence'],
                       label='label')

    k_mer_ = KMer(k=2)
    dataset_ = k_mer_.fit_transform(dataset_)
    print(dataset_.X)
    print(dataset_.features)
