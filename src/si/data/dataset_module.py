from typing import Tuple, Sequence
import numpy as np
import pandas as pd


class Dataset:
    """
    It represents a dataset
    """
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: list = None, label: str = None):
        """
        It initializes the dataset.

        :param X: numpy.ndarray, features matrix (n_samples, n_features) - independent variables
        :param y: np.ndarray, label vector (n_samples, 1) - dependent variable
        :param features: list of str, features names (n_features)
        :param label: str, label name
        """
        if X is None:
            raise ValueError("X cannot be None")

        if features is None:
            features = [str(i) for i in range(X.shape[1])]
        else:
            features = list(features)

        if y is not None and label is None:
            label = "y"

        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        It returns the shape of the dataset.
        :return: Tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        It verifies if the dataset has a label and returns True.
        Returns Boolean
        -------
        bool
        """
        if self.y is not None:
            return True
        else:
            return False

    def get_classes(self) -> np.ndarray:
        """
        It returns the unique classes (y) in the dataset.
        :return: np.ndarray
        """
        if self.has_label():  # if it has y vector (from has_label)
            return np.unique(self.y)  # returns the number of classes
        else:
            raise ValueError("Dataset does not have y")

    def get_mean(self) -> np.ndarray:
        """
        It returns the mean of each feature.
        :return: np.ndarray
        """
        if self.X is None:
            return
        else:
            return np.mean(self.X, axis=0)  # for each feature so axis=0

    def get_variance(self) -> np.ndarray:
        """
        It returns the variance of each feature.
        :return: ndarray
        """
        if self.X is None:
            return
        else:
            return np.var(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        It returns the median of each feature.
        :return: ndarray
        """
        if self.X is None:
            return
        else:
            return np.median(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        It returns the minimum of each feature.
        :return: ndarray
        """
        if self.X is None:
            return
        else:
            return np.min(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        It returns the maximum of each feature
        :return: ndarray
        """
        if self.X is None:
            return
        else:
            return np.max(self.X, axis=0)

    def summary(self):
        """
        It returns a Dataframe with summary of the dataset, which means with the mean, variance, median, minimum
        value and maximum value for each feature.
        :return: pandas.Dataframe, DataFrame (n_features, 5)
        """
        return pd.DataFrame(
            {'mean': self.get_mean(),
             'variance': self.get_variance(),
             'median': self.get_median(),
             'min': self.get_min(),
             'max': self.get_max()}
        )

# Métodos extra

    @classmethod
    def from_dataframe(cls, df, label: str = None):
        """
        It creates a Dataset object from a pandas DataFrame
        :return: Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self):
        """
        It converts the dataset to a pandas DataFrame
        :return: pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls, n_samples: int, n_features: int, n_classes: int = 2, features: Sequence[str] = None,
                    label: str = None):
        """
        It creates a Dataset object from random data.
        :param n_samples: int, number of samples
        :param n_features: int, number of features
        :param n_classes: int, number of classes
        :param features: list of str, feature names
        :param label: str, label name
        :return: Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)

#Exercicio 2 de avaliação

    def dropna(self):
        """
        It removes all the samples (rows) which have at least one null value (NaN).
        :return: Dataset
        """
        self.X = self.X[pd.isnull(self.X).any(axis=1)] #axis=1 because its for each sample (row)
        return Dataset(self.X, self.y, self.features, self.label)

    def fillna(self, value):
        """
        It replaces the NaN with another value.
        :param value: float, value that is going to replace NaN values
        :return: None
        """
        self.x = np.nan_to_num(self.X, nan=value)
        #np.nan_to_num -> replaces the NaN values with zeros or other finite number

    def print_dataframe(self):
        """
        It prints a dataset as a dataframe.
        :return: Dataframe
        """
        return pd.DataFrame(self.X, columns = self.features, index = self.y) #index = row labels



if __name__ == '__main__':
    X = np.array([[1,2,3], [1,2,3]])
    y = np.array([1,2])
    features = ["A", "B", "C"]
    label = "y"
    dataset = Dataset(X=X, y=y, features =features, label=label)
    print(dataset.shape())
    #print(dataset.has_label())
    #print(dataset.get_classes())
    #print(dataset.get_mean())
    #print(dataset.get_variance())
    #print(dataset.get_median())
    #print(dataset.get_min())
    #print(dataset.get_max())
    #print(dataset.summary())


