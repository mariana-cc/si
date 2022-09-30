import numpy as np
import pandas as pd
from numpy import ndarray


class Dataset:
    def __init__(self, x: ndarray, y: ndarray=None, features: list=None, label: str=None):
        self.x = x
        self.y = y
        self.features = features
        self.label = label

    def shape(self):
        return self.x.shape

    def has_label(self):
        if self.y is not None:
            return True
        else:
            return False

    def get_classes(self):
        if self.y is None:
            return
        else:
            return np.unique(self.y)

    def get_mean(self):
        if self.x is None:
            return
        else:
            return np.mean(self.x, axis=0)

    def get_variance(self):
        if self.x is None:
            return
        else:
            return np.var(self.x, axis=0)

    def get_median(self):
        if self.x is None:
            return
        else:
            return np.median(self.x, axis=0)

    def get_min(self):
        if self.x is None:
            return
        else:
            return np.min(self.x, axis=0)

    def get_max(self):
        if self.x is None:
            return
        else:
            return np.max(self.x, axis=0)

    def summary(self):
        return pd.DataFrame(
            {'mean': self.get_mean(),
             'variance': self.get_variance(),
             'median': self.get_median(),
             'min': self.get_min(),
             'max':self.get_max()}
        )






if __name__ = '__main__':
    x = np.array([[1,2,3], [1,2,3]])
    y = np.array([1,2])
    features = ["A", "B", "C"]
    label = "y"
    dataset = Dataset(x=x, y=y, features =features, label=label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())


