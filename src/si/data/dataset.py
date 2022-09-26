import numpy as np
import pandas as pd


class Dataset(X, y=None, features=None, label=None):
    def __init__(self):
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self):
        return self.X.shape

    def has_label(self):
        if self.y is not None:
            return True
        else:
            return False

    def get_classes(self):
        if self.y is None:
            return

        return no.unique(self.y)

    def get_mean(self):
        return np.mean(self.X, axis=0)

    def get_variance(self):
        return np.variance(self.X, axis=0)

    def get_median(self):
        return np.median(self.X, axis=0)

    def get_min(self):
        return np.minimum(self.X, axis=0)

    def get_max(self):
        return np.maximum(self.X, axis=0)

    def summary(self):
        return pd.DataFrame(
            {'mean': self.getmean(),
             'variance': self.get_variance(),
             'median': self.get_median(),
             'min': self.get_min(),
             'max':self.get_max()}
        )






if __name__ = '__main__':
    x = np.array([1,2,3], [1,2,3])
    y = np.array([1,2])
    features = ["A", "B", "C"]
    label = "y"
    dataset = Dataset(x, y, features =features, label=label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())


