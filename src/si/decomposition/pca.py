import numpy as np
from si.data.dataset_module import Dataset
from si.io.csv_file import read_csv

# Aula 3 - Exercício 3
class PCA:
    """
    It performs principal component analysis (PCA) on the dataset, to reduce the dimensions of a given dataset.
    It uses SVD (Singular Value Decomposition) to do so.
    """
    def __init__(self, n_components: int = 5):
        """
        Initializes the PCA.
        :param n_components: int, number of components
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset):
        """
        It fits PCA on the dataset by computing the mean for each feature, components and explained variance.
        :param dataset: Dataset, dataset object
        :return: self
        """
        # starts by centering the data
        self.mean = np.mean(dataset.X, axis=0) # axis= 0 because its for each feature (each column)
        X_centered = dataset.X - self.mean

        # calculates SVD of X
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # determines the components
        self.components = Vt[:self.n_components]

        # determines explained variances
        n = dataset.shape()[0]
        variances = (S ** 2) / (n - 1)
        self.explained_variance = variances[:self.n_components]

        return self

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        It transforms the dataset by reducing the dataset X, using SVD method.
        :param dataset: Dataset, dataset object
        :return: np.ndarray, transformed dataset (reduced X)
        """
        # starts by centering the data
        self.mean = np.mean(dataset.X, axis=0)
        X_centered = dataset.X - self.mean

        # calculates SVD of X
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # V is the transposed matrix of Vt
        V = Vt.T

        # calculates X reduced
        X_reduced = np.dot(X_centered, V)
        return X_reduced

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        It fits PCA and transforms the dataset.
        :param dataset: Dataset, dataset object
        :return: np.ndarray, transformed dataset (reduced X)
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == "__main__":

    new_dataset = Dataset.from_random(n_samples=5, n_features=5, label=None)
    pca = PCA(n_components=4)
    pca.fit(new_dataset)
    # print(pca.mean)
    # print(pca.components)
    # print(pca.explained_variance)
    X_reduced = pca.transform(new_dataset)
    print(X_reduced)

    #Exercício 3 - 3.3
    path = "../../../datasets/iris.csv"
    file_iris = read_csv(filename=path, sep=",", features=True, label=True)
    pca_iris = PCA(n_components=2)
    iris_reduced = pca_iris.fit_transform(file_iris)
    print(iris_reduced)


