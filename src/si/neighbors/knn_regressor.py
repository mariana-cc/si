import numpy as np
from typing import Union
from si.data.dataset_module import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance

# Exercise 3: Implement the KNNRegressor with RMSE

class KNNRegressor:
    """
    KNN Regressor - Implements the K-Nearest Neighbors regressor on a ML model  based on a similarity measure
    (like euclidean distance).
    """
    def __init__(self, k: int = 1, distance: callable = euclidean_distance):
        """
        Initialize the KNN Regressor.
        :param k: int, number of nearest neighbors to be used
        :param distance: callable, distance function to use
        """
        #parameters
        self.k = k
        self.distance = distance
        #attributes
        self.dataset = None

    def fit(self, dataset: Dataset):
        """
        It fits the model to the given dataset. Stores the training dataset.
        :param dataset: Dataset, dataset object
        :return: self
        """
        self.dataset = dataset
        return self

    def get_closest_label_means(self, sample: np.ndarray) -> np.ndarray:
        """
        It returns the mean of the closest labels of the given sample.
        :param sample: np.ndarray, sample to get the label
        :return: float, means of the labels values
        """
        # compute distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # get the indexes of the nearest neighbors
        label_indexes = np.argsort(distances)[:self.k]

        # get the labels values of indexes obtained
        labels_values = self.dataset.y[label_indexes]

        # compute the mean value and return it
        return np.mean(labels_values)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset.
        :param dataset: Dataset, dataset object
        :return: np.ndarray, predictions of the model
        """
        return np.apply_along_axis(self.get_closest_label_means, axis=1, arr=dataset.X)
        # axis=1 because its by each row

    def score(self, dataset: Dataset) -> float:
        """
        It obtains the score of the model, the error between the predicted and true classes, using RMSE formula.
        :param dataset: Dataset, dataset to evaluate
        :return: float, RMSE value of the model
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)

if __name__ == '__main__':
    # import dataset
    from si.data.dataset_module import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN Regressor, giving the k value we want (number of nearest neighbors to be used)
    knn = KNNRegressor(k=3, distance=euclidean_distance)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score_rmse = knn.score(dataset_test)
    print(f'The RMSE value of the model is: {score_rmse}')


