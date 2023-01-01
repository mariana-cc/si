import numpy as np
from typing import Union
from si.data.dataset_module import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance

class KNNClassifier:
    """
    KNN Classifier - The k-Nearst Neighbors classifier is a ML model that classifies new samples
    based on a similarity measure (like euclidean distance). This algorithm predicts the classes of new samples
    by looking at the classes of the k-nearest samples in the training data.
    """
    def __init__(self, k: int = 1, distance: callable = euclidean_distance):
        """
        Initialize the KNN classifier.
        :param k: int, number of nearest neighbors to be used
        :param distance: callable, distance function to use
        """
        #parameters
        self.k = k
        self.distance = distance
        #attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        It fits the model to the given dataset. Stores the training dataset.
        :param dataset: Dataset, dataset object
        :return: self
        """
        self.dataset = dataset
        return self

    def get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample.
        :param sample: np.ndarray, sample to get the closest label of
        :return: srt or int, the closest label
        """
        # compute distance between the sample and the training dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the labels of the k nearest neighbor
        labels_k_nearest_neighbor = self.dataset.y[k_nearest_neighbors]

        # get the most common label
        labels, counts = np.unique(labels_k_nearest_neighbor, return_counts=True) # array with labels
        # return_counts if True returns the number of times each unique item appears in the array
        return labels[np.argmax(counts)]
        # to obtain the most common, we must see the label that has more counts
        # argmax obtains the one label that has more counts

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset.
        :param dataset: Dataset, dataset to predict the classes
        :return: np.ndarray, predictions of the model
        """
        return np.apply_along_axis(self.get_closest_label, axis=1, arr=dataset.X)
        # axis=1 because its by each row

    def score(self, dataset: Dataset) -> float:
        """
        It obtains the score of the model, which means the accuracy of the model on the given dataset.
        :param dataset: Dataset, dataset to evaluate the model on
        :return: float, accuracy of the model
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)

if __name__ == '__main__':
    # import dataset
    from si.data.dataset_module import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN classifier, giving the k value we want (number of nearest neighbors to be used)
    knn = KNNClassifier(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')
