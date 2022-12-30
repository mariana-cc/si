import numpy as np
from si.data.dataset_module import Dataset
from si.statistics.euclidean_distance import euclidean_distance

class KMeans:
    """
    It integrates the methods to perform k-means clustering on the dataset.
    It groups samples into clusters by trying to reduce the distance between the samples and their closest centroid
    and returns the centroids and the indexes of the closest centroid for each sample.
    """
    def __init__(self, k : int, max_iter : int = 1000, distance : callable = euclidean_distance):
        """
        K-means clustering algorithm.
        :param k: int, number of clusters
        :param max_iter: int, maximum nunber of interations
        :param distance: Callable, function that calculates the distance of each sample to each centroid
        """
        #parameters
        self.k = k
        self.max_iter = max_iter
        self.distance = distance
        #attributes
        self.centroids = None
        self.labels = None

    def _init_centroids(self, dataset: Dataset):
        """
        It generates initial k centroids.
        :param dataset: Dataset, dataset object
        :return: None
        """
        seeds = np.random.permutation(dataset.shape()[0])[:self.k] #randomly selects k samples from the dataset
        self.centroids = dataset.X[seeds] #selected samples used as centroids

    def get_closest_centroid(self, sample: np.ndarray) -> np.ndarray:
        """
        It gets the centroid closest to each sample.
        :param sample: np.ndarray, sample
        :return: np.ndarray, array with the closet centroid to each sample
        """
        distances = self.distance(sample, self.centroids) #calculates the distance of x(samples) and each y(centroids)
        closest_centroid_index = np.argmin(distances, axis=0) # index of the centroid with the minimum distance for each sample
        return closest_centroid_index

    def fit(self, dataset : Dataset) -> 'KMeans':
        """
        It fits k-means clustering to the dataset.
            1 - initializes centroids, 2 - iteratively updates them until convergence or max_iter.
            Convergence is reached when the centroids do not change anymore.
        :param dataset: Dataset, dataset object
        :return: self, KMeans
        """
        #generate initial centroids
        self._init_centroids(dataset)

        #fitting the k-means
        converge = False
        i = 0 #iteration counter
        labels = np.zeros(dataset.shape()[0]) #array of zeros to store the labels of each sample
        while not converge and i < self.max_iter:
        #while the algorithm has not converged and the maximum number of iterations has not been reached
            #get closest centroid
            new_labels = np.apply_along_axis(self.get_closest_centroid, axis=1, arr=dataset.X)

            #new centroids
            centroids = []
            for j in range(self.k):
                centroid = np.mean(dataset.X[new_labels == j], axis=0)
                centroids.append(centroid)
            self.centroids = np.array(centroids)

            #check if the centroids have changed
            converge = np.any(new_labels != labels)
            #replace labels
            labels = new_labels
            #increment counting
            i += 1

        self.labels = labels
        return self

    def _get_distances(self, sample: np.ndarray) -> np.ndarray:
        """
        It computes the distance between each samples and the closest centroid.
        :param sample: np.ndarray, sample
        :return: np.ndarray, array with the distances obtained
        """
        return self.distance(sample, self.centroids)


    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        It transforms the dataset, computing the distance between each sample and closest centroid.
        :param dataset: Dataset, dataset object
        :return: np.ndarray, transformed dataset
        """
        centroid_distances = np.apply_along_axis(self._get_distances, axis=1, arr=dataset.X)
        return centroid_distances

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        It fits the model and transforms the dataset.
        :param dataset: Dataset
        :return: np.ndarray, transformed dataset
        """
        self.fit(dataset)
        return self.transform(dataset)


    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the labels of the dataset.
        :param dataset: Dataset, dataset object
        :return: np.ndarray, predicted labels
        """
        return np.apply_along_axis(self.get_closest_centroid, axis=1, arr=dataset.X)

    def fit_predict(self, dataset: Dataset) -> np.ndarray:
        """
        It fits and predicts the labels of the dataset.
        :param dataset: Dataset, dataset object
        :return: np.ndarray, predicted labels
        """
        self.fit(dataset)
        return self.predict(dataset) #calls both methods

if __name__ == '__main__':
    from si.data.dataset_module import Dataset
    dataset_ = Dataset.from_random(100, 5)

    k_ = 3
    kmeans = KMeans(k_)
    res = kmeans.fit_transform(dataset_)
    predictions = kmeans.predict(dataset_)
    print(res.shape)
    print(predictions.shape)



