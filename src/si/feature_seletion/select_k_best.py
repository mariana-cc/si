import numpy as np
from si.data.dataset_module import Dataset
from si.statistics.f_classification import f_classification

class SelectKBest:
    """
    Class that select the best features according to the k highest scores. Feature ranking defined by variance
    analysis using a scoring function (score_func).
    """
    def __init__(self, score_func: callable = f_classification, k: int = 10):
        """
        Initializes the select k best object.
        :param score_func: callable, function that calculates the score for each feature
        :param k: int, number of features to select
        """
        self.score_func = score_func
        self.k = k
        self.F = None #ainda não os temos
        self.p = None

    def fit(self, dataset):
        """
        It fits SelectKBest to compute the F scores and p-values for each feature, using the function score_func.
        :param dataset: Dataset, dataset object
        :return: self
        """
        self.F, self.p = self.score_func(dataset) #devolve os valores de F e p
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the data, by selecting the k fetaures with the highest scores (calculated in the method above)
        :param dataset: Dataset, dataset object
        :return: Dataset, dataset with the selected features
        """
        idxs = np.argsort(self.F)[- self.k:] #argsort devolve os índices de ordenação do array
        features = np.array(dataset.features)[idxs] #vai selecionar as features utilizando os indxs
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset) -> Dataset:
        """
        It fits SelectKBest and transforms the dataset (selecting the features with the k highest scores).
        :param dataset: Dataset, dataset object
        :return: Dataset, dataset with the selected features
        """
        self.fit(dataset)
        return self.transform(dataset) #calls both methods

if __name__ == '__main__':
    X = np.array([[1, 2, 3, 4], [3, 6, 5, 1], [7, 4, 1, 5], [1, 3, 2, 9]])
    y = np.array([1, 1, 0, 0])
    dataset_create = Dataset(X, y)
    select = SelectKBest(f_classification,1) #chamar o método f_classification p/ o cálculo e introduzir valor de k
    select = select.fit(dataset_create)
    new_dataset = select.transform(dataset_create)
    print(new_dataset)


