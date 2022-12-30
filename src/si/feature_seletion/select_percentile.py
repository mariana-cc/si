import numpy as np
from si.data.dataset_module import Dataset
from si.statistics.f_classification import f_classification

#Exercício 3 (3.1. e 3.2.)

class SelectPercentile():
    """
    Class that selects the best features according to a given percentile. Feature ranking defined by variance
    analysis using a scoring function (score_func).
    """
    def __init__(self, score_func: callable = f_classification, percentile: int = 20):
        """
        Initializes the select percentile.
        :param score_func: callable, function that calculates the score for each feature
        :param percentile: int, percentile of features to select
        """
        self.score_func = score_func
        self.percentile = percentile #se em percentagem temos de depois dividir por 100
        self.F = None #ainda não os temos
        self.p = None

    def fit(self, dataset: Dataset):
        """
        It fits SelectPercentil to compute the F scores and p-values for each feature, using the function score_func.
        :param dataset: Dataset, dataset object
        :return: self
        """
        self.F, self.p = self.score_func(dataset)  # devolve os valores de F e p
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It tranforms the data, by selecting the percentile of best features with the highest score
        until the percentile given is reached.
        :param dataset: Dataset, dataset object
        :return: Dataset, dataset with selected features
        """
        value = int(len(dataset.features)) #valor total de features no dataset
        mask = value * (percentile/100) #percentil dado em percentagem
        idxs = np.argsort(self.F)[- mask:]
        features = np.array(dataset.features)[idxs]  # vai selecionar as features utilizando os indxs
        features_dataset = dataset.X[:, idxs]
        return Dataset(features_dataset, y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It calculates the score for each feature and selects the percentile best features.
        :param dataset: Dataset, dataset object
        :return: Dataset, dataset with selected features
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == '__main__':
    X = np.array([[1, 2, 3, 4], [3, 6, 5, 1], [7, 4, 1, 5], [1, 3, 2, 9]])
    y = np.array([1, 1, 0, 0])
    dataset_create = Dataset(X, y)
    select = SelectPercentile(f_classification,20)
    select = select.fit(dataset_create)
    new_dataset = select.transform(dataset_create)
    print(new_dataset)
