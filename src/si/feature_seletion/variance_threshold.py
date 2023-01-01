import numpy as np
from si.data.dataset_module import Dataset

class VarianceThreshold:
    """
    It integrates the methods for Variance Threshold feature selection. It selects and remover the features with
    a variance value below a given threshold value.
    """
    def __init__(self, threshold: float = 0.0):
        """
        Initializes the variance threshold.
        :param threshold: float, threshold value to use for feature selection
        """
        # condition
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
        # Parameter
        self.threshold = threshold
        # Attribute
        self.variance = None #para já está vazio porque nós não temos o dataset aqui

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        It receives the dataset and estimates the attribute from the data. In this case,
        calculates the variance of each feature (column).
        :param dataset: Dataset, dataset object
        :return: self
        """
        variance = np.var(dataset.X, axis=0) #ou podíamos usar o método da aula passada
        self.variance = variance #atribuir a variável
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It removes all the features whose variance (calculated with the method above) isn't superior to the
        threshold.
        :param dataset: Dataset, dataset object
        :return: Dataset, dataset with the selected features
        """
        mask = self.variance > self.threshold #mask é valor boolean para identificar as colunas que estão acima
        n_X = dataset.X[:, mask] # dataset com todas as linhas/amostras mas apenas com as colunas/features
        # selecionadas nas mask
        features = np.array(dataset.features)[mask] # selects features with threshold > variance
        return Dataset(n_X, y=dataset.y, features=list(features), label = dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits and transforms the data. Calculates the variance of each feature and then selects the features
        with a variance superior to the threshold.
        :param dataset: Dataset, dataset object
        :return: Dataset, dataset with the selected features
        """
        fitted_model = self.fit(dataset)
        return self.transform(fitted_model)

if __name__ == '__main__':
    from si.data.dataset_module import Dataset
    dataset = Dataset(X = np.array([[0, 2, 0, 3],
                                    [0, 1, 4, 3],
                                    [0, 1, 1, 3]]),
                      y = np.array([0, 1, 0]),
                      features = ["f1", "f2", "f3", "f4"],
                      label = "y")
    select = VarianceThreshold() #dar o valor de threshold que queremos considerar
    select = select.fit(dataset)
    dataset_t = select.transform(dataset)
    print(dataset_t.X)




