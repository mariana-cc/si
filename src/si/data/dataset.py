import numpy as np
import pandas as pd
from numpy import ndarray


class Dataset:
    '''
    Classe que representa um dataset
    '''
    def __init__(self, x: ndarray, y: ndarray=None, features: list=None, label: str=None):
        '''
        Método que inicializa o dataset

        :param x: tabela das features (variáveis independentes)
        :param y: amostras (variáveis dependentes)
        :param features: nome das features
        :param label: nome das amostras
        '''
        self.x = x
        self.y = y
        self.features = features
        self.label = label

    def shape(self):
        '''
        Método que retorna as dimensões do dataset
        :return:Tuple
        '''
        return self.x.shape

    def has_label(self):
        '''
        Método que verifica se o dataset tem y (amostras)
        :return: Boolean
        '''
        if self.y is not None:
            return True
        else:
            return False

    def get_classes(self):
        '''
        Método que retorna as classes do dataset
        :return: ndarray
        '''
        if self.y is None:
            return
        else:
            return np.unique(self.y)

    def get_mean(self):
        '''
        Método que retorna a média do dataset para cada feature
        :return: ndarray
        '''
        if self.x is None:
            return
        else:
            return np.mean(self.x, axis=0)

    def get_variance(self):
        '''
        Método que retorna a variância do dataset para cada feature
        :return: ndarray
        '''
        if self.x is None:
            return
        else:
            return np.var(self.x, axis=0)

    def get_median(self):
        '''
        Método que retorna a mediana do dataset para cada feature
        :return: ndarray
        '''
        if self.x is None:
            return
        else:
            return np.median(self.x, axis=0)

    def get_min(self):
        '''
        Método que retorna o valor mínimo do dataset para cada feature
        :return: ndarray
        '''
        if self.x is None:
            return
        else:
            return np.min(self.x, axis=0)

    def get_max(self):
        '''
        Método que retorna o valor máximo do dataset para cada feature
        :return: ndarray
        '''
        if self.x is None:
            return
        else:
            return np.max(self.x, axis=0)

    def summary(self):
        '''
        Método que retorna um panda DataFrame com o sumário do dataset com a média, variância, mediana,
        valor mínimo e valor máximo para cada feature
        :return: DataFrame
        '''
        return pd.DataFrame(
            {'mean': self.get_mean(),
             'variance': self.get_variance(),
             'median': self.get_median(),
             'min': self.get_min(),
             'max':self.get_max()}
        )

    def remove_null(self):
        '''
        Método que remove os valores nulos.
        :return: DataFrame
        '''
        if self.x is None:
            return pd.DataFrame(self.x).dropna(axis=0)

    def replace_null(self, value):
        '''
        Método que substitui os valores nulos por outro valor
        :param value: valor que vai substituir o valor nulo
        :return: DataFrame
        '''
        if self.x is None:
            return pd.DataFrame(self.x).fillna(value)



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


