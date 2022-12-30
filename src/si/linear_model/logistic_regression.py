import numpy as np

from si.data.dataset_module import Dataset
from si.data.statistics import sigmoide_function
from si.data.metrics import accuracy

class LogisticRegression:
    '''
    Objeto para o cálculo da regressão logística
    '''
    def __init__(self, l2_penalty, alpha, max_iter):
        '''
        Método inicializa o objeto.
        :param l2_penalty: coeficiente da regularização L2
        :param alpha: learning rate (taxa de aprendizagem)
        :param max_iter: número máximo de iterações
        '''
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = None #coeficientes/parâmetros do modelo para as variáveis de entrada (features)
        self.theta_zero = None #o coeficiente/parâmetro zero (também conhecido como interceção)

    def fit(self, dataset: Dataset) -> 'LogisticRegression':
        '''
        Método que faz fit do modelo ao dataset de input
        :param dataset: Dataset, dataset de input
        :return: o modelo para o dataset de input
        '''
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = sigmoide_function(y_pred)

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

        return self

    def predict(self, dataset: Dataset) -> np.array:
        '''
        Método que prevê o output do dataset
        :param dataset: Dataset, dataset que queremos prever o output
        :return: np.array, a predição do dataset
        '''
        predictions = sigmoide_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        #convert the predictions to 0 or 1 (Binarization)
        mask = predictions >= 0.5 #meio da função sigmoide
        predictions[mask] = 1
        predictions[~mask] = 0
        return predictions

    def score(self, dataset: Dataset) -> float:
        '''

        :param dataset: Dataset, dataset em que queremos calcular a accuracy
        :return: float, cálculo da accuracy
        '''
        y_pred = self.predict(dataset)
        return acurracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        '''

        :param dataset: Dataset,
        :return: float, cost
        '''
        predictions = sigmoide_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        cost = (-dataset.y * np.log(predictions)) - ((1 - dataset.y) * np.log(1 - predictions))
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * dataset.shape()[0]))
        return cost

    if __name__ == '__main__':
        #import dataset
        from si.data.dataset_module import Dataset
        from si.model_selection.split import train_test_split

        #load and split the dataset


