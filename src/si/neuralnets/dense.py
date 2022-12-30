import numpy as np


class Dense:
    def __init__(self, input_size: int, output_size: int):
        #parametros
        self.input_size = input_size
        self.output_size = output_size
        #atributos
        self.weights = np.random.random(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def foward (self, input_data: np.ndarray) -> np.ndarray:
        return np.dot(input_data, self.weights) + self.bias
        #dot para multiplicar matrizes
        #ele mostrou tbm outra forma

        # falta cenas

    def backwards(self, error, )
        #get error for previous layer
        error_to_propagate = np.dot(error, self.weights.T)

        #update


class SigmoideActivation:

    def __init__(self, X):
        self.X = X

    def forward:
        self.X = X
        return 1/ (1+ np.exp(-X))

    def backward(self, error, learning_rate):
        sigmoide_derivative = 1 / (1+ np.exp(-self.X))
        sigmoide_derivative = simple_derivate * (1 - sigmoide_derivative)

        #falta coisas



