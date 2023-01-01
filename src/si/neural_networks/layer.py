import numpy as np

class Dense:
    """
    A dense layer is a layer where each neuron is connected to all neurons in the previous layer.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the dense layer.
        :param input_size: int, the number of inputs the layer will receive
        :param output_size: int, the number of outputs the layer will produce
        """
        # Parameters
        self.input_size = input_size
        self.output_size = output_size
        # Attributes
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def foward (self, X: np.ndarray) -> np.ndarray:
        """
        It performs forward pass of the layer using the given input.
        Returns a 2 dimension numpy array with shape (1, output_size).
        :param X: np.ndarray, the input to the layer
        :return: np.ndarray, the output to the layer.
        """
        return np.dot(X, self.weights) + self.bias
        # dot to multiply the matrix

    def backwards(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        It computes the backward pass of the layer. Returns the error of the previous layer.
        :param error: np.ndarray, error value of the loss function
        :param learning_rate: float, the learning rate
        :return: np.ndarray, error of the previous layer
        """
        #get error of the previous layer
        error_to_propagate = np.dot(error, self.weights.T)

        #update the weights and bias
        self.weights = self.weights - learning_rate * np.dot(self.X.T, error)
        # x.T is used to multiply the error by the input data due to matrix multiplication rules
        self.bias = self.bias - learning_rate * np.sum(error, axis=0)
        return error_to_propagate

class SigmoideActivation:
    """
     A sigmoid activation layer.
    """
    def __init__(self):
        """
        Initialize the sigmoid activation layer.
        """
        # Attribute
        self.X = None

    def forward:
        self.X = X
        return 1/ (1+ np.exp(-X))

    def backward(self, error, learning_rate):
        sigmoide_derivative = 1 / (1+ np.exp(-self.X))
        sigmoide_derivative = simple_derivate * (1 - sigmoide_derivative)

        #falta coisas



