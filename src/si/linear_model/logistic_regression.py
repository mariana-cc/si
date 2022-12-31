import numpy as np
from si.data.dataset_module import Dataset
from si.statistics import sigmoid_function
from si.metrics import accuracy

class LogisticRegression:
    """
    The LogisticRegression is a logistic model using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique.
    """
    def __init__(self, l2_penalty, alpha, max_iter):
        """
        Initialize the Ridge Regression model.
        Parameters
        :param l2_penalty: float, the L2 regularization parameter
        :param alpha: float, the learning rate
        :param max_iter: int, the maximum number of iterations

        Attributes
        theta: np.ndarray, the model parameters, namely the coefficients of the linear model.
            For example, x0 * theta[0] + x1 * theta[1] + ...
        theta_zero: float, the intercept of the logistic model
        """
        # Parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # Attributes
        self.theta = None # model coefficient for the entry variables (features)
        self.theta_zero = None # intercept of the linear model, zero coefficient
        # Exercise 6.1. - add cost history
        self.cost_history = None  # history of the cost function

    def fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        It fits the model to the dataset.
        :param dataset: Dataset, dataset to fit the model to
        :return: self, fitted model
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # cost history
        self.cost_history = {}  # dic empty

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = sigmoid_function(y_pred)

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # computes the cost function
            self.cost_history[i] = self.cost(dataset)

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            # Exercise 6.3 - When the difference between the cost of the previous iteration and the cost of the
            # current iteration is less than 0.0001, the Gradient Descend should stop.
            if i > 0:
                if self.cost_history[i - 1] - self.cost_history[i] < 0.0001:
                    break

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        It predicts the output of the dataset.
        :param dataset: Dataset, dataset to predict the output of
        :return: np.ndarray, the predictions of the dataset
        """
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        # convert the predictions to 0 or 1 (Binarization)
        mask = predictions >= 0.5 # middle of the sigmoid function
        predictions[mask] = 1
        predictions[~mask] = 0
        return predictions

    def score(self, dataset: Dataset) -> float:
        """
        It computes the Mean Square Error (MSE) of the model on the dataset
        :param dataset: Dataset, dataset to compute the MSE on
        :return: float, the MSE value of the model
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        It computes the cost function (J function) of the model on the dataset using L2 regularization.
        :param dataset: Dataset, dataset to compute the cost function on
        :return: float, the cost function of the model
        """
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        cost = (-dataset.y * np.log(predictions)) - ((1 - dataset.y) * np.log(1 - predictions))
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * dataset.shape()[0]))
        return cost

    # Exercise 6.2. - add a method that computes a plot of the cost function history of the model
    def cost_function_plot(self):
        """
        It plots the cost function history of the model.
        :return: None
        """
        import matplotlib.pyplot as plt

        # plot - Y axis should contain the cost value while the X axis should contain the iterations
        x_iterations = list(self.cost_history.keys())
        y_values = list(self.cost_history.values())

        # plot construction
        plt.plot(x_iterations, y_values, '-r')
        plt.title("Cost History of the model")
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()

if __name__ == '__main__':
    # import dataset
    from si.data.dataset_module import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # fit the model
    model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    model.fit(dataset_train)

    # compute the score
    score = model.score(dataset_test)
    print(f"Score: {score}")

    # compute the cost
    cost = model.cost(dataset_)
    print(f"Cost: {cost}")

    # plot
    model.cost_function_plot()


