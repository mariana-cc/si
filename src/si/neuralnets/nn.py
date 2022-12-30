

class NN:
    def __init__(self,
                 layers: list,
                 epachs: int = 1000,
                 learning_rate: float = 0.01,
                 loss: Callable = mse,
                 loss_derivate: Callable = mse_derivate,
                 verbose: bool = False):
        '''
        Inicializes the neural network model
        :param layers:
        :param epoch:
        :param learning_rate:
        :param loss:
        :param loss_derivate:
        :param verbose:
        '''
        #parameters
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.loss_derivate = loss_derivate
        self.verbose = verbose
        #atributes
        self.history = {}

    def fit(self, dataset: Dataset) -> 'NN':
        X = dataset.X
        Y = dataset.Y

        for epoch in range(1, self.epochs +1):
            #foward propagation
            for layer in self.layers:
                X = layer.foward(X)

            #backward propagation
            error = self.loss_derivate(y, X)
            for layer in self.layers[::-1]: #última layer
                error = layer.backward(error, self.learning_rate)

            #save history
            cost = self.loss(y, X)
            self.history(epoch) = cost

            #print loss - é a nossa loss function
            if self.verbose:
                print(f'Epoch {epoch}/{self.epochs} - cost{cost}') #nao sei se esta linha está bem



    #nós não temos o backwar implementado por isso vamos ter de ir a todas e colocar - vamos fazer na próxima aula??



