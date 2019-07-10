import matplotlib.pyplot as plt
import math
import numpy as np

from activations import activations
from losses import losses

class NeuralNetwork:
    """ Class implementation of a fully-connected neural network """
    
    
    def __init__(self, layers, loss, init_strategy):
        """
        Initializes the architecture of neural network.
        
        Parameters:
            layers (list)       : list of dicts describing each layer
            loss    (str)       : loss to optimize
            init_strategy (dict): dict specifying weight initialization strategy
        """
        
        # validate parameters
        for i, layer in enumerate(layers):
            assert 'num_neurons' in layer, 'please provide num_neurons for each layer'
            if i > 0:
                assert 'activation' in layer, 'please provide activation for each layer after input layer'
                assert layer['activation'] in activations.keys(), 'unknown activation found'
        assert loss in losses.keys(), 'unknown loss function found'
        
        # create random initial weights and biases for each layer (sampling uniformly in range [w_init_min, w_init_max])
        self.L = len(layers)
        self.layers = layers
        self.loss = loss
        self.w = {}
        self.b = {}
        for i in range(2, self.L + 1):
            if init_strategy['type'] == 'uniform':
                self.w[i] = np.random.uniform(low=init_strategy['min'], high=init_strategy['max'], size=(layers[i - 1]['num_neurons'], layers[i - 2]['num_neurons']))
                self.b[i] = np.random.uniform(low=init_strategy['min'], high=init_strategy['max'], size=(layers[i - 1]['num_neurons'], ))
            elif init_strategy['type'] == 'gaussian':
                self.w[i] = np.random.normal(loc=init_strategy['mean'], scale=init_strategy['std. dev'], size=(layers[i - 1]['num_neurons'], layers[i - 2]['num_neurons']))
                self.b[i] = np.random.normal(loc=init_strategy['mean'], scale=init_strategy['std. dev'], size=(layers[i - 1]['num_neurons'], ))
            elif init_strategy['type'] == 'gaussian_scaled':
                self.w[i] = np.random.normal(loc=init_strategy['mean'], scale=init_strategy['std. dev'], size=(layers[i - 1]['num_neurons'], layers[i - 2]['num_neurons'])) * math.sqrt(2.0 / layers[i - 2]['num_neurons'])
                self.b[i] = np.random.normal(loc=init_strategy['mean'], scale=init_strategy['std. dev'], size=(layers[i - 1]['num_neurons'], )) * math.sqrt(2.0 / layers[i - 2]['num_neurons'])
            else:
                assert False, 'unknown init strategy provided'
                
    
    def fit(self, X_train, Y_train, X_test, Y_test, learning_rate, batch_size, epochs):
        """
        Trains the neural network on given matrices and plots training/testing errors per epoch.
        
        Parameters:
            X_train (np array): training samples, dim num_features x num_samples
            Y_train (np array): training labels,  dim num_classes x num_samples
            X_test  (np array): testing samples, dim num_features x num_samples
            Y_test  (np array): testing labels,  dim num_classes  x num_samples
            learning_rate (float): learning rate of stochastic gradient descent
            batch_size      (int): gradient step is taken over a single batch of this size
            epochs          (int): number of epochs to train for
        """
        
        num_samples = X_train.shape[1]
        train_accuracy, test_accuracy, train_error, test_error = [], [], [], []
        
        for epoch in range(epochs):
            # print progress
            train_accuracy_this_epoch = self.score(X_train, Y_train)
            test_accuracy_this_epoch = self.score(X_test, Y_test)
            train_accuracy.append(train_accuracy_this_epoch)
            test_accuracy.append(test_accuracy_this_epoch)
            train_error.append(self.get_error(X_train, Y_train))
            test_error.append(self.get_error(X_test, Y_test))
            print('{}/{} #epochs done, train_accuracy = {:.3f}, test_accuracy = {:.3f}'.format(epoch, epochs, train_accuracy_this_epoch, test_accuracy_this_epoch))
            
            # train on each minibatch
            for i in range(0, num_samples, batch_size):
                X_minibatch = X_train.T[i:i + batch_size, :].T
                Y_minibatch = Y_train.T[i:i + batch_size, :].T
                self.take_gradient_step_on_minibatch(X_minibatch, Y_minibatch, learning_rate)            
            
        # print final accuracies
        print('{}/{} #epochs done, train_accuracy = {:.3f}, test_accuracy = {:.3f}'.format(epochs, epochs, self.score(X_train, Y_train), self.score(X_test, Y_test)))
    
        # plot accuracy plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title('Accuracy as a function of training epochs')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epochs')
        ax.plot(range(len(train_accuracy)), train_accuracy, color='blue')
        ax.plot(range(len(train_accuracy)), train_accuracy, marker='o', color='blue', label='Train accuracy')
        ax.plot(range(len(test_accuracy)), test_accuracy, color='#f39c12')
        ax.plot(range(len(test_accuracy)), test_accuracy, marker='o', color='#f39c12', label='Test accuracy')
        ax.legend()
        plt.show()
        plt.close(fig)
        
        # plot loss plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title('Loss as a function of training epochs')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epochs')
        ax.plot(range(len(train_error)), train_error, color='blue')
        ax.plot(range(len(train_error)), train_error, marker='o', color='blue', label='Train loss')
        ax.plot(range(len(test_error)), test_error, color='#f39c12')
        ax.plot(range(len(test_error)), test_error, marker='o', color='#f39c12', label='Test loss')
        ax.legend()
        plt.show()
        plt.close(fig)
    

    def take_gradient_step_on_minibatch(self, X_minibatch, Y_minibatch, learning_rate):
        """
        This method takes one gradient step using minibatch provided as parameter.
        This method contains the actual backpropagation code.
        
        Parameters:
            X_minibatch (np array): minibatch training samples, dim num_features x num_samples
            Y_minibatch (np array): minibatch training labels,  dim  num_classes x num_samples
            learning_rate  (float): learning rate of stochastic gradient descent
        """
        
        # forward pass
        z, a = {}, {}
        a[1] = X_minibatch
        for l in range(2, self.L + 1):
            z[l] = ( np.matmul(self.w[l], a[l - 1]).T + self.b[l] ).T
            a[l] = activations[self.layers[l - 1]['activation']]['sigma'](z[l])
        
        # backward pass
        delta = {}
        delta[self.L] = np.multiply( losses[self.loss]['gradient_wrt_last_activation'](a[self.L], Y_minibatch), activations[self.layers[-1]['activation']]['sigma_prime'](z[self.L]) )
        for l in range(self.L - 1, 1, -1):
            delta[l] = np.multiply(np.matmul(self.w[l + 1].T, delta[l + 1]), activations[self.layers[l - 1]['activation']]['sigma_prime'](z[l]))
        
        # compute gradients
        dw, db = {}, {}
        for l in range(self.L, 1, -1):
            dw[l] = np.matmul(delta[l], a[l-1].T) / float(X_minibatch.shape[1])
            db[l] = np.mean(delta[l], axis=1)
        
        # take gradient step
        for l in range(self.L, 1, -1):
            self.w[l] = self.w[l] - ( learning_rate * dw[l] )
            self.b[l] = self.b[l] - ( learning_rate * db[l] )
                

    def forward(self, X):
        """
        Computes forward-pass i.e. output (probabilities) on samples of X
        
        Parameters:
            X (np array): samples, dim num_features x num_samples
        
        Returns:
            A_L (np array): output probabilities, dim num_classes x num_samples
        """
        
        z, a = {}, {}
        a[1] = X
        for l in range(2, self.L + 1):
            z[l] = ( np.matmul(self.w[l], a[l - 1]).T + self.b[l] ).T
            a[l] = activations[self.layers[l - 1]['activation']]['sigma'](z[l])
        return a[self.L]
    
    
    def predict(self, X):
        """
        Computes predictions (labels) on samples of X, made by this neural network
        
        Parameters:
            X (np array): samples, dim num_features x num_samples
        
        Returns:
            Y (np array): predicted labels, dim (num_samples,)
        """
        
        Y_scores = self.forward(X)
        return np.argmax(Y_scores, axis=0)
    
    
    def score(self, X, Y):
        """
        Computes accuracy of this (possibly trained) neural network on X, Y

        Parameters:
            X (np array): samples, dim num_features x num_samples
            Y (np array): labels,  dim num_classes x num_samples
            
        Returns:
            float: accuracy of model on X, Y
        """
        
        Y_predicted_labels = self.predict(X)
        Y_labels = np.argmax(Y, axis=0)
        N = float(X.shape[1])
        return np.sum(Y_predicted_labels == Y_labels) / N

    
    def get_error(self, X, Y):
        """
        Computes error (i.e. loss) value of this model on X, Y

        Parameters:
            X (np array): samples, dim num_features x num_samples
            Y (np array): labels,  dim num_classes x num_samples
            
        Returns:
            float: loss of model on X, Y
        """
        
        A_L = self.forward(X)
        return losses[self.loss]['error'](A_L, Y)