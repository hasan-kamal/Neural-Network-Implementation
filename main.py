import numpy as np
from mnist import MNIST
from sklearn.preprocessing import StandardScaler

from neural_network import NeuralNetwork
from activations import activations
from losses import losses
from utils import show_confusion_matrix

# returns one-hot encoded Y
def one_hot_encode(Y_label, num_classes):
    Y = np.zeros(shape=(Y_label.shape[0], num_classes))
    for i in range(Y_label.shape[0]):
        Y[i, Y_label[i]] = 1.0
    return Y


if __name__ == '__main__':

	# load training data and create X_train, Y_train numpy arrays
	mndata = MNIST('./python-mnist/data')
	images, labels = mndata.load_training()
	X_train = np.array(images)
	Y_train_labels = np.array(labels)
	Y_train = one_hot_encode(Y_train_labels, 10)

	# load testing data and create X_test, Y_test numpy arrays
	mndata = MNIST('./python-mnist/data')
	images, labels = mndata.load_testing()
	X_test = np.array(images)
	Y_test_labels = np.array(labels)
	Y_test = one_hot_encode(Y_test_labels, 10)

	# standardize data
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# train on a simple neural network having one hidden layer
	nn_model = NeuralNetwork(
	    layers = [
	        {'num_neurons': 784},
	        {'num_neurons': 256, 'activation': 'sigmoid'},
	        {'num_neurons': 10, 'activation': 'sigmoid'}
	    ],
	    loss='mse',
	    init_strategy={'type': 'uniform', 'min': -0.001, 'max': 0.001}
	)
	nn_model.fit(
	    X_train=X_train_scaled.T,
	    Y_train=Y_train.T,
	    X_test=X_test_scaled.T,
	    Y_test=Y_test.T,
	    learning_rate=0.1,
	    batch_size=100,
	    epochs=20)

	# analyze the results of above model by plotting confusion matrix
	Y_predicted_labels = nn_model.predict(X_test_scaled.T)
	label_to_string_map = { i : i for i in range(10) }
	show_confusion_matrix(Y_predicted_labels, Y_test_labels, label_to_string_map)

	# train a neural network having three hidden layers
	nn_model = NeuralNetwork(
	    layers = [
	        {'num_neurons': 784},
	        {'num_neurons': 256, 'activation': 'sigmoid'},
	        {'num_neurons': 128, 'activation': 'sigmoid'},
	        {'num_neurons': 64, 'activation': 'sigmoid'},
	        {'num_neurons': 10, 'activation': 'softmax'}
	    ],
	    loss='mse',
	    init_strategy={'type': 'uniform', 'min': -1.0, 'max': 1.0}
	)
	nn_model.fit(
	    X_train=X_train_scaled.T,
	    Y_train=Y_train.T,
	    X_test=X_test_scaled.T,
	    Y_test=Y_test.T,
	    learning_rate=0.1,
	    batch_size=100,
	    epochs=20)