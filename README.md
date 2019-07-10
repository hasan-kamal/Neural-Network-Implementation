# Neural-Network-Implementation

### Introduction
- This is a working implementation of a **vectorized fully-connected neural network** in NumPy
- Backpropagation algorithm is implemented in a **full-vectorized fashion over a given minibatch**
- This enables us to take advantage of powerful built-in NumPy APIs (and avoid clumsy nested loops!), consequently improving training speed
- Backpropagation code lies in the method `take_gradient_step_on_minibatch` of class `NeuralNetwork` (see `src/neural_network.py`)
- Refer to in-code documentation and comments for description of how the code is working

### Repository structure
- Directory `src/` contains the implementation of neural networks
    1. `src/neural_network.py` contains the actual implementation of the `NeuralNetwork` class (including vectorized backpropagation code)
    2. `src/activations.py` and  `src/losses.py` contain implementations of activation functions and losses, respectively
    3. `src/utils.py` contains code to display confusion matrix
- `main.py` contains driver code that trains an example neural network configuration using the `NeuralNetwork` class

### Installation
- To download MNIST data, install [python-mnist](https://pypi.org/project/python-mnist/) through `git clone` method (run the script to download data; ensure `python-mnist` directory exists inside the root directory of this project)

### Contents
1. Implement class _NeuralNetwork_
2. Implement common activation and loss functions
3. Test implementation on MNIST data

### Results
- 1 hidden-layer (256 dimensional) with sigmoid activation on MNIST data
![Accuracy Plot](results/accuracy_plot.png)
![Confusion Matrix](results/confusion_matrix.png)

### References
1. ["How the backpropagation algorithm works"](http://neuralnetworksanddeeplearning.com/chap2.html) by [Michael Nielsen](http://michaelnielsen.org/)
