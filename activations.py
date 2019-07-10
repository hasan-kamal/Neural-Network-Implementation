import numpy as np

# list of activations and their corresponding forward, backward functions
activations = {
    'sigmoid':{
        'sigma': lambda z: 1.0 / (1.0 + np.exp(-z)),
        'sigma_prime': lambda z: np.multiply(1.0 / (1.0 + np.exp(-z)), 1.0 - (1.0 / (1.0 + np.exp(-z))))
        },
    'relu':{
        'sigma': lambda z: np.maximum(np.zeros(shape=z.shape), z),
        'sigma_prime': lambda z: np.where(z <= 0.0, 0.0, 1.0)
        },
    'softmax': {
        'sigma': lambda z: np.exp(z - np.max(z, axis=0)) / np.sum(np.exp(z - np.max(z, axis=0)), axis=0),
        'sigma_prime': lambda z: np.multiply( np.exp(z - np.max(z, axis=0)) / np.sum(np.exp(z - np.max(z, axis=0)), axis=0), 1.0 - ( np.exp(z - np.max(z, axis=0)) / np.sum(np.exp(z - np.max(z, axis=0)), axis=0) ) )
    },
    'linear':{
        'sigma': lambda z: z,
        'sigma_prime': lambda z: np.ones(shape=z.shape)
    },
    'tanh':{
        'sigma': lambda z: ( 2.0 * 1.0 / (1.0 + np.exp(-2.0 * z)) ) - 1.0,
        'sigma_prime': lambda z: 1.0 - np.square( ( 2.0 * 1.0 / (1.0 + np.exp(-2.0 * z)) ) - 1.0 )
    }
}