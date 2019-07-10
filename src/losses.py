import numpy as np

# list of loss functions and their corresponding forward, backward functions
losses = {
    'mse': {
        'error': lambda A_L, Y: np.mean((np.sum(np.multiply(Y - A_L, Y - A_L), axis=0))),
        'gradient_wrt_last_activation': lambda A_L, Y: A_L - Y
    }
}