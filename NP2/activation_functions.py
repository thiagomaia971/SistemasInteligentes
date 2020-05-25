import numpy as np

def heaviside_step_function(u):
    return 1 if u >= 0 else 0

def signum_function(u):
    return 1 if u >= 0 else -1

def sigmoid_function(u):
    # return 1/(1 + np.exp(-u))
    return (1 - np.exp(-u)) / (1 + np.exp(-u))

def arredondar(value):
    return 1 if value >= 0.5 else 0.0