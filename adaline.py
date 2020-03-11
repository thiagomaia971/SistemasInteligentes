import numpy as np
from activation_functions import heaviside_step_function
import math as math

class Adaline():
    
    def __init__(self, input_size, act_func=heaviside_step_function, epochs=1000, learning_rate=0.0025, precision=0.000001):
        self.act_func = act_func
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size + 1) 
        self.precision = precision
        
    
    def predict(self, inputs):
        u = np.dot(inputs, self.weights)
        return u
        # return self.act_func(u)
        
    # n = learning_rate
    # y = output model
    # o = desired output
    # w = w + n (o-y)x
    # E = (o-y)²
    
    # x^K = conjuto de amostra = training_inputs
    # d^K = saída desejada = 
    def train(self, training_inputs, labels):
        error = True
        epochs = 0

        eqm = 0
        eqmAnterior = 0
        eqmAtual = 0
        while(error == True or epochs < self.epochs):
            eqmAnterior = eqm/len(training_inputs)
            for inputs, label in zip(training_inputs, labels):
                inputs = np.append(-1, inputs)
                u = self.predict(inputs)
                self.weights = self.weights + self.learning_rate* label - u * inputs
                eqm += self.EQM(label, u)

            epochs = epochs + 1
            eqmAtual = eqm
            if (eqmAnterior - eqmAtual <= self.precision):
                error = False
                
            print(f"epochs: {epochs}")
            print(f"eqmAnterior: {eqmAnterior}")
            print(f"eqmAtual: {eqmAtual}")
            
    def EQM(self, label, u): 
        return math.pow(label-u, 2)
