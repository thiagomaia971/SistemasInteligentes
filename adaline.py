import numpy as np
from activation_functions import heaviside_step_function
import math as math

class Adaline():
    
    def __init__(self, input_size, act_func=heaviside_step_function, epochs=1000, learning_rate=0.0025, precision=0.00001):
        self.act_func = act_func
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size + 1) 
        self.precision = precision
        
    
    def predict(self, inputs):
        predict = np.dot(inputs, self.weights)
        return predict

    def train(self, training_inputs, labels):
        epochs = 0.0

        eqmAnterior = 0.0
        eqmAtual = 0.0
        while(True):
            eqmAnterior = eqmAtual
            eqmAtual = 0.0
            for inputs, output in zip(training_inputs, labels):
                inputs = np.append(-1, inputs)
                predict = self.predict(inputs)
                self.weights += self.learning_rate * (output - predict) * inputs
                eqmAtual += float(self.EQM(output, predict))

            epochs = epochs + 1
            eqmAtual = float(eqmAtual/len(training_inputs))
            
            if (epochs > self.epochs or float(abs(eqmAtual-eqmAnterior)) <= self.precision):
                break

        print(f"epochs: {epochs}")
        print(f"eqmAnterior: {eqmAnterior}")
        print(f"eqmAtual: {eqmAtual}")
        print(f"diff: {abs(eqmAtual - eqmAnterior)}")
            
    def EQM(self, output, predict): 
        return math.pow(output-predict, 2)
