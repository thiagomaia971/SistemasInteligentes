import numpy as np
from activation_functions import heaviside_step_function
import math as math

class Adaline():
    
    def __init__(self, log, input_size, act_func=heaviside_step_function, epochs=1000, learning_rate=0.0025, precision=0.00001):
        self.log = log
        self.act_func = act_func
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size + 1) 
        self.precision = precision
        
    
    def predict(self, inputs):
        inputs = np.append(-1, inputs)
        predict = np.dot(inputs, self.weights)
        return predict

    def train(self, training_inputs, labels):
        self.log.printWeights(f'>>>>> Initial weights', self.weights)

        epochs = 0.0
        eqmAnterior = 0.0
        eqmAtual = 0.0
        while(True):
            eqmAnterior = eqmAtual
            eqmAtual = 0.0
            for inputs, output in zip(training_inputs, labels):
                predict = self.predict(inputs)
                inputs = np.append(-1, inputs)
                self.weights += self.learning_rate * (output - predict) * inputs
                eqmAtual += float(self.EQM(output, predict))

            epochs = epochs + 1
            eqmAtual = float(eqmAtual/len(training_inputs))
            
            if (epochs + 1 > self.epochs or float(abs(eqmAtual-eqmAnterior)) <= self.precision):
                break

        self.log.printWeights(f'>>>>> Final weights', self.weights)
        self.log.print(f"epochs: {epochs}")
        # self.log.print(f"eqmAnterior: {eqmAnterior}")
        # self.log.print(f"eqmAtual: {eqmAtual}")
        # self.log.print(f"diff: {abs(eqmAtual - eqmAnterior)}")
            
    def EQM(self, output, predict): 
        return math.pow(output-predict, 2)
