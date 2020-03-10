import numpy as np
from activation_functions import heaviside_step_function

class Adaline():
    
    def __init__(self, input_size, act_func=heaviside_step_function, epochs=100, learning_rate=0.01, precision=1):
        self.act_func = act_func
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size + 1) 
        self.precision = precision
        
    
    def predict(self, inputs):
        inputs = np.append(-1, inputs)
        u = np.dot(inputs, self.weights)
        return self.act_func(u)
        
    # n = learning_rate
    # y = output model
    # o = desired output
    # w = w + n (o-y)x
    # E = (o-y)²
    
    # x^K = conjuto de amostra = training_inputs
    # d^K = saída desejada = 
    def train(self, training_inputs, labels):
        error = True
        for e in range(self.epochs):
            error = False
            print(f'>>> Start epoch {e + 1}')
            print(f'Actual weights {self.weights}')
            eqmAnterior = self.precision + 1
            eqmAtual = 0

            while(eqmAnterior - eqmAtual <= self.precision)
                

            # for inputs, label in zip(training_inputs, labels):
            #     print(f'Input {inputs}')
            #     predicton = self.predict(inputs)
            #     if predicton != label:
            #         print(f'Expected {label}, got {predicton}. Start trainning!')
            #         inputs = np.append(-1, inputs)
            #         self.weights = self.weights + self.learning_rate * (label - predicton) * inputs
            #         print(f'New weights {self.weights}')
            #         error = True
            #         break
            #     else:
            #         print(f'Everything is OK!')
            
            print('')
            if not error:
                break
            