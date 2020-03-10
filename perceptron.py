import numpy as np
from activation_functions import heaviside_step_function

class Perceptron():
    
    def __init__(self, input_size, act_func=heaviside_step_function, epochs=1000, learning_rate=0.0025):
        self.act_func = act_func
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size + 1) 
        
    
    def predict(self, inputs):
        inputs = np.append(-1, inputs)
        u = np.dot(inputs, self.weights)
        return self.act_func(u)
        
    def train(self, training_inputs, labels):
        error = True
        for e in range(self.epochs):
            error = False
            print(f'>>> Start epoch {e + 1}')
            print(f'Actual weights {self.weights}')
            for inputs, label in zip(training_inputs, labels):
                print(f'Input {inputs}')
                predicton = self.predict(inputs)
                if predicton != label:
                    print(f'Expected {label}, got {predicton}. Start trainning!')
                    inputs = np.append(-1, inputs)
                    self.weights = self.weights + self.learning_rate * (label - predicton) * inputs
                    print(f'New weights {self.weights}')
                    error = True
                    break
                else:
                    print(f'Everything is OK!')
            
            print('')
            if not error:
                break
            
