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
        print(f'Initial weights {self.weights}')
        finalEpochs = 0

        for e in range(self.epochs):
            finalEpochs = e
            error = False
            
            for inputs, label in zip(training_inputs, labels):
                predicton = self.predict(inputs)
                if predicton != label:
                    inputs = np.append(-1, inputs)
                    self.weights = self.weights + self.learning_rate * (label - predicton) * inputs
                    error = True
                    break
            
            if not error:
                print(f'Final epoch: {e + 1}')
                break

        print(f'Final epoch: {finalEpochs + 1}')
        print(f'Final weights {self.weights}')
            
