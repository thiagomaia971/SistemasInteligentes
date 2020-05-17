import numpy as np
from activation_functions import signum_function

class MLP():
    def __init__(self, inputSize, outputSize, actfunc=signum_function, precision=0.000001, learningRate=0.1, camadaIntermediaria = 15):
        self.camadaIntermediaria = camadaIntermediaria
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.actfunc = actfunc
        self.precision = precision
        self.learningRate = learningRate
        self.pesosEntradas = np.random.rand(self.camadaIntermediaria, self.inputSize)
        self.pesosSaidas = np.random.rand(self.outputSize, self.camadaIntermediaria)
        self.biasEntradas = np.zeros([self.camadaIntermediaria,1])
        self.biasSaida = np.zeros([self.outputSize,1])
        