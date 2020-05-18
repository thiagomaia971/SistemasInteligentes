import numpy as np
from activation_functions import sigmoid_function

class MLP():
    def __init__(self, inputSize, outputSize, actfunc=sigmoid_function, precision=0.000001, learningRate=0.1, camadaIntermediaria = 15, epocas = 148, tamanhoLote = 19):
        self.camadaIntermediaria = camadaIntermediaria
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.actfunc = actfunc
        self.precision = precision
        self.learningRate = learningRate
        self.epocas = epocas
        self.tamanhoLote = tamanhoLote
        
        self.inicializarPesos()
        self.iniciandoBias()
        
    def inicializarPesos(self):
        self.pesosEntradas = np.random.rand(self.camadaIntermediaria, self.inputSize) #weight1
        self.pesosSaidas = np.random.rand(self.outputSize, self.camadaIntermediaria)  #weight2
        
    def iniciandoBias(self):
        self.biasEntradas = np.zeros([self.camadaIntermediaria,1]) #bias1
        self.biasSaida = np.zeros([self.outputSize,1])             #bias1
        
    def train(self, training_inputs, outputs):
        currentEpoca = 0
        eqm = 0
        erroAnterior = 1000
        
        novoPeso1 = 0; novoPeso2 = 0
        novoBias1 = 0; novoBias2 = 0
        
        while (currentEpoca <= self.epocas and abs(eqm - erroAnterior) > self.precision):
            #for j in range(0, len(training_inputs), self.tamanhoLote):
            #    loteTreinamento = training_inputs[j, j+ self.tamanhoLote]
                
            gradientSum_w1 = 0; gradientSum_w2 = 0
            gradientSum_b1 = 0; gradientSum_b2 = 0
                
            for x,y in zip(training_inputs, outputs):
                # Forward
                # First Hidden Layer
                inj1 = np.dot(self.pesosEntradas,x)+self.biasEntradas
                aj1 = self.actfunc(inj1)
                # Output Layer
                inj2 = np.dot(self.pesosSaidas,x)+self.biasSaida
                aj2 = self.actfunc(inj2)
                
                # Backforward
                d_aj2 = aj2 - y
                d_weight2 = np.dot(d_aj2,aj1.T)
                d_bias2 = d_aj2
                
                daj1 = np.dot(self.pesosSaidas.T, d_aj2)*self.actfunc(inj1)
                d_weight1 = np.dot(daj1,x.T)
                d_bias1 = daj1
                
                # Accumulate Gradients
                gradientSum_b1 += d_bias1; gradientSum_b2 += d_bias2
                gradientSum_w1 += d_weight1; gradientSum_w2 += d_weight2
                
            novoPeso1 = novoPeso1 - (self.learningRate* gradientSum_w1)
            novoPeso2 = novoPeso2 - (self.learningRate*gradientSum_w2)
            novoBias1 = novoBias1 - (self.learningRate*gradientSum_b1)
            novoBias2 = novoBias2 - (self.learningRate*gradientSum_b2)
            
            self.biasEntradas += novoBias1; self.biasSaida += novoBias2
            self.pesosEntradas += novoPeso1; self.pesosSaidas += novoPeso2
            
    #def prediction(self, inputSize):
        
            
                
        