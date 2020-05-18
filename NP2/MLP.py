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
        self.pesosEntradas = np.random.rand(self.camadaIntermediaria, self.inputSize + 1) #weight1
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
            gradientSum_w1 = 0; gradientSum_w2 = 0
            gradientSum_b1 = 0; gradientSum_b2 = 0
                
            for x,y in zip(training_inputs, outputs):
                self.forward(x)
                
                # Backforward
                self.matrizError = self.matrizResposta - y
                d_weight2 = np.dot(np.asmatrix(self.matrizError).T, np.asmatrix(self.matrizIntermediaria))
                d_bias2 = d_aj2
                
                daj1 = np.dot(self.pesosSaidas.T, d_aj2)*self.actfunc(matrizIntermediaria)
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

    def forward(self, x): 
        x = np.append(-1, x)
        
        self.matrizIntermediaria = np.dot(self.pesosEntradas,x) #+self.biasEntradas
        for i in range(len(self.matrizIntermediaria)):    
            self.matrizIntermediaria[i] = self.actfunc(self.matrizIntermediaria[i])
            
        # Output Layer
        self.matrizResposta = np.dot(self.pesosSaidas, self.matrizIntermediaria) #+self.biasSaida
        for i in range(len(self.matrizResposta)):
            self.matrizResposta[i] = self.actfunc(self.matrizResposta[i])
            
    #def prediction(self, inputSize):
        
            
                
        