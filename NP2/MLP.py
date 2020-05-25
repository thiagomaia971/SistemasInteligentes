import numpy as np
from activation_functions import sigmoid_function
from activation_functions import arredondar
import math 
import matplotlib.pyplot as plt

class MLP():
    def __init__(self, inputSize, outputSize, actfunc=sigmoid_function, precision=0.000001, learningRate=0.001, camadaIntermediaria = 15, epocas = 100):
        self.camadaIntermediaria = camadaIntermediaria
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.actfunc = actfunc
        self.precision = precision
        self.learningRate = learningRate
        self.epocas = epocas
        
        self.inicializarPesos()
        self.inicializarBias()
        
    def inicializarPesos(self):
        self.pesosEntradas = np.random.random((self.camadaIntermediaria, self.inputSize + 1))
        self.pesosSaidas = np.random.random((self.outputSize, self.camadaIntermediaria + 1))
        
    def inicializarBias(self):
        self.bias1 = np.zeros([self.camadaIntermediaria,1])
        self.bias2 = np.zeros([self.outputSize,1])
        
    def train(self, training_inputs, outputs):
        figure = plt.figure(figsize=(13, 6))
        eqmAnterior = 1000.0
        eqmAtual = 0.0
        
        currentEpoca = 0
        eqm = float(abs(eqmAtual-eqmAnterior))
        eqmHistory = []
        
        while (currentEpoca < self.epocas):# and eqm > self.precision):
            eqmLocal = list()
            
            eqmAtual = 0
            
            for x,y in zip(training_inputs, outputs):
                x = np.append(-1, x)
                prediction = self.forward(x)
                self.backward(prediction, x, y)
                eqmLocal.append(sum( 0.5 * (y - self.result2Gradient ** 2)))
                
            eqmHistory.append(sum(eqmLocal)/len(training_inputs))
            currentEpoca = currentEpoca + 1
                
            print(f'Epoca atual: {currentEpoca} | EQM atual: {eqmHistory[currentEpoca - 1]}')
        
        subplot = figure.add_subplot(111)
        width = range(self.epocas)
        height = eqmHistory
        subplot.plot(width, height, marker=',')
        subplot.set_title(f'Se fuder')
        subplot.set_xlabel(r'Épocas')
        subplot.set_ylabel(r'Erro quadratico medio')
        plt.draw()
        plt.pause(0.1)
        plt.show()
        
        
        print('')
        print(f'epocas: ', currentEpoca)
        print(f'eqm: ',eqmHistory[currentEpoca - 1])
                

    def forward(self, x):
        self.result1 = np.dot(self.pesosEntradas, x)
        self.result1Gradient = self.g(self.result1)
        self.result1Gradient = np.append(-1, self.result1Gradient)
        
        self.result2 = np.dot(self.pesosSaidas, self.result1Gradient)
        self.result2Gradient = self.g(self.result2)
        
        return self.result2Gradient
            
    def backward(self, prediction, x, y):  
        d2 = np.zeros(self.result2.shape)
        for j in range(d2.shape[0]):
            d2[j] = (y[j] - self.result2Gradient[j]) * self.dg(self.result2[j])
            
        for j in range(self.pesosSaidas.shape[0]):
            for i in range(self.pesosSaidas.shape[1]):
                self.pesosSaidas[j,i] = self.pesosSaidas[j,i] + self.learningRate * d2[j] * self.result1Gradient[i]
                
        d1 = np.zeros(self.result1.shape)
        for j in range(d1.shape[0]):
            for k in range(0, self.pesosSaidas.shape[0]):
                d1[j] += d2[k] * self.pesosSaidas[k, j + 1]
            d1[j] *= self.dg(self.result1[j])
            
        for j in range(self.pesosEntradas.shape[0]):
            for i in range(self.pesosEntradas.shape[1]):
                self.pesosEntradas[j,i] = self.pesosEntradas[j,i] + self.learningRate * d1[j] * x[i]
    
    def error(self, x, y, prediction):
        result1 = prediction - y
        for i in range(len(result1)):
            result1[i] = math.pow(result1[i], 2)/2
        return result1
            
    def predict(self, training_inputs, outputs):
        acertos = 0
        for x,y in zip(training_inputs, outputs):
            x = np.append(-1, x)
            prediction = self.forward(x)
            predictionArredondado = np.array(prediction)
            
            for i in range(len(prediction)):
                predictionArredondado[i] = arredondar(float(prediction[i]))
                
            if (predictionArredondado == np.array([1,0,0])).all():
                tipo ="Tipo A"
                
            elif (predictionArredondado == np.array([0,1,0])).all():
                tipo ="Tipo B"
            
            elif (predictionArredondado == np.array([0,0,1])).all():
                tipo ="Tipo C"
                
            else:
                tipo ="Nenhum dos tipos"
                
            acertou = False
            if (prediction == np.array(y)).all():
                acertos = acertos + 1
                acertou = True
                
            print(f'{y} | {prediction} | {predictionArredondado} | {tipo} | {acertou}')
        
        print(f'Acertos: {acertos}/{len(training_inputs)}')
    
    def g(self, x):
        func = lambda x: (1 - np.exp(-x)) / (1 + np.exp(-x))
        vfunc = np.vectorize(func)
        return vfunc(x)
        
    def dg(self, x):
        func = lambda x: (1 - np.exp(-x)) / (1 + np.exp(-x))
        vfunc = np.vectorize(func)
        return vfunc(x)
            