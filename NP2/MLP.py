import numpy as np
from activation_functions import sigmoid_function
from activation_functions import arredondar
import math 

class MLP():
    def __init__(self, inputSize, outputSize, actfunc=sigmoid_function, precision=0.0000001, learningRate=0.01, camadaIntermediaria = 15, epocas = 1000):
        self.camadaIntermediaria = camadaIntermediaria
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.actfunc = actfunc
        self.precision = precision
        self.learningRate = learningRate
        self.epocas = epocas
        
        self.inicializarPesos()
        
    def inicializarPesos(self):
        self.pesosEntradas = np.random.random((self.camadaIntermediaria, self.inputSize))
        self.pesosSaidas = np.random.random((self.outputSize, self.camadaIntermediaria))
        
    def train(self, training_inputs, outputs):
        eqmAnterior = 0.0
        eqmAtual = 1.0
        
        currentEpoca = 0
        eqm = float(abs(eqmAtual-eqmAnterior))
        
        while (currentEpoca <= self.epocas and eqm > self.precision):
            currentEpoca = currentEpoca + 1
            
            eqmAnterior = eqmAtual
            eqmAtual = 0.0
            
            for x,y in zip(training_inputs, outputs):
                prediction = self.forward(x)
                erros = self.error(x, y, prediction)
                eqmAtual += float(sum(erros))
                self.backward(prediction, x, y)
                
            eqmAtual = float(eqmAtual/len(training_inputs))
            eqm = float(abs(eqmAtual-eqmAnterior))
            
        print(f'epocas: ', currentEpoca)
        print(f'eqm: ',eqm)
            
    def predict(self, training_inputs, outputs):
        for x,y in zip(training_inputs, outputs):
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
                    
                print(f'', y, prediction, predictionArredondado, tipo)
                

    def forward(self, x):
        result1 = np.dot(self.pesosEntradas, x)
        self.intermediateResult = result1
        # self.intermediateResult = self.actfunc(result1)
        result3 = np.dot(self.intermediateResult, self.pesosSaidas.T)
        result4 = self.actfunc(result3)
        return result4
    
    def error(self, x, y, prediction):
        result1 = prediction - y
        for i in range(len(result1)):
            result1[i] = math.pow(result1[i], 2)
            
        return result1
            
    def backward(self, prediction, x, y):
        delta = prediction - y
        result1 = np.asmatrix(self.learningRate * delta).transpose()
        
        result2 = result1.dot(np.asmatrix(self.intermediateResult)).tolist()
        novoPesosSaidas = self.pesosSaidas - result2
        
        result3 = np.dot(result1, np.asmatrix(x))
        result4 = np.dot(self.pesosSaidas.T, result3)
        novoPesosEntradas = self.pesosEntradas - result4
        
        self.pesosEntradas = np.array(novoPesosEntradas)
        self.pesosSaidas = np.array(novoPesosSaidas)
        
            