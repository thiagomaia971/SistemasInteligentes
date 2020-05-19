import numpy as np
from activation_functions import sigmoid_function
from activation_functions import arredondar
import math 

class MLP():
    def __init__(self, inputSize, outputSize, actfunc=sigmoid_function, precision=0.000001, learningRate=0.01, camadaIntermediaria = 15, epocas = 200):
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
        self.pesosEntradas = np.random.random((self.camadaIntermediaria, self.inputSize))
        self.pesosSaidas = np.random.random((self.outputSize, self.camadaIntermediaria))
        
    def inicializarBias(self):
        self.bias1 = np.zeros([self.camadaIntermediaria,1])
        self.bias2 = np.zeros([self.outputSize,1])
        
    def train(self, training_inputs, outputs):
        eqmAnterior = 1000.0
        eqmAtual = 0.0
        
        currentEpoca = 0
        eqm = float(abs(eqmAtual-eqmAnterior))
        
        while (currentEpoca <= self.epocas):# and eqm > self.precision):
            currentEpoca = currentEpoca + 1
            
            #eqmAnterior = eqmAtual
            
            for x,y in zip(training_inputs, outputs):
                prediction = self.forward(x)
                self.backward(prediction, x, y)
                #erros = self.error(x, y, prediction)
                #eqmAtual += float(sum(erros))
                
            #eqmAtual = float(eqmAtual/len(training_inputs))
            #eqm = float(abs(eqmAtual-eqmAnterior))
        
        print(f'epocas: ', currentEpoca)
        print(f'eqm_anterior: ',eqmAnterior)
        print(f'eqm_atual: ',eqmAtual)
        print(f'eqm: ',eqm)
                

    def forward(self, x):
        self.inj1 = np.dot(self.pesosEntradas, x)
        self.aj1 = self.actfunc(self.inj1)
        
        self.inj2 = np.dot(self.pesosSaidas, self.inj1)
        self.aj2 = self.actfunc(self.inj2)
        
        return np.array(self.aj2)
        #result1 = np.dot(self.pesosEntradas, x)
        #self.intermediateResult = result1
        # #self.intermediateResult = self.actfunc(result1)
        #result3 = np.dot(self.pesosSaidas, self.intermediateResult)
        #result4 = self.actfunc(result3)
        # #result4 = result3
        #return result4
    
    def error(self, x, y, prediction):
        result1 = prediction - y
        for i in range(len(result1)):
            result1[i] = math.pow(result1[i], 2)/2
        return result1
            
    def backward(self, prediction, x, y):        
        self.d_aj2 = self.aj2 - y
        self.d_pesosSaidas = np.dot(np.asmatrix(self.d_aj2).T, np.asmatrix(self.aj1))
        
        self.daj1 = np.dot(self.pesosSaidas.T, self.d_aj2) * self.actfunc(self.inj1)
        self.d_pesosEntradas = np.dot(np.asmatrix(self.daj1).T, np.asmatrix(x))
        
        velocity_pesosEntradas = self.pesosEntradas - (self.learningRate* self.d_pesosEntradas)
        velocity_pesosSaidas = self.pesosSaidas - (self.learningRate* self.d_pesosSaidas)
        
        v = self.pesosEntradas + velocity_pesosEntradas
        v1 = self.pesosSaidas + velocity_pesosSaidas
        
        self.pesosEntradas = np.array(velocity_pesosEntradas)
        self.pesosSaidas = np.array(velocity_pesosSaidas)
        
        #result1 = np.asmatrix(np.dot(self.learningRate, delta)).transpose()
        
        #result2 = result1.dot(np.asmatrix(self.intermediateResult))
        #novoPesosSaidas = self.pesosSaidas - result2
        
        #result3 = np.dot(result1, np.asmatrix(x))
        #result4 = np.dot(self.pesosSaidas.T, result3)
        #novoPesosEntradas = self.pesosEntradas - result4
        
        #self.pesosEntradas = np.array(novoPesosEntradas)
        #self.pesosSaidas = np.array(novoPesosSaidas)
            
    def predict(self, training_inputs, outputs):
        acertos = 0
        for x,y in zip(training_inputs, outputs):
                prediction = self.forward(x)
                predictionArredondado = np.array(prediction)
                s = np.argmax(prediction)
                
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
                    
                if (predictionArredondado == np.array(y)).all():
                    acertos = acertos + 1
                    
                print(f'', y, prediction, predictionArredondado, tipo)
        
        print(f'Acetos: {acertos}/{len(training_inputs)}')
        
            