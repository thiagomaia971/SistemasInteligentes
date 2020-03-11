import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perceptron import Perceptron
from adaline import Adaline

# ✔️ Embaralhar
# ✔️ Dividar as bases em dois
#    ✔️ 25%(base de teste) e 75%(base de treinamento)

# ✔️ Rodar 5 treinamentos para os dois alg
# ✔️ taxa de aprendizado (n) = 2,5*10^-3
# ✔️ epocas < 1000
# precisão de 10^-6 (adaline)
# ✔️ usar (base de treinamento)

# Excel:
# Valores iniciais do vetor de peso (antes do treinamento)
# Valores finais do vetor de peso (após do treinamento)
# E a quantidade de épocas
# Duas tabelas, uma para cada alg.

# Para o Adaline, deixa pro ultimo

def pegarBase():
    dataset = pd.read_csv('databases/Perceptron - rocks and mines/sonar.all-data')
    dataset.replace(['M', 'R'], [1, 0], inplace=True)
    return dataset.iloc[:, 0:61].values
    
def embaralhar(base):
    np.random.shuffle(base)

def pegarBaseTreinamento(base):
    l = int(0.75*len(base))
    treinamento = [None]*l
    for e in range(l):
        treinamento[e] = base[e]
    return treinamento

def pegarBaseTeste(base):
    l = int(0.25*len(base))
    teste = [None]*l
    baseReversed = list(reversed(base))
    for e in range(l):
        teste[e] = baseReversed[e]
    return teste

def pegarInputTreinamento(baseTreinamento):
    inputTreinamento = [None]*len(baseTreinamento)
    for e in range(len(baseTreinamento)):
        inputTreinamento[e] = baseTreinamento[e][0:60]
    return inputTreinamento

def pegarOutputTreinamento(baseTreinamento):
    outputTreinamento = [None]*len(baseTreinamento)
    for e in range(len(baseTreinamento)):
        outputTreinamento[e] = baseTreinamento[e][60:]
    return outputTreinamento

def rodarPerceptron(inputTreinamento, outputTreinamento):
    print(">> Perceptron")
    for e in range(5):
        print(f">> Treinamento {e + 1}")
        perceptron = Perceptron(len(inputTreinamento[0]))
        perceptron.train(inputTreinamento, outputTreinamento)
        print("")
    print("")
    print("")

def rodarAdaline(inputTreinamento, outputTreinamento):
    print(">> Adaline")
    for e in range(5):
        print(f">> Treinamento {e + 1}")
        perceptron = Adaline(len(inputTreinamento[0]))
        perceptron.train(inputTreinamento, outputTreinamento)
        print("")

base = pegarBase()
embaralhar(base)
baseTreinamento = pegarBaseTreinamento(base)
baseTeste = pegarBaseTeste(base)

inputTreinamento = pegarInputTreinamento(baseTreinamento)
outputTreinamento = pegarOutputTreinamento(baseTreinamento)

# rodarPerceptron(inputTreinamento, outputTreinamento)
rodarAdaline(inputTreinamento, outputTreinamento)
    

#plt.xlim(-1,3)
#plt.ylim(-1,3)
#for i in range(len(d)):
#    if d[i] == 1:
#        plt.plot(X[i, 0], X[i, 1], 'ro')
#    else:
#        plt.plot(X[i, 0], X[i, 1], 'bo')
#        
#f = lambda x: (p.weights[0]/p.weights[2]) - (p.weights[1]/p.weights[2] * x)
#xH = list(range(-1,3))
#yH = list(map(f, xH))
#plt.plot(xH, yH, 'y-')


#p.train(X, d)


#print(p.predict(X[0]))
#print(p.predict(X[1]))
#print(p.predict(X[2]))
#print(p.predict(X[3]))
#
#plt.xlim(-1,3)
#plt.ylim(-1,3)
#for i in range(len(d)):
#    if d[i] == 1:
#        plt.plot(X[i, 0], X[i, 1], 'ro')
#    else:
#        plt.plot(X[i, 0], X[i, 1], 'bo')
#        
#f = lambda x: (p.weights[0]/p.weights[2]) - (p.weights[1]/p.weights[2] * x)
#xH = list(range(-1,3))
#yH = list(map(f, xH))
#plt.plot(xH, yH, 'g-')
    