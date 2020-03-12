import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perceptron import Perceptron
from adaline import Adaline
from log import Log
from activation_functions import signum_function

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

log = Log()

def pegarBase():
    dataset = pd.read_csv('databases/Perceptron - rocks and mines/sonar.all-data')
    dataset.replace(['M', 'R'], [-1, 1], inplace=True)
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
    
def testar(resultados, ehPerceptron):
    for e in range(5):
        alg = resultados[e]
        log.print(f'>>Teste {e + 1}')
        hits = 0
        
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for inputs, label in zip(inputTeste, outputTeste):
            predict = alg.predict(inputs)
            if not ehPerceptron:
                predict = signum_function(predict)

            if predict == label:
                hits += 1
            
            if label == 1 and predict == label:
                tp += 1
            if label == 1 and predict != label:
                fp += 1

            if label == -1 and predict == label:
                tn += 1
            if label == -1 and predict != label:
                fn += 1

        log.print(f'Total: {len(inputTeste)}')
        log.print(f'Hits: {hits}')
        log.print(f'Misses: {len(inputTeste)-hits}')
        log.print(f'Precision: {(hits/len(inputTeste)) * 100}%')
        log.print(f'Matriz: [{tp},{fp}][{tn},{fn}]\n')


def rodarPerceptron(inputTreinamento, outputTreinamento):
    log.print(">> Perceptron")
    perceptrons = [None]*5
    for e in range(5):
        log.print(f">> Treinamento {e + 1}")
        perceptron = Perceptron(log, len(inputTreinamento[0]))
        perceptron.train(inputTreinamento, outputTreinamento)
        log.print("")
        perceptrons[e] = perceptron
    testar(perceptrons, True)

def rodarAdaline(inputTreinamento, outputTreinamento):
    log.print(">> Adaline")
    adalines = [None]*5
    for e in range(5):
        log.print(f">> Treinamento {e + 1}")
        adaline = Adaline(log, len(inputTreinamento[0]))
        adaline.train(inputTreinamento, outputTreinamento)
        adalines[e] = adaline
    testar(adalines, False)

base = pegarBase()
embaralhar(base)
baseTreinamento = pegarBaseTreinamento(base)
baseTeste = pegarBaseTeste(base)

log.print(baseTreinamento)
log.print("")

inputTreinamento = pegarInputTreinamento(baseTreinamento)
outputTreinamento = pegarOutputTreinamento(baseTreinamento)

inputTeste = pegarInputTreinamento(baseTeste)
outputTeste = pegarOutputTreinamento(baseTeste)

rodarPerceptron(inputTreinamento, outputTreinamento)
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


#log.print(p.predict(X[0]))
#log.print(p.predict(X[1]))
#log.print(p.predict(X[2]))
#log.print(p.predict(X[3]))
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
    