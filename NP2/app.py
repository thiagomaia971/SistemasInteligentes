import pandas as pd
from MLP import MLP

def pegarBase():
    dataset = pd.read_csv('databases/Trabalho Pratico - MLP - ClassificacYaYo de PadroYes - trein.csv')
    
    linhasFinais = []
    linhas = dataset.iloc[:, 0:33].values
    for linha in linhas:
        novaLinha = []
        for x in range(5):
            novaLinha.append(linha[x])

        if linha[5] == 1: 
            novaLinha.append('A')
        elif linha[6] == 1:
            novaLinha.append('B')
        elif linha[7] == 1: 
            novaLinha.append('C')
        
        linhasFinais.append(novaLinha)
        

    return linhasFinais

def pegarInput(base):
    inputBase = []
    for dado in base:
        vectorAux = []
        for x in range(5):
            vectorAux.append(dado)

        inputBase.append(vectorAux)

    return inputBase

def pegarOutPut(base):
    outPutBase = []
    for dado in base:
        outPutBase.append(dado[5])

    return outPutBase    

base = pegarBase()

inputExecution = pegarInput(base)
outPutExecution = pegarOutPut(base) 

MLP(len(input[0]))