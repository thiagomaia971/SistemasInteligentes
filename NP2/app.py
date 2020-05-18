import pandas as pd
from MLP import MLP

def pegarBase():
    dataset = pd.read_csv('databases/teste.csv')
    
    linhasFinais = []
    linhas = dataset.iloc[:, 0:33].values
    for linha in linhas:
        novaLinha = []
        for x in range(7):
            novaLinha.append(linha[x])
            
        linhasFinais.append(novaLinha)
        

    return linhasFinais

def pegarInput(base):
    inputBase = []
    for dado in base:
        vectorAux = []
        for x in range(4):
            vectorAux.append(dado[x])

        inputBase.append(vectorAux)

    return inputBase

def pegarOutPut(base):
    outPutBase = []
    for dado in base:
        vectorAux = []
        for x in range(3):
            vectorAux.append(dado[4+x])

        outPutBase.append(vectorAux)

    return outPutBase    

base = pegarBase()

inputExecution = pegarInput(base)
outPutExecution = pegarOutPut(base) 

mlp = MLP(len(inputExecution[0]), len(outPutExecution[0]))
mlp.train(inputExecution, outPutExecution)
print("oi")