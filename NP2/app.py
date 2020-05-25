import pandas as pd
from MLP import MLP
import matplotlib.pyplot as plt

rodarXor = False

def pegarBase(baseDeDados):
    dataset = pd.read_csv(baseDeDados)
    return dataset

def pegarInput(dataset):
    if rodarXor:
        return dataset.iloc[:, 0:2].values
    return dataset.iloc[:, 0:4].values

def pegarOutPut(dataset):
    if rodarXor:
        return dataset.iloc[:, 2:3].values
    return dataset.iloc[:, 4:7].values

if rodarXor:    
    base = pegarBase('databases/xor.csv')
    testeBase = pegarBase('databases/xor.csv')
else:   
    base = pegarBase('databases/trein.csv')
    testeBase = pegarBase('databases/teste.csv')

inputExecution = pegarInput(base)
outPutExecution = pegarOutPut(base) 

mlp = MLP(len(inputExecution[0]), len(outPutExecution[0]))
mlp.train(inputExecution, outPutExecution)
    
mlp.predict(pegarInput(testeBase), pegarOutPut(testeBase))