import pandas as pd
import matplotlib.pyplot as plt

from perceptron import Perceptron

dataset = pd.read_csv('databases/iris.data')
dataset.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [1, 0, 0], inplace=True)
X = dataset.iloc[:, 0:4].values
d = dataset.iloc[:, 4:].values

p = Perceptron(len(X[0]), epochs=10000)

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


p.train(X, d)


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
    