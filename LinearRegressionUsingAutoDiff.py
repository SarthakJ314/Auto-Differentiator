from BackProp import *
import numpy as np
import matplotlib.pyplot as plt

numP = int(input("Number of random points: "))

x = np.random.randint(1,11,numP)
y = np.random.randint(1,11,numP)
w = np.array([Number(1), Number(1)])

lin = np.linspace(0, 10, 100)

lr = 0.001

for i in range(100000):
    loss=0
    for j in range(len(x)):
        loss += (y[j]-(w[0]*x[j]+w[1]))**2
    BackProp(loss)
    w[0] = Number(w[0].x - (lr * w[0].grad))
    w[1] = Number(w[1].x - (lr * w[1].grad))
    BackClear(loss)
    if i % 1000 == 0:
        plt.plot(x, y, 'o')
        plt.plot(lin, w[0].x*lin+w[1].x, color='red')
        plt.show()
