from BackProp import *
import numpy as np
import matplotlib.pyplot as plt

x_s = np.linspace(-5, 5, 10000)

def f(x):
   return 2**x + sigmoid(x) #Enter Function here

plot_x = []
plot_x2 = []
plot_grad = []
for i in x_s:
    x = Number(i)
    y = f(x)
    BackProp(y)
    print(x.grad)
    plot_x.append(i)
    plot_x2.append(y.x)
    plot_grad.append(x.grad)
plt.plot(plot_x, plot_x2)
plt.plot(plot_x, plot_grad)
plt.setp(plt.gca(), ylim=(-5, 5))
plt.show()
while True:
    inp = input('x: ')
    if inp == 'stop':
        break
    x = Number(int(inp))
    y = f(x)
    print(y.x)
