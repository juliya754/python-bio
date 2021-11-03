# -*- coding: utf-8 -*-
"""
Лабораторная работа №1.2
Параметрическая идентификация линейной математической модели биологического объекта на основе эмпирических данных
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

n = 100
x1 = range(0,n)
 
y4 = [np.sqrt(x**2. + 0.6)+ np.random.rand() for x in x1]

def R_xy(x,y):
    sigma_X = np.std(x)
    sigma_Y = np.std(y)
    res=np.sum((x-np.average(x))*(y-np.average(y)))/((len(x)-1)*sigma_X*sigma_Y)
    return res

a,b = sp.symbols("a,b",float=True)
dFda = -2 * np.sum((np.array(y4)-a*np.array(x1)-b)*x1)
dFdb = -2 * np.sum(np.array(y4)-a*np.array(x1)-b)
print(dFda)
print(dFdb)

res = sp.solve([dFda,dFdb],[a,b])
print(res)

fig, ax = plt.subplots()

ax.scatter(x1, y4, s=30, color='orange', label = 'Эмпирические данные')
ax.plot(x1, res[a]*np.array(x1)+res[b], label = 'y={0:.6f}*x+{0:.6f}'.format(res[a],res[b]))
ax.legend()
ax.text(0, 75, 'rxy={0:.2f}'.format(R_xy(x1,y4)))
ax.set_xlabel('Канал 4')

acc1 = (res[a]*np.sum(np.array(x1)**2)+res[b]*np.sum(x1)-np.sum(np.array(x1)*np.array(y4)))/np.sum(np.array(x1)*np.array(y4))*100
acc2 = (res[a]*np.sum(np.array(x1))+res[b]*n-np.sum(np.array(y4)))/np.sum(np.array(y4))*100
print('acc1 = {0:e}, acc2 = {0:e}'.format(acc1, acc2))

fig.savefig('lab12.png')