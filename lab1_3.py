# -*- coding: utf-8 -*-
"""
Лабораторная работа №1.3
Параметрическая идентификация нелинейной математической модели биологического объекта на основе эмпирических данных
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

n = 100
x1 = np.array(range(0,n))
 
y4 = np.array([np.sqrt(x**2. + 0.6)+ np.random.rand() for x in x1])
y1 = np.array([x*x + x*np.random.rand()*100 for x in x1])

def R_xy(x,y):
    sigma_X = np.std(x)
    sigma_Y = np.std(y)
    res=np.sum((x-np.average(x))*(y-np.average(y)))/((len(x)-1)*sigma_X*sigma_Y)
    return res

a,b,c = sp.symbols("a,b,c",float=True)
dFda = -2 * np.sum((y1-a*x1**2-b*x1-c)*x1**2)
dFdb = -2 * np.sum((y1-a*x1**2-b*x1-c)*x1)
dFdc = -2 * np.sum(y1-a*x1**2-b*x1-c)
print(dFda)
print(dFdb)
print(dFdc)

res = sp.solve([dFda,dFdb,dFdc],[a,b,c])
print(res)

fig, ax = plt.subplots()

ax.scatter(x1, y1, s=30, color='orange', label = 'Эмпирические данные')
ax.plot(x1, res[a]*x1**2+res[b]*x1+res[c], label = 'y={0:.6f}*$x^2$+{0:.6f}*x+{0:.6f}'.format(res[a],res[b],res[c]))
ax.legend()
ax.text(80, 60, 'rxy={0:.2f}'.format(R_xy(x1,y1)))
ax.set_xlabel('Канал 1')

acc1 = (res[a]*np.sum(x1**4)+res[b]*np.sum(x1**3)+res[c]*np.sum(x1**2)-np.sum(x1**2*y1))/np.sum(x1**2*y1)*100
acc2 = (res[a]*np.sum(x1**3)+res[b]*np.sum(x1**2)+res[c]*np.sum(x1)-np.sum(x1*y1))/np.sum(x1*y1)*100
acc3 = (res[a]*np.sum(x1**2)+res[b]*np.sum(x1)+res[c]*n-np.sum(x1**2*y1))/np.sum(y1)*100
print('acc1 = {0:e}, acc2 = {0:e}, acc3 = {0:e}'.format(acc1, acc2,acc3))

fig.savefig('lab13.png')