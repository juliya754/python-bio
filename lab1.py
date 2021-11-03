# -*- coding: utf-8 -*-
"""
Лабораторная работа №1
Структурная идентификация математической модели биологического объекта с использованием методов корреляционного анализа
"""

import numpy as np

n = 100
x1 = range(0,n)
 
y1 = [x*x + x*np.random.rand()*100 for x in x1]
y2 = [x + x*np.random.rand()*100 for x in x1]
y3 = [np.random.rand()*100 for x in x1]
y4 = [np.sqrt(x**2. + 0.6)+ np.random.rand() for x in x1]
    
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(t, A, h, T, phi):
    return A*np.exp(-h*t)*np.sin(2*np.pi/T*t + phi)

def R_xy(x,y):
    sigma_X = np.std(x)
    sigma_Y = np.std(y)
    res=np.sum((x-np.average(x))*(y-np.average(y)))/((len(x)-1)*sigma_X*sigma_Y)
    return res
print(R_xy(x1,y1))

popt, pcov = curve_fit(func, x1, y1, (1e3, 1e-2, 1., -1e1), maxfev=10**6)    
A, h, T, phi = popt

fig, ax = plt.subplots(figsize=(12,10),nrows=2, ncols=2)

ax[0][0].scatter(x1, y1, s=30, color='orange')
ax[0][0].plot(x1, func(x1, *popt))
ax[0][0].text(50, 10, 'rxy={0:.2f}'.format(R_xy(x1,y1)))
ax[0][0].set_xlabel('Канал 1')

popt, pcov = curve_fit(func, x1, y2, (1e3, 1e-2, 1., -1e1), maxfev=10**6)    
A, h, T, phi = popt

ax[0][1].scatter(x1, y2, s=30, color='orange')
ax[0][1].plot(x1, func(x1, *popt))
ax[0][1].text(50, 10, 'rxy={0:.2f}'.format(R_xy(x1,y2)))
ax[0][1].set_xlabel('Канал 2')

popt, pcov = curve_fit(func, x1, y3, (1e3, 1e-2, 1., -1e1), maxfev=10**6)    
A, h, T, phi = popt

ax[1][0].scatter(x1, y3, s=30, color='orange')
ax[1][0].plot(x1, func(x1, *popt))
ax[1][0].text(50, 10, 'rxy={0:.2f}'.format(R_xy(x1,y3)))
ax[1][0].set_xlabel('Канал 3')

popt, pcov = curve_fit(func, x1, y4, (1e3, 1e-2, 1., -1e1), maxfev=10**6)    
A, h, T, phi = popt

ax[1][1].scatter(x1, y4, s=30, color='orange')
ax[1][1].plot(x1, func(x1, *popt))
ax[1][1].text(50, 10, 'rxy={0:.2f}'.format(R_xy(x1,y4)))
ax[1][1].set_xlabel('Канал 4')

fig.savefig('lab1.png')