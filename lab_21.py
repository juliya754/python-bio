# -*- coding: utf-8 -*-
"""
Лабораторная работа №2.1
Разработка генератора случайных чисел для имитационного моделирования
"""

import numpy as np
import matplotlib.pyplot as plt

# 1) визуальная проверка случайного распределения выбранного языка программирования

fig, ax = plt.subplots(figsize=(18,18), nrows=3, ncols=3)

N = [10**(i+2) for i in range(3)]

for i in range(len(N)):
    x = np.random.rand(N[i])
    
    nbins = 10
    n, bins = np.histogram(x, nbins)
    pdfx = np.zeros(n.size)
    pdfy = np.zeros(n.size)
    for k in range(n.size):
        pdfx[k] = (bins[k]+bins[k+1])*5
        pdfy[k] = n[k]
    ax[i][0].bar(pdfx, pdfy, edgecolor='black')
    ax[i][0].plot(pdfx, pdfy, '--r')
    ax[i][0].set_title('Гистограмма для {0} псевдослучайных чисел'.format(N[i]))
    
# 2) универсальный генератор случайных чисел с заранее заданным распределением
# 1 Задать общее кол-во отрезков гистограммы 
m = 10
P = [[0.01 for j in range(m-1)],[0.01, 0.01, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01]]
for j in range(0,2):
    for i in range(N):
        # Задать общее кол-во чисел гистограммы
        n = N[i]
        # 2 Ввести параметры гистограммы xi и P(xi)
        X = [i for i in range(m)]
        P[j].append(1 - np.sum(P))
        # расчет количества чисел для каждого промежутка
        k=[]
        for i in range(1,m):
            k.append(int(n*P[i]*(X[i]-X[i-1])))
        l = 1
        A = []
        for i in range(1,m-1):
            for j in range(1,k[i]+1):
                A.append(X[i]+np.random.rand()*(X[i]-X[i-1]))
                l+=1
        # перемешивание полученных случайных чисел
        for i in range(1,l):
            p1 = np.random.randint(0,n-1) + 1
            p2 = np.random.randint(0,n-1) + 1
            if p1 != p2:
                b = A[p1]
                A [p1] = A[p2]
                A[p2] = b
            else:
                break
        # запись в файлы значений А (для распределения P[2])
        if j == 2:
            file_name = "A_result_{0}.txt".format(n) 
            L = open(file_name, "w")
            for el in A:
                print(el,file=L)
            L.close()
        # 3) примеры гисторграмм распределения
        n, bins = np.histogram(A, nbins)
        pdfx = np.zeros(n.size)
        pdfy = np.zeros(n.size)
        for k in range(n.size):
            pdfx[k] = (bins[k]+bins[k+1])/1.5
            pdfy[k] = n[k]
        ax[i][j+1].bar(pdfx, pdfy, edgecolor='black')
        ax[i][j+1].plot(pdfx, pdfy, '--r')
        ax[i][j+1].set_title('Гистограмма для {0} псевдослучайных чисел'.format(n))
         