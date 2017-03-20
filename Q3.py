import numpy as np
import pandas as pd
import scipy
import matplotlib.pylab as plt


def basis_spline(T):
    def _f(x):
        return T+T*min(T,x) -0.5*min(T,x)**2
    return np.vectorize(_f)

def scalar_product(T1,T2):
    return - 1/6*min(T1,T2)**3 + T1*T2*(1+min(T1,T2)/2)


def making_matrix(alpha, T, y):
    A = np.zeros((len(T) + 1, len(T) + 1))
    for i in range(len(T) + 1):
        for j in range(len(T) + 1):
            if j == 0 and i == 0:
                A[i, j] = 0
            elif j == 0:
                A[i, j] = alpha * T[i - 1]
            elif i == 0:
                A[i, j] = T[j - 1]
            elif i == j:
                A[i, j] = alpha * scalar_product(T[i - 1], T[j - 1]) + 1
            else:
                A[i, j] = alpha * scalar_product(T[i - 1], T[j - 1])

    p = np.zeros(len(T) + 1)
    p[1:] = alpha * T * y
    return A, p, np.linalg.solve(A, p)

_,_,beta = making_matrix(10,np.array([2,3,4,5,7,10,20,30]),np.array([-0.79, -0.73, -0.65, -0.55, -0.33, -0.04, 0.54, 0.73]))

def forward_curve(T, beta):
    def _f(x):
        temp = beta[0]
        for i in range(1,len(beta)):
            temp += beta[i]*basis_spline(T[i-1])(x)
        return temp
    return np.vectorize(_f)


curve = forward_curve(np.array([2,3,4,5,7,10,20,30]),beta)

plt.plot(np.arange(0,30,0.1),[curve(x) for x in np.arange(0,30,0.1)])
plt.show()