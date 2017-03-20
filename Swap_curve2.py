import numpy as np
import pandas as pd
import datetime as dt
import pickle
from scipy.stats import norm
import matplotlib.pylab as plt
from abc import ABCMeta, abstractmethod
from scipy.sparse import spdiags


market = pickle.load(open('swap_market.p','rb'))
print(market)
er = input('__')



def delta(t1,t2):
    return t2 -t1


class Instrument(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, maturity, rate):
        self.maturity = maturity
        self.rate = rate
        self.name = None
    
    def timeofpayments(self):
        pass
    
    def coupons(self):
        pass







class LIBOR(Instrument):
    def __init__(self,maturity,rate):
        self.name = "LIBOR"
        self.maturity = maturity
        self.rate = rate
        self.payments = {}
        self.price = 1
        self.payment_days = [maturity]

    def get_payment_days(self):
        return [self.maturity]
    
    def setup_payments(self,spotday):
        self.payments[self.maturity] = 1 +delta(spotday,self.maturity)*self.rate
        

class FRA(Instrument):
    def __init__(self,maturity,rate):
        self.name = "LIBOR"
        self.maturity = maturity
        self.rate = rate
        self.payment_days = [maturity]
        self.payments = {}
        self.price = 0
        

    def set_payment_days(self,payment_days):
        self.payment_days = payment_days + self.payment_days
        
    def get_payment_days(self):
        return self.payment_days
    
    def setup_payments(self,spotday):
        self.payments[self.payment_days[-1]] = 1 +delta(self.payment_days[-2],self.payment_days[-1])*self.rate
        self.payments[self.payment_days[-2]] = -1
        
class SWAP(Instrument):
    def __init__(self,maturity,rate):
        self.name = "LIBOR"
        self.maturity = maturity
        self.rate = rate
        self.payment_days = [maturity]
        self.payments = {}
        self.price = 1
        #self.price = 0

    def set_payment_days(self,payment_days):
        self.payment_days = payment_days + self.payment_days
        
    def get_payment_days(self):
        return self.payment_days 
    
    def setup_payments(self,spotday):
        temp = [spotday] + self.get_payment_days()
        for i in range(1,len(temp)):
            self.payments[temp[i]] = delta(temp[i-1],temp[i])*self.rate
        
        self.payments[temp[-1]] += 1
        #self.payments[temp[0]] = -1
        


class Swap(Instrument):
    def __init__(self,maturity,rate):
        self.name = "Swap2"
        self.maturity = maturity
        self.rate = rate
        self.payment_days = swap_days(0.5, self.maturity)
        self.payments = {}
        self.price = 1

    def set_payment_days(self,payment_days):
        self.payment_days = payment_days + self.payment_days
        
    def get_payment_days(self):
        return self.payment_days 
    
    def setup_payments(self):
        for key in self.payment_days:
            self.payments[key] = 0.5*self.rate
        self.payments[self.maturity] += 1

    def swap_days(t0, maturity):
        days = [t0]
        while days[-1] + 0.5 <= maturity:
            days.append(days[-1] + 0.5)
        return days




list_of_instruments = []
for index, row in market.iterrows():
    if row['Source'] == 'LIBOR':
        instument = LIBOR(row['Maturity'].to_pydatetime().date(),row['Market Quotes']/100)
        list_of_instruments.append(instument)
    if row['Source'] == 'Futures':
        instument = FRA(row['Maturity'].to_pydatetime().date(),(100-row['Market Quotes'])/100)
        instument.set_payment_days(P[index])
        list_of_instruments.append(instument)
    if row['Source'] == 'Swap':
        instument = SWAP(row['Maturity'].to_pydatetime().date(),row['Market Quotes']/100)
        instument.set_payment_days(P[index])
        list_of_instruments.append(instument)
    if row['Source'] == 'Swap2':
        instument = Swap(row['Maturity'],row['SwapRate'])
        list_of_instruments.append(instument)



for inst in list_of_instruments:
    inst.setup_payments()
spotday = 0



dates = set()
for istru in list_of_instruments:
    dates.update(istru.get_payment_days())
total = sorted(list(dates))



def building_curve(instruments, times, spotday):
    T = [spotday] + times

    C = np.zeros((len(instruments),len(times)))
    p = np.zeros(len(instruments))

    for (i,inst) in enumerate(instruments):
        p[i] = inst.price
        for j in range(len(total)):
            if times[j] in inst.payment_days:
                C[i,j] = inst.payments[times[j]]



    W_diag = [1/np.sqrt(delta(T[i-1],T[i])) for i in range(1,len(T))]
    W = np.diag(W_diag)


    diags = np.array([0, -1])
    data = np.array([[1]*len(times),[-1]*len(times)])
    M = spdiags(data, diags, len(times), len(times)).toarray()
    M_minus = np.linalg.inv(M)
    W_minus =np.linalg.inv(W)
    one = np.zeros(len(total))
    one[0] = 1
    one = np.atleast_2d(one).T
    A = np.dot(np.dot(C,M_minus),W_minus)
    A_pseudo = np.dot(A.T, np.linalg.inv(np.dot(A,A.T)))
    price = np.atleast_2d(p).T

    Delta = np.dot(A_pseudo,price - np.dot(np.dot(C, M_minus),one))
    discount_curve = np.dot(M_minus, np.dot(W_minus,Delta) + one)



    return total,discount_curve


def curve_building_1st_smoothness(instruments, times, spotday):
    T = [spotday] + times
    tau = T[-1]

    C = np.zeros((len(instruments) + 1, len(T)))
    C[0,0] = 1
    p = np.zeros(len(instruments) + 1)
    p[0] = 1

    for (i,inst) in enumerate(instruments):
        p[i + 1] = inst.price
        for j in range(1,len(T)):
            if T[j] in inst.payment_days:
                C[i + 1,j] = inst.payments[T[j]]

    A = np.zeros((len(T), len(T)))
    for i,t_i in enumerate(T):
        for j,t_j in enumerate(T):
            A[i,j] = 1 + min(t_i, t_j)
    price = np.atleast_2d(p).T
    Z = np.dot(np.dot(C.T,np.linalg.inv(np.dot(C,np.dot(A, C.T)))),price)

    def _f(x):
        out = 0
        for i,t in enumerate(T):
            out += (1 + min(x, t))*Z[i]
        return out

    def _g(x):
        out = 0
        for i,t in enumerate(T):
            out += 1*(x<=t)*Z[i]
        return -out/_f(x)

    return np.vectorize(_f) , np.vectorize(_g)

def curve_building_2st_smoothness(instruments, times, spotday):
    T = [spotday] + times
    tau = T[-1]

    C = np.zeros((len(instruments) + 1, len(T)))
    C[0,0] = 1
    p = np.zeros(len(instruments) + 1)
    p[0] = 1

    for (i,inst) in enumerate(instruments):
        p[i + 1] = inst.price
        for j in range(1,len(T)):
            if T[j] in inst.payment_days:
                C[i + 1,j] = inst.payments[T[j]]

    B = np.zeros((len(T), len(T)))
    for i,t_i in enumerate(T):
        for j,t_j in enumerate(T):
            B[i,j] = 1 - 1/6* min(t_i, t_j)**3 + t_i*t_j*(1 + 0.5*min(t_i, t_j))
    price = np.atleast_2d(p).T
    Z = np.dot(np.dot(C.T,np.linalg.inv(np.dot(C,np.dot(B, C.T)))),price)

    def _f(x):
        out = 0
        for i,t in enumerate(T):
            out += (1 - 1/6*min(x,t)**3 + t/2*min(x,t)**2 - t**2/2*min(x,t) + x*(1+t/2)*t)*Z[i]
        return out

    def _g(x):
        out = 0
        for i, t in enumerate(T):
            out += (t - 0.5*min(x,t)**2 + t*min(x,t)) * Z[i]
        return -out / _f(x)

    return np.vectorize(_f), np.vectorize(_g)


#x,y = building_curve(list_of_instruments,total,spotday)

#plt.plot(x,y, 'o')
f1, g1 = curve_building_1st_smoothness(list_of_instruments,total,spotday)
f2, g2 = curve_building_2st_smoothness(list_of_instruments,total,spotday)
plt.plot(total, [g1(t) for t in total])
plt.plot(total, [g2(t) for t in total])
#plt.plot(total, [curve_building_1st_smoothness(list_of_instruments,total,spotday)(t) for t in total])
plt.show()

