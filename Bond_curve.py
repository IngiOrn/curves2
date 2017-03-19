import numpy as np
import pandas as pd
import datetime as dt
import pickle
import matplotlib.pylab as plt
from abc import ABCMeta, abstractmethod
from scipy.sparse import spdiags


market = pd.read_pickle('market_bonds.p')
P = pickle.load( open( "Pdays.p", "rb" ) )


def delta(t1,t2):
    return (min(t2.day,30) + max(30-t1.day, 0))/360 + (t2.month - t1.month -1)/12 + t2.year - t1.year

spotday = dt.date(1996,9,4)


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


class BOND(Instrument):
    def __init__(self,maturity,coupon,price, next_cp):
        self.name = "BOND"
        self.maturity = maturity
        self.rate = coupon/2
        self.payments = {}
        self.price = price
        self.next_cp = next_cp
        self.payment_days = bond_days(self.next_cp,self.maturity)
        
    def print_maturity(self):
        print(self.maturity)
    
    def get_payment_days(self):
        return self.payment_days
    
    def setup_payments(self):
        for key in self.payment_days:
            self.payments[key] = self.rate
        self.payments[self.maturity] += 100


def bond_days(first, maturity):
    days = [first]
    while days[-1] < maturity:
        if days[-1].month <= 6:
            days.append(dt.date(days[-1].year, days[-1].month+6, days[-1].day))
        else:
            days.append(dt.date(days[-1].year +1, days[-1].month-6, days[-1].day))
    return days 


class LIBOR(Instrument):
    def __init__(self,maturity,rate):
        self.name = "LIBOR"
        self.maturity = maturity
        self.rate = rate
        self.payments = {}
        self.price = 1
        self.payment_days = [maturity]
        
    def print_maturity(self):
        print(self.maturity)
    
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
        
        
    def print_maturity(self):
        print(self.maturity)
    
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
        
    def print_maturity(self):
        print(self.maturity)
    
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
    if row['Source'] == 'Bond':
        instument = BOND(row['Maturity'],row['coupon'],row['Price'], row['Next'])
        list_of_instruments.append(instument)


for inst in list_of_instruments:
    inst.setup_payments()

dates = set()
for istru in list_of_instruments:
    dates.update(istru.get_payment_days())
total = sorted(list(dates))
T = [spotday]+ total

def curve_building(instruments, times, spotday):
    T = [spotday] + times
    C = np.zeros((len(instruments),len(times)))
    p = np.zeros(len(instruments))
    for (i,inst) in enumerate(instruments):
        p[i] = inst.price
        for j in range(len(times)):
            if total[j] in inst.payment_days:
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
    plt.plot(times,discount_curve)
    plt.show()

curve_building(list_of_instruments,total, spotday)