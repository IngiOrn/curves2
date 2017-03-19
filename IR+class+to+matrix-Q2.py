import numpy as np
import pandas as pd
import datetime as dt
import pickle
import matplotlib.pylab as plt

market = pd.read_pickle('market.p')
P = pickle.load( open( "P.p", "rb" ) )

print(market)
print(P)


tr = input('___')

market = market[0:8]


def delta(t1,t2):
    return (t2 -t1)/dt.timedelta(360)


# In[537]:

spotday = dt.date(2012,10,3)


# In[538]:

from abc import ABCMeta, abstractmethod
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
        


# In[540]:

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



# In[541]:

for inst in list_of_instruments:
    inst.setup_payments(spotday)


# In[542]:

dates = set()
for istru in list_of_instruments:
    dates.update(istru.get_payment_days())
total = sorted(list(dates))
T = [spotday]+ total


# In[543]:

C = np.zeros((len(list_of_instruments),len(total)))
p = np.zeros(len(list_of_instruments))


# In[544]:

pd.DataFrame(C)


# In[545]:

p


# In[546]:

for (i,inst) in enumerate(list_of_instruments):
    p[i] = inst.price
    for j in range(len(total)):
        if total[j] in inst.payment_days:
            C[i,j] = inst.payments[total[j]]


# In[547]:

pd.DataFrame(C)


# In[548]:

W_diag = [1/np.sqrt(delta(T[i-1],T[i])) for i in range(1,len(T))]


# In[549]:

W = np.diag(W_diag)


# In[550]:

W


# In[551]:

from scipy.sparse import spdiags
#M = np.zeros((len(total), len(total)))


# In[552]:

diags = np.array([0, -1])
data = np.array([[1]*len(total),[-1]*len(total)])
M = spdiags(data, diags, len(total), len(total)).toarray()


# In[553]:

M_minus = np.linalg.inv(M)


# In[554]:

M_minus


# In[555]:

W_minus =np.linalg.inv(W)


# In[556]:

one = np.zeros(len(total))
one[0] = 1
one = np.atleast_2d(one).T


# In[557]:

C.shape


# In[558]:

M_minus.shape


# In[559]:

W_minus.shape


# In[560]:

A = np.dot(np.dot(C,M_minus),W_minus)


# In[ ]:




# In[561]:

A_pseudo = np.dot(A.T, np.linalg.inv(np.dot(A,A.T)))


# In[562]:

price = np.atleast_2d(p).T


# In[563]:

Delta = np.dot(A_pseudo,price - np.dot(np.dot(C, M_minus),one))


# In[564]:

discount_curve = np.dot(M_minus, np.dot(W_minus,Delta) + one)


# In[565]:

plt.plot(total,discount_curve)


# In[566]:

discount_curve


# In[570]:

(discount_curve[-2]/discount_curve[-1] - 1)/(delta(total[-2],total[-1]))


# In[569]:

delta(total[-2],total[-1])


# In[ ]:




# In[ ]:




# In[ ]:




# In[134]:

np.dot(M_minus, W_minus)


# In[81]:

p


# In[30]:

P[9]


# In[ ]:




# In[ ]:




# In[98]:

l.setup_payments(dt.date(2015,1,1))


# In[99]:

l.payments[dt.date(2015,5,6)]


# In[100]:

f = FRA(dt.date(2015,5,6),0.01)


# In[101]:

f.set_payment_days([dt.date(2015,3,3)])


# In[102]:

f.get_payment_days()


# In[103]:

f.setup_payments(dt.date(2015,1,1))


# In[104]:

f.price


# In[105]:

s = SWAP(dt.date(2017,1,1), 0.04)


# In[106]:

s.set_payment_days([dt.date(2016,1,1)])


# In[107]:

s.get_payment_days()


# In[108]:

s.setup_payments(dt.date(2015,1,1))


# In[109]:

s.payments


# In[110]:

dt.timedelta(30)


# In[111]:

def delta(t1,t2):
    return (t2 -t1)/dt.timedelta(360)


# In[112]:

delta(dt.date(2015,1,2), dt.date(2015,6,6))


# In[113]:

R = {dt.date(2015,1,2):3}


# In[114]:

R[dt.date(2015,1,2)]


# In[115]:

M =[l,f,s]


# In[126]:

dates = set()
for istru in M:
    dates.update(istru.get_payment_days())
total = sorted(list(dates))


# In[130]:

total = sorted(list(dates))


# In[93]:

T = [spotday]+ total


# In[94]:

T


# In[92]:

total


# In[ ]:




# In[ ]:




# In[124]:

dates.update([1,3,4,5])


# In[125]:

dates


# In[132]:

total.index(total[2])


# In[ ]:




# In[ ]:



