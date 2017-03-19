
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import datetime as dt
import pickle


# In[31]:

market = pd.read_pickle('market.p')
P = pickle.load( open( "P.p", "rb" ) )


# In[20]:

def delta(t1,t2):
    return (t2 -t1)/dt.timedelta(360)


# In[23]:

spotday = dt.date(2012,10,3)


# In[4]:

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
    


# In[60]:

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
        


# In[61]:

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



# In[62]:

for inst in list_of_instruments:
    inst.setup_payments(spotday)


# In[63]:

list_of_instruments[5].payments


# In[64]:

enumerate(list_of_instruments)


# In[82]:

for (i,inst) in enumerate(list_of_instruments):
    p[i] = inst.price
    for j in range(len(total)):
        if total[j] in inst.payment_days:
            C[i,j] = inst.payments[total[j]]


# In[78]:

C[9]


# In[83]:

p


# In[47]:

dates = set()
for istru in list_of_instruments:
    dates.update(istru.get_payment_days())
total = sorted(list(dates))   


# In[79]:

C = np.zeros((len(list_of_instruments),len(total)+1))
p = np.zeros(len(list_of_instruments))


# In[95]:

W_diag = [1/np.sqrt(delta(T[i-1],T[i])) for i in range(1,len(T))]


# In[103]:

#W = np.zeros((len(T),len(T)))


# In[102]:

W = np.diag(W_diag)


# In[106]:

from scipy.sparse import spdiags
#M = np.zeros((len(total), len(total)))


# In[114]:

diags = np.array([0, -1])
data = np.array([[1]*len(total),[-1]*len(total)])
M_minus =spdiags(data, diags, len(total), len(total)).toarray()


# In[115]:

M_minus


# In[118]:

W_minus =np.linalg.inv(W)


# In[126]:

one = np.zeros(len(total))
one[0] = 1
one = np.atleast_2d(one).T


# In[131]:

C.shape


# In[132]:

M_minus.shape


# In[133]:

W_minus.shape


# In[137]:

A = np.dot(np.dot(C,M_minus),W_minus)


# In[ ]:




# In[148]:

A_pseudo = np.dot(A.T, np.linalg.inv(np.dot(A,A.T)))


# In[141]:

price = np.atleast_2d(p).T


# In[150]:

Delta = np.dot(A_pseudo,price - np.dot(np.dot(C, M_minus),one))


# In[152]:

discount_curve = np.dot(M_minus, np.dot(W_minus,Delta) + one)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




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



