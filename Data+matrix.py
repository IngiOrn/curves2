
# coding: utf-8

# In[68]:

import pandas as pd
import datetime as dt
f = lambda x: x.date()


# In[3]:

days = pd.read_excel('paymentdays.xlsx')
matrix = pd.read_excel('matrix.xlsx')
swapdays = pd.read_excel('swap_days.xlsx') 
futuredays = pd.read_excel('F_days.xlsx')
swapmat = pd.read_excel('swapm.xlsx')


# In[4]:

days['date'] = days['days'].dt.to_pydatetime()
swapdays['date'] = swapdays['days'].dt.to_pydatetime()
futuredays['date'] = futuredays['days'].dt.to_pydatetime()
swapmat['date'] = swapmat['days'].dt.to_pydatetime()


# In[5]:

days['date'] = days['date'].apply(f)
swapdays['date'] = swapdays['date'].apply(f)
futuredays['date'] = futuredays['date'].apply(f)
swapmat['date'] = swapmat['date'].apply(f)


# In[6]:

swdays = swapdays['date'].as_matrix().tolist()
fdays = futuredays['date'].as_matrix().tolist()


# In[9]:

swdays = swapdays['date'].as_matrix()


# In[72]:

#pd.DataFrame(swdays)


# In[13]:

swapmat = swapmat['date'].as_matrix()


# In[18]:

swhere = []
for d in swapmat:
    swhere.append(swdays[swdays < d].tolist())


# In[69]:

swapmat


# In[73]:

swhere


# In[27]:

paymentdays = days['days'].dt.to_pydatetime()


# In[4]:

paymentdays[20] 


# In[9]:

matrix['Maturity Dates'][2].to_pydatetime().date()


# In[35]:

pd.DataFrame([(None,2),(2,3)], columns='A')


# In[42]:

matrix


# In[43]:

matrix['Maturity'] = matrix['Maturity Dates'].dt.to_pydatetime()


# In[56]:

matrix.to_pickle("market.p") 


# In[57]:

pd.read_pickle('market.p')


# In[27]:

paymentdays[1] - paymentdays[0]


# In[52]:

paymentdays[3].date()


# In[65]:

P = [[None],[None],[None],[None]]+[[fdays[i]] for i in range(len(fdays))]+[swhere[i] for i in range(len(swapmat))]


# In[66]:

P


# In[23]:

[swhere[i] for i in range(len(swapmat))]


# In[36]:

pp= pd.DataFrame(list(range(len(P))), columns=['p_index'])


# In[49]:

new = pd.concat(matrix, matrix) #,join='inner')


# In[37]:

P


# In[50]:

import pickle


# In[67]:

pickle.dump(P, open( "P.p", "wb" ) )


# In[52]:

P2 = pickle.load( open( "P.p", "rb" ) )


# In[54]:

P2[0]


# In[61]:

for index, row in matrix.iterrows():
    print(row['Maturity'].date())


# In[ ]:



