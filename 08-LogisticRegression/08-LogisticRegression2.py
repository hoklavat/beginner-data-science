#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#Apply commented fix to a statsmodels library if error occurs at reg_results.summary()
#from scipy import stats
#stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


# In[2]:


raw_data = pd.read_csv('Admittance.csv')


# In[3]:


data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})
data


# In[4]:


y = data['Admitted']
x1 = data['SAT']


# In[5]:


x = sm.add_constant(x1)
reg_log = sm.Logit(y, x) #logistic regression function
results_log = reg_log.fit()


# In[6]:


results_log.summary()


# In[7]:


x0 = np.ones(168) #array with 168-elements filled with 1
reg_log = sm.Logit(y, x0)
results_log = reg_log.fit()
results_log.summary() #no change in LL-Null

