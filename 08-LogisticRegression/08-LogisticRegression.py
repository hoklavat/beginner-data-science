#!/usr/bin/env python
# coding: utf-8

# # LOGISTIC REGRESSION
# ### used when possible outcomes are categorical, not numerical.
# ### predicts the probability of an event occuring.

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() #overriding seaborn plot settings instead of matplot


# In[2]:


raw_data = pd.read_csv('Admittance.csv')
raw_data


# In[3]:


data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0}) #categorical to numerical
data


# In[4]:


y = data['Admitted']
x1 = data['SAT']


# In[5]:


plt.scatter(x1, y, color='C0')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('Admitted', fontsize=20)
plt.show()


# ## LINEAR REGRESSION

# In[6]:


x = sm.add_constant(x1)
reg_lin = sm.OLS(y, x)
results_lin = reg_lin.fit()
plt.scatter(x1, y, color='C0')
y_hat = x1 * results_lin.params[1] + results_lin.params[0]
plt.plot(x1, y_hat, lw=2.5, color='C8')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('Admitted', fontsize=20)
plt.show() #linear regression produces meaningless result


# ## LOGISTIC REGRESSION

# In[7]:


reg_log = sm.Logit(y, x) #logistic regression function
results_log = reg_log.fit()

def f(x, b0, b1): #probability calculation function
    return np.array(np.exp(b0+x*b1)/(1+np.exp(b0+x*b1)))

f_sorted = np.sort(f(x1, results_log.params[0], results_log.params[1]))
x_sorted = np.sort(np.array(x1))

plt.scatter(x1, y, color='C0')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('Admitted', fontsize=20)
plt.plot(x_sorted, f_sorted, color='C3')
plt.show() #logistic regression curve shows probability of admission given SAT score

