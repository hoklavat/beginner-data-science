#!/usr/bin/env python
# coding: utf-8

# ## BINARY PREDICTOR
# ### corresponding to dummy variables in linear regression, that used to express categorical data as numerical

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


raw_data = pd.read_csv('BinaryPredictors.csv')
raw_data


# In[3]:


data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0}) #convert categorical to numerical
data['Gender'] = data['Gender'].map({'Female':1, 'Male':0})
data


# In[4]:


y = data['Admitted']
x1 = data[['SAT', 'Gender']]


# In[5]:


x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()
results_log.summary()


# In[6]:


np.exp(1.9449) #given the same sat score female is seven times more likely to be admitted than male.


# ## ACCURACY

# In[7]:


np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)}) #only take 2 decimal digits
results_log.predict() #predicted values, non zero or one is probability.


# In[8]:


np.array(data['Admitted']) #actual values


# In[9]:


results_log.pred_table() #table of predicted and actual values


# In[10]:


cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0:'Actual 0', 1:'Actual 1'})
cm_df #confusion matrix: formatted table for comparision of predicted and actual values


# In[11]:


cm = np.array(cm_df)
accuracy_train = (cm[0,0]+cm[1,1])/cm.sum()
accuracy_train #accuracy is 94%


# ## TESTING

# In[12]:


test = pd.read_csv('TestDataSet.csv')
test


# In[13]:


test['Admitted'] = test['Admitted'].map({'Yes':1, 'No':0})
test['Gender'] = test['Gender'].map({'Female':1, 'Male':0})
test


# In[14]:


x


# In[15]:


test_actual = test['Admitted']
test_data = test.drop(['Admitted'], axis=1)
test_data = sm.add_constant(test_data)
test_data = test_data[x.columns.values]
test_data


# In[16]:


def confusion_matrix(data, actual_values, model):
    pred_values = model.predict(data)
    bins = np.array([0, 0.5, 1])
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
    accuracy = (cm[0, 0]+cm[1, 1])/cm.sum()
    return cm, accuracy


# In[17]:


cm = confusion_matrix(test_data, test_actual, results_log)
cm


# In[18]:


cm_df = pd.DataFrame(cm[0])
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0:'Actual 0', 1:'Actual 1'})
cm_df


# In[19]:


print('Misclassification Rate: ' + str((1+1)/19)) #failure rate

