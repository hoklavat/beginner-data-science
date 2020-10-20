#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


data = pd.read_csv('CountryClustersStandardized.csv', index_col='Country')


# In[3]:


x_scaled = data.copy()
x_scaled = x_scaled.drop(['Language'], axis=1)


# In[4]:


x_scaled


# In[5]:


sns.clustermap(x_scaled, cmap='mako')

