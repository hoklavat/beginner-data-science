#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


m1 = np.array([[5, 12, 6], [-3, 0, 14]])
m1


# In[3]:


m2 = np.array([[9, 8, 7], [1, 3, -5]])
m2


# In[4]:


t = np.array([m1, m2])
t


# In[5]:


t.shape


# In[6]:


t_manual = np.array([[[5, 12, 6], [-3, 0, 14]], [[9, 8, 7], [1, 3, -5]]])
t_manual

