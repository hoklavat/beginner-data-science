#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# ## SCALAR

# In[2]:


s = 5


# In[3]:


s


# ## VECTOR

# In[4]:


v = np.array([5, -2, 4]) #column vector


# In[5]:


v


# In[6]:


v = v.reshape(3, 1) #row vector


# In[7]:


v


# ## MATRICE

# In[8]:


m = np.array([[5, 12, 6], [-3, 0, 14]])


# In[9]:


m


# ## DATA TYPES

# In[10]:


type(s) #integer


# In[11]:


type(v) #1-dimensional array


# In[12]:


type(m) #2-dimensional array


# In[13]:


s_array = np.array(5)
s_array


# In[14]:


type(s_array)


# ## DATA SHAPES

# In[15]:


m.shape


# In[16]:


v.shape


# In[17]:


v.reshape(1, 3) #reshape without changing data


# In[18]:


v.reshape(3, 1)


# In[19]:


s.shape #error, no array


# In[20]:


s_array.shape


# ## MATRICE OPERATIONS

# In[21]:


m1 = np.array([[5, 12, 6], [-3, 0, 14]])
m1


# In[22]:


m2 = np.array([[9, 8, 7], [1, 3, -5]])
m2


# In[23]:


m1 + m2 #addition, same dimensions


# In[24]:


m1 - m2 #subtraction


# In[25]:


m1 + 1 #add 1 to each element


# In[26]:


m1.T #transposing matrice


# In[27]:


v.T #transposing vector


# In[28]:


v.reshape(1, 3).T


# In[29]:


np.array([5]).T #scalar transpose to self


# In[30]:


v1 = np.array([2, 8, -4])
v2 = np.array([1, -7, 3])


# In[31]:


np.dot(v1, v2) #dot product of two vectors


# In[32]:


np.dot(5, 6) #dot product of scalars


# In[33]:


5 * v1 # scalar * vector


# In[34]:


m3 = np.array([[5, 12, 6], [-3, 0, 14]])
m4 = np.array([[2, -1], [8, 0], [3, 0]])
np.dot(m3, m4) #matrice * matrice

