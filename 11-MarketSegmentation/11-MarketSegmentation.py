#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# In[2]:


data = pd.read_csv('Example.csv') #table of customers rated by their satisfaction and loyalty
data #satisfaction-discreete-nonstandardized:{1, 10}, loyaly-continuous-standardized:[-2.5, 2.5]


# In[3]:


plt.scatter(data['Satisfaction'], data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty') #relationship between satisfaction and loyalty


# In[4]:


x = data.copy()


# In[5]:


kmeans = KMeans(2)
kmeans.fit(x)


# In[6]:


clusters = x.copy()
clusters['cluster_pred'] = kmeans.fit_predict(x)


# In[7]:


plt.scatter(clusters['Satisfaction'], clusters['Loyalty'], c=clusters['cluster_pred'], cmap='rainbow')
plt.xlabel = ('Satisfaction')
plt.ylabel = ('Loyalty') #requires standardization for satisfaction. loyalty is pre-standardized


# ## STANDARDIZE VARIABLES

# In[8]:


from sklearn import preprocessing
x_scaled = preprocessing.scale(x)
x_scaled #standardized satisfaction


# ## ELBOW METHOD

# In[9]:


wcss = [] #within-cluster-sum-of-squares (wcss)
for i in range(1, 10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

wcss #wcss for 1 to 9 cluster solutions. 9 is arbitrarily selected.


# In[10]:


plt.plot(range(1, 10), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #4 tips: 5 clusters


# ## EXPLORE CLUSTERING SOLUTIONS and SELECT NUMBER of CLUSTERS

# In[11]:


kmeans_new = KMeans(5) #tries: 2, 3, 4, 5, 9
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)


# In[12]:


clusters_new


# In[13]:


plt.scatter(clusters_new['Satisfaction'], clusters_new['Loyalty'], c=clusters_new['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

