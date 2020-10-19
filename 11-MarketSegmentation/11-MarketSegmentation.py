#11-MarketSegmentation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('Example.csv') #table of customers rated by their satisfaction and loyalty
data #satisfaction-discreete-nonstandardized:{1, 10}, loyaly-continuous-standardized:[-2.5, 2.5]

plt.scatter(data['Satisfaction'], data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty') #relationship between satisfaction and loyalty

x = data.copy()

kmeans = KMeans(2)
kmeans.fit(x)

clusters = x.copy()
clusters['cluster_pred'] = kmeans.fit_predict(x)

plt.scatter(clusters['Satisfaction'], clusters['Loyalty'], c=clusters['cluster_pred'], cmap='rainbow')
plt.xlabel = ('Satisfaction')
plt.ylabel = ('Loyalty') #requires standardization for satisfaction. loyalty is pre-standardized

#STANDARDIZE VARIABLES
from sklearn import preprocessing
x_scaled = preprocessing.scale(x)
x_scaled #standardized satisfaction

#ELBOW METHOD
wcss = [] #within-cluster-sum-of-squares (wcss)
for i in range(1, 10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

wcss #wcss for 1 to 9 cluster solutions. 9 is arbitrarily selected.

plt.plot(range(1, 10), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #4 tips: 5 clusters

#EXPLORE CLUSTERING SOLUTIONS and SELECT NUMBER of CLUSTERS
kmeans_new = KMeans(5) #tries: 2, 3, 4, 5, 9
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)

clusters_new

plt.scatter(clusters_new['Satisfaction'], clusters_new['Loyalty'], c=clusters_new['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')