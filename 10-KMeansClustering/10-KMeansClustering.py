#10-KMeansClustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans #KMeans clustering class

data = pd.read_csv('CountryClusters.csv') #country, global coordinates(latitude, longitude), language
data

plt.scatter(data['Longitude'], data['Latitude'])
plt.xlim(-180, 180)
plt.ylim(-90, 90) #countries by their coordinates, australia at bottom-right

x = data.iloc[:, 1:3] #slice table: [all rows, 1st-2nd columns]
x

kmeans = KMeans(3) #number of clusters: 3, arbitrary selection
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
identified_clusters

data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters
data_with_clusters

plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'],c=data_with_clusters['Cluster'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90) #plot three clusters

