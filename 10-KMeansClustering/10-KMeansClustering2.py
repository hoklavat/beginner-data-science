#10-KMeansClustering2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('CountryClusters.csv')
data

data_mapped = data.copy()
data_mapped['Language'] = data_mapped['Language'].map({'English':0, 'French':1, 'German':2}) #categorical to numerical
data_mapped

plt.scatter(data_mapped['Longitude'], data_mapped['Latitude'])
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()

x = data_mapped.iloc[:, 1:4] #[all rows, 1st,2nd,3rd columns]
x

kmeans = KMeans(2) #number of clusters: 2
kmeans.fit(x) #cluster with respect to 3 variables: latitude, longitude and language

identified_clusters = kmeans.fit_predict(x)
identified_clusters

data_with_clusters = data_mapped.copy()
data_with_clusters['Cluster'] = identified_clusters
data_with_clusters

plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'],c=data_with_clusters['Cluster'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90) 

kmeans.inertia_ #for two clusters: within-cluster-sum-of-squares; sum of distances to centroid

wcss = []
for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(kmeans.inertia_)

wcss #minimize wcss to obtain best possible cluster.

#ELBOW METHOD
number_clusters = range(1, 7)
plt.plot(number_clusters, wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-cluster sum of squares') #number of clusters: 3