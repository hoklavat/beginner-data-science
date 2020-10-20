#12-HeatmapDendrogram

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv('CountryClustersStandardized.csv', index_col='Country')

x_scaled = data.copy()
x_scaled = x_scaled.drop(['Language'], axis=1)

x_scaled

sns.clustermap(x_scaled, cmap='mako')