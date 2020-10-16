#05-Standardization
#standardization: process of transforming data into a standard scale for predict method. (original variable - mean)/(standard deviation)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

data = pd.read_csv("MultipleLinearRegression.csv")
data.head()

data.describe()

x = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x)

x_scaled = scaler.transform(x)

x_scaled

# Regression with Scaled Features
reg = LinearRegression()
reg.fit(x_scaled,y)

reg.coef_

reg.intercept_

# Summary Table
reg_summary = pd.DataFrame([['Bias'],['SAT'],['Rand 1,2,3']], columns=['Features']) #Machine Learning Jargon: intercept >> bias
reg_summary['Weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1] #Machine Learning Jargon: coefficients >> weights

reg_summary #weight closer to zero has smaller impact

# Making Predictions with Standardized Coefficients (Weights)
new_data = pd.DataFrame(data=[[1700,2], [1800,1]], columns=['SAT', 'Rand 1,2,3'])
new_data

reg.predict(new_data) #should be standardized

new_data_scaled = scaler.transform(new_data)
new_data_scaled

reg.predict(new_data_scaled)

# Effect of Removing Unsignificant Variable Rand 1,2,3
reg_simple = LinearRegression()
x_simple_matrix = x_scaled[:,0].reshape(-1,1)
reg_simple.fit(x_simple_matrix, y)

reg_simple.predict(new_data_scaled[:,0].reshape(-1,1))