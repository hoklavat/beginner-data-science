#05-MultipleLinearRegression(Sklearn)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

data = pd.read_csv("MultipleLinearRegression.csv")
data.head()

data.describe() #machine learning jargon: sample >> observations

x = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']

reg = LinearRegression()
reg.fit(x,y)

reg.coef_ #coefficients for linear regression equation

reg.intercept_ #intercept for linear regression equation

reg.score(x,y) #R-Squared

x.shape

r2 = reg.score(x,y)
n = x.shape[0]
p = x.shape[1]
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1) #adjusted R-square formula
adjusted_r2


# Feature Selection
# select only worthy variables by their p-values. 
# variable with p-value having 0 in the first three decimal places is eligible.

from sklearn.feature_selection import f_regression

f_regression(x,y) #F-statistics and corresponding p-values

p_values = f_regression(x,y)[1] #only fetch p-values
p_values

p_values.round(3) #first three decimal places are enough to interpret and they must be 0.

reg_summary = pd.DataFrame(data=x.columns.values, columns=['Features']) #column names
reg_summary

reg_summary['Coefficients'] = reg.coef_ #add new column representing coefficients
reg_summary['p-Values'] = p_values.round(3) #add new column representing p-values
reg_summary #rand 1,2,3 should be removed. because first three decimal places are not 0.