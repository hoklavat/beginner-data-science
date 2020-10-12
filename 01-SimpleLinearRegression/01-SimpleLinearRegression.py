#01-SimpleLinearRegression

#IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#LOAD DATA
data = pd.read_csv('01-SimpleLinearRegression.csv')

#SHOW DATA
data

#BASIC STATISTICS
data.describe()

#REGRESSION VARIABLES
y = data['GPA']
x1 = data['SAT']

#EXPLORE DATA
plt.scatter(x1, y)
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()

#REGRESSION COMPUTATION
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
results.summary()

#REGRESSION LINE
plt.scatter(x1,y)
yhat = 0.0017 * x1 + 0.275
fig = plt.plot(x1, yhat, lw = 4, c='orange', label='regression line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()