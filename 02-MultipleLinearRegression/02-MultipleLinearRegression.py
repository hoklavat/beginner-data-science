#02-MultipleLinearRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()

data = pd.read_csv('02-Multiple Linear Regression.csv')
data
data.describe()

y = data['GPA']
x1 = data[['SAT', 'Rand 1,2,3']]
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
results.summary()