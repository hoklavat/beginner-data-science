#04-SimpleLinearRegression(Sklearn)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

data = pd.read_csv("SimpleLinearRegression.csv")
data.head()

x = data['SAT'] #machine learning jargon: input, independent variable >> feature
y = data['GPA'] #machine learning jargon: output, dependent variable >> target

x.shape

y.shape

x_matrix = x.values.reshape(-1,1) #convert one-dimensional data into two-dimensional matrix for fit method
x_matrix.shape

reg = LinearRegression()
reg.fit(x_matrix,y) #calculate linear regression for further investigation