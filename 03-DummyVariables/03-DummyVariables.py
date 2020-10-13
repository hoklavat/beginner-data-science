#03-DummyVariables

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

raw_data = pd.read_csv('03-DummyVariables.csv')
raw_data
data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'YES':1, 'NO':0}) #categorical to numerical. YES:1, NO:0
data
data.describe()

y = data['GPA']
x1 = data[['SAT', 'Attendance']]
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
results.summary()

plt.scatter(data['SAT'], y, c = data['Attendance'], cmap = 'RdYlGn_r') #attended:red, non-attended:green
yhat_yes = 0.8665 + 0.0014 * data['SAT'] #attended equation
yhat_no = 0.6439 + 0.0014 * data['SAT'] #non-attended equation
yhat = 0.0017*data['SAT'] + 0.275 #both attended and non-attended equation
fig = plt.plot(data['SAT'], yhat_yes, lw = 2, c='#a50026', label='regression line 1')
fig = plt.plot(data['SAT'], yhat_no, lw = 2, c='#006837', label='regression line 2')
fig = plt.plot(data['SAT'], yhat, lw = 3, c='#4c72b0', label='regression line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()


#PREDICTION BASED ON REGRESSION
x
new_data = pd.DataFrame({'const':1, 'SAT': [1700, 1670], 'Attendance':[0, 1]}) #filtered data
new_data = new_data[['const', 'SAT', 'Attendance']] #arrange columns
new_data
new_data.rename(index = {0:'Bob', 1:'Alice'}) #change row number with names
predictions = results.predict(new_data) #prediction function
predictions

predictionsdf = pd.DataFrame({'Predictions': predictions}) #add predictions column to the end of new_data
joined = new_data.join(predictionsdf)
joined.rename(index = {0:'Bob', 1:'Alice'})