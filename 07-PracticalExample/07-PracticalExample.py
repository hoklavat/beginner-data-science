#07-PracticalExample

#PREDICTING PRICE of a USED CAR
#variables: mileage, engine volume, brand, registration and body type

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()

raw_data = pd.read_csv('RealLifeExample.csv')
raw_data.head()

raw_data.describe(include='all')

#CLEAN UNNECESSARY DATA
data = raw_data.drop(['Model'], axis=1) #exclude Model column, for a row: axis=0
data.describe(include='all')

data.isnull() #show missing values

data.isnull().sum() #count of all missing values

data_no_mv = data.dropna(axis=0) #exclude rows having missing values
data_no_mv.describe(include='all')

sns.displot(data_no_mv['Price']) #plot distribution by price

q = data_no_mv['Price'].quantile(0.99) #considering only 99% of data away from outliers. new data set is closer to mean.
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')

sns.displot(data_1['Price']) #anomalities near outliers are lessened.

sns.displot(data_no_mv['Mileage']) #repeat same process for mileage, enginev, year

q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]

sns.displot(data_2['Mileage'])

sns.displot(data_no_mv['EngineV'])

data_3 = data_2[data_2['EngineV']<6.5]

sns.displot(data_3['EngineV'])

sns.displot(data_no_mv['Year'])

q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]

sns.displot(data_4['Year'])

data_cleaned = data_4.reset_index(drop=True) #completely forget previous indexes, refresh indexes

data_cleaned.describe(include='all')

#CHECK THE OLS ASSUMPTIONS
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3)) #scatter plot of regressors
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')

sns.displot(data_cleaned['Price'])

log_price = np.log(data_cleaned['Price']) #natural logarithm of data. linearization.
data_cleaned['log_price'] = log_price
data_cleaned

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3)) #scatter plot of regressors
ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')
plt.show()

data_cleaned = data_cleaned.drop(['Price'], axis=1)

#CHECK MULTICOLLINEARITY
data_cleaned.columns.values

from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['features'] = variables.columns

vif #1: no multicollinearity, 1-5: perfectly okay, >5: unacceptable

data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1) # year has high multicollinearity, so drop it.

#CREATE DUMMY VARIABLES
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True) #N categorical variables = N-1 dummy variable + 1 benchmark

data_with_dummies.head()

#REARRANGE

data_with_dummies.columns.values

cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']

data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()

#LINEAR REGRESSION
targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'], axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)

inputs_scaled = scaler.transform(inputs)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size = 0.2, random_state = 365)

reg = LinearRegression()
reg.fit(x_train, y_train)

y_hat = reg.predict(x_train)

plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat)', size=18)
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()

sns.distplot(y_train - y_hat)
plt.title('Residuals PDF', size=18) #normality, zero mean, homoscedasticity

reg.score(x_train, y_train) #our model is explaining 75 percent of the variability of the data

reg.intercept_

reg.coef_ #weights

reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary

data_cleaned['Brand'].unique() #brands. audi is benchmark, because it isn't taken as dummy. positive weight higher than benchmark or vice-versa

#TESTING
y_hat_test = reg.predict(x_test)

plt.scatter(y_test, y_hat_test, alpha=0.2) #alpha specifies the opacity, more saturated color higher concentration.
plt.xlabel('Targets (y_test)', size=18)
plt.ylabel('Predictions (y_hat_test)', size=18)
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()

df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction']) #dataframe performance, convert log to original with exp.
df_pf.head()

df_pf['Target'] = np.exp(y_test)
df_pf

y_test = y_test.reset_index(drop=True)
y_test.head()

df_pf['Target'] = np.exp(y_test)
df_pf

df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf

df_pf.describe()

pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by=['Difference%'])