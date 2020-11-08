#19-02-Absenteeism(MachineLearning)

#IMPORT LIBRARIES
import pandas as pd
import numpy as np

#LOAD DATA
data_preprocessed = pd.read_csv('AbsenteeismData_Preprocessed.csv')
data_preprocessed.head()

#CREATE TARGETS
data_preprocessed['Absenteeism Time in Hours'].median()
targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 
                   data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)
targets
data_preprocessed['Excessive Absenteeism'] = targets
data_preprocessed.head()
targets.sum() / targets.shape[0] #percent of 1s
data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours','Day of the Week',
                                            'Daily Work Load Average','Distance to Work'],axis=1)
data_with_targets is data_preprocessed
data_with_targets.head()

#SELECT INPUTS FOR REGRESSION
data_with_targets.shape
data_with_targets.iloc[:, 0:14] #all rows, first 14 columns are inputs.
data_with_targets.iloc[:, :-1] #same as above, skip last column.
unscaled_inputs = data_with_targets.iloc[:, :-1]

#STANDARDIZE DATA
from sklearn.preprocessing import StandardScaler
absenteeism_scaler = StandardScaler() #standard scaler scales dummy variables also, so write custom one.

# import the libraries needed to create the Custom Scaler
# note that all of them are a part of the sklearn package
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# create the Custom Scaler class
class CustomScaler(BaseEstimator,TransformerMixin): 
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
     def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
 
    def transform(self, X, y=None, copy=None):        
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

unscaled_inputs.columns.values
columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Education']
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
scaled_inputs
scaled_inputs.shape

#SPLIT, TRAIN, TEST, SHUFFLE
from sklearn.model_selection import train_test_split
train_test_split(scaled_inputs, targets)
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size = 0.8, random_state=20) #train size is 80%
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#TRAIN MODEL
reg = LogisticRegression()
reg.fit(x_train, y_train)
reg.score(x_train, y_train)

#MANUAL ACCURACY CHECK
model_outputs = reg.predict(x_train)
model_outputs
y_train
model_outputs == y_train
np.sum(model_outputs==y_train) #number of true.
model_outputs.shape[0]
np.sum(model_outputs==y_train) / model_outputs.shape[0] #accuracy

#FIND INTERCEPTS AND COEFFICIENTS
reg.intercept_
reg.coef_
unscaled_inputs.columns.values
feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame(columns=['Feature Name'], data=feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)
summary_table
summary_table.index = summary_table.index + 1 #shift all indexes by one.
summary_table.loc[0] = ['Intercept', reg.intercept_[0]] #add intecept column.
summary_table = summary_table.sort_index()
summary_table

#INTERPRETING COEFFICIENTS
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)
summary_table #for a unit change in standardized feature odds increase by a multiple equal to the odds ratio. (1=no change)
summary_table.sort_values('Odds_ratio', ascending=False) 
#features with coefficient around 0 and odd ratio around 1 is not important.
#daily work load average is least important in features. others day of week, distance to work.

#TESTING MODEL
reg.score(x_test, y_test)
predicted_proba = reg.predict_proba(x_test) #predicted probability
predicted_proba
predicted_proba.shape
predicted_proba[:, 1]

#SAVE MODEL
import pickle

with open('model', 'wb') as file:
    pickle.dump(reg, file)

with open('scaler','wb') as file:
    pickle.dump(absenteeism_scaler, file)