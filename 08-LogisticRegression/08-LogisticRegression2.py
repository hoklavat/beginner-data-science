#08-LogisticRegression2

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#Apply commented fix to a statsmodels library if error occurs at reg_results.summary()
#from scipy import stats
#stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

raw_data = pd.read_csv('Admittance.csv')

data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})
data

y = data['Admitted']
x1 = data['SAT']

x = sm.add_constant(x1)
reg_log = sm.Logit(y, x) #logistic regression function
results_log = reg_log.fit()

results_log.summary()

x0 = np.ones(168) #array with 168-elements filled with 1
reg_log = sm.Logit(y, x0)
results_log = reg_log.fit()
results_log.summary() #no change in LL-Null