#06-TrainTestSplit
#overfitting: training has focused on the particular training set so much, eventually it has missed the point and captures all noise.
#prevent overfitting by splitting data into training (80%) and testing (20%) data.

import numpy as np
from sklearn.model_selection import train_test_split

a = np.arange(1,101) #make a new array with values 1 to 100
a

b = np.arange(501,601) #make a new array with values 501 to 600
b

train_test_split(a) #split array into two arrays randomly with %75-%25 ratio by default


a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42) 
#test_size=0.2: corresponds to 80%-20% split ratio
#random_state=42: constant random seed, same pattern after each split.
#shuffle=False: no randomization.

a_train.shape, a_test.shape

a_train

a_test

b_train.shape, b_test.shape

b_train

b_test

