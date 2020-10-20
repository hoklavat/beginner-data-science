#13-ScalarVectorMatrice

import numpy as np

#SCALAR
s = 5
s

#VECTOR
v = np.array([5, -2, 4]) #column vector
v

v = v.reshape(3, 1) #row vector
v

#MATRICE
m = np.array([[5, 12, 6], [-3, 0, 14]])
m

#DATA TYPES
type(s) #integer
type(v) #1-dimensional array
type(m) #2-dimensional array

s_array = np.array(5) #1-dimensional array with 5 elements
s_array
type(s_array)

#DATA SHAPES
m.shape

v.shape
v.reshape(1, 3) #reshape without changing data
v.reshape(3, 1)

s.shape #error, not an array
s_array.shape

#MATRICE OPERATIONS
m1 = np.array([[5, 12, 6], [-3, 0, 14]])
m1

m2 = np.array([[9, 8, 7], [1, 3, -5]])
m2

m1 + m2 #addition, same dimensions

m1 - m2 #subtraction

m1 + 1 #scalar addition

m1.T #transposing matrice

v.T #transposing vector
v.reshape(1, 3).T #convert vector to valid matrice

np.array([5]).T #scalar transposes to self

v1 = np.array([2, 8, -4])
v2 = np.array([1, -7, 3])

np.dot(v1, v2) #dot product of two vectors

np.dot(5, 6) #dot product of scalars

5 * v1 # scalar * vector

m3 = np.array([[5, 12, 6], [-3, 0, 14]])
m4 = np.array([[2, -1], [8, 0], [3, 0]])

np.dot(m3, m4) #matrice * matrice