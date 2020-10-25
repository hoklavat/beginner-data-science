#15-NeuralNetwork

import numpy as np #mathematical operations
import matplotlib.pyplot as plt #graphing
from mpl_toolkits.mplot3d import Axes3D as ax #3D graphing

#GENERATE RANDOM INPUT DATA
observations = 1000
xs = np.random.uniform(low=-10, high=10, size=(observations, 1)) #1000 to 1 matrice containing random numbers between -10/10
zs = np.random.uniform(-10, 10, (observations, 1))
inputs = np.column_stack((xs,zs))
print(inputs.shape)

#CREATE TARGETS
noise = np.random.uniform(-1, 1, (observations, 1))
targets = 2*xs - 3*zs + 5 + noise
print(targets.shape)

#PLOT TRAINING DATA
targets = targets.reshape(observations,)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, zs, targets)
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('Targets')
ax.view_init(azim=100)
plt.show()
targets = targets.reshape(observations,1)

#INITIALIZE VARIABLES
init_range = 0.1
weights = np.random.uniform(-init_range, init_range, size=(2,1)) #random weights
biases = np.random.uniform(-init_range, init_range, size=1) #random biases
print(weights)
print(biases)

#SET LEARNING RATE
learning_rate = 0.01

#TRAIN MODEL
for i in range(1000):
    outputs = np.dot(inputs, weights) + biases
    deltas = outputs - targets #loss
    loss = np.sum(deltas ** 2) / 2 / observations #L2-Norm loss function
    print(loss)
    deltas_scaled = deltas / observations
    weights = weights - learning_rate * np.dot(inputs.T, deltas_scaled)
    biases = biases -  learning_rate * np.sum(deltas_scaled)

print(weights, biases) # targets = f(x,z) = 2*x - 3*z + 5 + noise

plt.plot(outputs, targets)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()
