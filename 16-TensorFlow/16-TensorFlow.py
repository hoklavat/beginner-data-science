#16-TensorFlow

#IMPORT LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#RANDOM DATA GENERATION
observations = 1000 #number of observations
xs = np.random.uniform(low=-10, high=10, size=(observations, 1)) #weight-1, 100row-1column matrice
zs = np.random.uniform(low=-10, high=10, size=(observations, 1)) #weight-2
generated_inputs = np.column_stack((xs, zs))
noise = np.random.uniform(-1, 1, (observations, 1))
generated_targets = 2*xs - 3*zs + 5 + noise
np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)

#SOLVING FOR TENSORFLOW
training_data = np.load('TF_intro.npz')

input_size = 2
output_size = 1
model = tf.keras.Sequential([
                             tf.keras.layers.Dense(output_size,
                                                   kernel_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1), #initialize weights randomly
                                                   bias_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1) #initialize biases randomly
                                                  ) #output = np.dot(inputs, weights) + bias
                            ])
custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
#model.compile(optimizer='sgd', loss='mean_squared_error')
model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=2)#epoch: iteration over the full data set, verbose=0: no output

#EXTRACT WEIGHTS and BIASES
model.layers[0].get_weights()

weights = model.layers[0].get_weights()[0]
weights

biases = model.layers[0].get_weights()[1]
biases

#EXTRACT OUTPUTS (MAKE PREDICTIONS)
model.predict_on_batch(training_data['inputs'])

training_data['targets']

#PLOT DATA
plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()