#17-ImageRecognition
#mnist data set contains 70000 handwritten numbers in 28x28 pixels.

#IMPORT PACKAGES
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

#DATA PREPROCESSING
mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True) #load mnist dataset into 2-tuple structure [input, target] with info

mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test'] #train and test data sets
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples #number of validation samples. take 10% of train data set as validation set
num_validation_samples = tf.cast(num_validation_samples, tf.int64) #convert float to integer
num_test_samples = mnist_info.splits['test'].num_examples #number of test samples
num_test_samples = tf.cast(num_test_samples, tf.int64)

def scale(image, label): #scale pixel value from (0-255) to (0-1) by dividing to 255
    image = tf.cast(image, tf.float32)
    image /= 255. #dot:floating-point
    return image, label

scaled_train_and_validation_data = mnist_train.map(scale) #scale train data set
test_data = mnist_test.map(scale) #scale test data set

#shuffling: keeping the same information in different order to prevent consecutively same numbers
BUFFER_SIZE = 10000 #prevents memory leak for large data sets
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
validation_data = shuffled_train_and_validation_data.take(num_validation_samples) #take 10% of train data
train_data = shuffled_train_and_validation_data.skip(num_validation_samples) #skip validation data and take rest 90%

BATCH_SIZE = 100 #split data into batches
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

validation_inputs, validation_targets = next(iter(validation_data))

#OUTLINE THE MODEL
input_size = 784
output_size = 10
hidden_layer_size = 50

model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape=(28, 28, 1)), #image size
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(output_size, activation='softmax')
                            ])

#OPTIMIZER and LOSS FUNCTION
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#TRAINING
NUM_EPOCHS = 5
model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), validation_steps=10, verbose=2)

#TESTING
test_loss, test_accuracy = model.evaluate(test_data)
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.)) #test accuracy and validation accuracy should approximately be equal.

