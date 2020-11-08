#18-CustomerInterest(Preprocessing)
#Will old customers buy again?

#EXTRACT DATA
import numpy as np
from sklearn import preprocessing

raw_csv_data = np.loadtxt('AudiobooksData.csv', delimiter=',')
unscaled_inputs_all = raw_csv_data[:, 1:-1] #inputs columns
targets_all = raw_csv_data[:, -1] #targets column (last one)

#BALANCE THE DATASET
num_one_targets = int(np.sum(targets_all)) #number of 1s
zero_targets_counter = 0
indices_to_remove = []

#number of 1s and 0s should be balanced
for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1;
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i) 

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

#STANDARDIZE INPUTS
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors) #standardize all inputs

#SHUFFLE DATA
shuffled_indices = np.arange(scaled_inputs.shape[0]) #evenly spaced values within a given interval
np.random.shuffle(shuffled_indices) #randomize indices
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

#SPLIT TRAIN, VALIDATION, TEST
samples_count = shuffled_inputs.shape[0]
train_samples_count = int(0.8*samples_count)
validation_samples_count = int(0.1*samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

#SAVE DATASETS
np.savez('AudioBooksData_Train', inputs=train_inputs, targets=train_targets)
np.savez('AudioBooksData_Validation', inputs=validation_inputs, targets=validation_targets)
np.savez('AudioBooksData_Test', inputs=test_inputs, targets=test_targets)