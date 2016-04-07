import numpy as np
import math
import copy
from NN import NN
from sklearn.datasets.samples_generator import make_classification
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

np.random.seed(0)

# Specify test values
input_dim = 20 # dimensions of input
n_redundant = 0 # number of dimensions of input which are redundant
output_dim = 5 # dimensions of output
num_passes = 3000 # number of training passes for the NN
n_samples = 2000 # number of samples
test_fraction = 0.2 # fraction of generated data to set aside as test data
hidden_layer = [3, 5, 10, 20, 40, 80] # number of hidden layer sizes to test

X, y = make_classification(n_samples=n_samples, n_classes=output_dim, n_features=input_dim, n_informative=input_dim-n_redundant, n_redundant=n_redundant, random_state=0)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(X, y, random_state = 0, test_size = test_fraction)

new_labels_train = []

for sample in labels_train:
	new_sample = []
	for i in range(output_dim):
		if sample == i:
			new_sample.append(1)
		else:
			new_sample.append(0)
	new_labels_train.append(new_sample)

for i in hidden_layer:
	clf = NN(i, input_dim, output_dim)
	print("Training a NN with " + repr(i) + " size hidden layer...")
	clf.train(np.array(features_train), np.array(new_labels_train), print_loss=True, num_passes=num_passes)
	print("\nPredicting...")
	pred = clf.predict(features_test)
	print("\nAccuracy Score:")
	print(accuracy_score(pred, labels_test))
	print("\n")