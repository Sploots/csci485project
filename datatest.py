import numpy as np
import math
import copy
from NN import NN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_classification
from sklearn.metrics import accuracy_score
from sklearn import cross_validation

np.random.seed(0)
input_dim = 20
output_dim = 5

#X, y = make_blobs(n_samples=100, centers=output_dim, n_features=input_dim, random_state=0)
X, y = make_classification(n_samples=2000, n_classes=output_dim, n_features=input_dim, n_informative=input_dim, n_redundant=0, random_state=0)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(X, y, random_state = 0, test_size = 0.2)

for i in range(len(features_train)):
	printlist = features_train[i].tolist()
	printlist.append(labels_train[i])
	print(printlist)

new_labels_train = []

for sample in labels_train:
	new_sample = []
	for i in range(output_dim):
		if sample == i:
			new_sample.append(1)
		else:
			new_sample.append(0)
	new_labels_train.append(new_sample)

dimensions = [3, 5, 10, 20, 40, 80]

for i in dimensions:
	clf = NN(i, input_dim, output_dim)
	print("Training a NN with " + repr(i) + " size hidden layer...")
	clf.train(np.array(features_train), np.array(new_labels_train), print_loss=True, num_passes=5000)
	print("Predicting...")
	pred = clf.predict(features_test)
	print("Accuracy Score:")
	print(accuracy_score(pred, labels_test))