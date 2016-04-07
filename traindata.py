import sys
import pickle
import random
import copy
import math
import numpy as np
from NN import NN
from MC import MCbot

np.random.seed(0)

args = []

for arg in sys.argv:
   args.append(arg)

if len(args) != 5:
	print("Must have exactly 4 commandline arguments: <features filename> <labels filename> <# hidden layer nodes> <# training passes>")
	sys.exit()
else:
	features = args[1]
	labels = args[2]
	num_nodes = int(args[3])
	num_passes = int(args[4])

n_games = 0

with open(features, 'rb') as f:
    features_train = pickle.load(f)
    f.close()

with open(labels, 'rb') as f:
    labels_train = pickle.load(f)
    f.close()

features_dim = len(features_train[0])
labels_dim = len(labels_train[0])

for x in features_train:
    if np.sum(np.absolute(np.array(x))) == 0:
        n_games += 1

print("Training on dataset of " + repr(n_games) + " games...")

clf = NN(num_nodes, features_dim, labels_dim)

features_train = np.array(features_train)
labels_train = np.array(labels_train)

clf.train(features_train, labels_train, num_passes, print_loss=True)

with open('models/NN' + repr(features_dim) + repr(labels_dim), 'wb') as f:
    pickle.dump(clf,f)
    f.close()