import pickle
import random
import copy
import math
import numpy as np
from NN import NN
from MC import MCbot

np.random.seed(0)

# Define the size of the board
board_rows = 3
board_cols = 3
boardlist_size = board_rows*board_cols

num_nodes = 400
num_passes = 6000

game_startindex = 0

with open('MCTrainX' + repr(board_rows) + repr(board_cols), 'rb') as f:
    train_X_large = pickle.load(f)
    f.close()

with open('MCTrainY' + repr(board_rows) + repr(board_cols), 'rb') as f:
    train_Y_large = pickle.load(f)
    f.close()

for x in train_X_large:
    if np.sum(np.absolute(np.array(x))) == 0:
        game_startindex += 1

print("Training on dataset of " + repr(game_startindex) + " games...")

clf = NN(num_nodes, boardlist_size, boardlist_size)

train_X = np.array(train_X_large)
train_Y = np.array(train_Y_large)

clf.train(train_X, train_Y, num_passes, print_loss=True)

with open('NN' + repr(board_rows) + repr(board_cols), 'wb') as f:
    pickle.dump(clf,f)
    f.close()