import numpy as np
import math
import pickle
import copy
from NN import NN
from TicTacToe import TicTacToe

board_rows = 3
board_cols = 3

with open('NN' + repr(board_rows) + repr(board_cols), 'rb') as f:
    clf = pickle.load(f)
    f.close()

with open('TrainX' + repr(board_rows) + repr(board_cols), 'rb') as f:
    train_X_large = pickle.load(f)
    f.close()

with open('TrainY' + repr(board_rows) + repr(board_cols), 'rb') as f:
    train_Y_large = pickle.load(f)
    f.close()

print(clf.nn_hdim)

#clf.train(np.array(train_X_large), np.array(train_Y_large), num_passes=1, verbose=True)