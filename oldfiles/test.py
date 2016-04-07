import numpy as np
import TicTacToe
import pickle
from NN import NN
from TicTacToe import TicTacToe
np.random.seed(0)

# Define the size of the board
board_rows = 3
board_cols = 3
boardlist_size = board_rows*board_cols

with open('TrainX' + repr(board_rows) + repr(board_cols), 'rb') as f:
    train_X_large = pickle.load(f)
    f.close()

with open('TrainY' + repr(board_rows) + repr(board_cols), 'rb') as f:
    train_Y_large = pickle.load(f)
    f.close()

with open('Train2X' + repr(board_rows) + repr(board_cols), 'rb') as f:
    train2_X_large = pickle.load(f)
    f.close()

with open('Train2Y' + repr(board_rows) + repr(board_cols), 'rb') as f:
    train2_Y_large = pickle.load(f)
    f.close()

for i in range(len(train_X_large)):
	print "Testing item " + repr(i)
	if repr(train_X_large[i])!= repr(train2_X_large[i]):
		print("Difference in X item " + (i))

	if repr(train_Y_large[i]) != repr(train2_Y_large[i]):
			print("Difference in Y item " + (i))

'''
clf = NN(10,9,9)

# Generate a dataset
X1 = np.array([[1,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,1]])
y1 = np.array([[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0,0]])

clf.train(X1, y1, print_loss=True)
print(clf.predict(X1))
print(clf.probs(X1))


X2 = np.array([[1,0,0,0,0,0,0,2,0],[1,0,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,1,0]])
y2 = np.array([[0,0,0,0,0,0,0.2,0.8,0],[0,0,0,0,0,0,0.7,0.3,0],[0,0,0,0,0,0,0.7,0.3,0]])

clf.train(X2, y2, print_loss=True)
print(clf.predict(X1))
print(clf.probs(X1))
print(clf.predict(X2))

print("Done!")

Game = TicTacToe(3,3,3)
Game.addmark(1)
Game.printboard()
print(Game.winner())
Game.addmark(2)
Game.printboard()
print(Game.winner())
Game.addmark(4)
Game.printboard()
print(Game.winner())
Game.addmark(5)
Game.printboard()
print(Game.winner())
Game.addmark(7)
Game.printboard()
print(Game.winner())
Game.addmark(8)
Game.printboard()
print(Game.winner())
Game.addmark(7)
Game.printboard()
print(Game.winner())
Game.addmark(8)
Game.printboard()
print(Game.winner())
Game.addmark(0)
Game.printboard()
print(Game.winner())
'''