import sys
import pickle
import random
import numpy as np
from NN import NN
from TicTacToe import TicTacToe

args = []

for arg in sys.argv:
   args.append(arg)

if len(args) != 5:
	print("Must have exactly 3 commandline arguments: <board rows> <board columns> <k-in-a-row to win> <NN model filename>")
	sys.exit()
else:
	board_rows = int(args[1])
	board_cols = int(args[2])
	k_to_win = int(args[3])
	nn_model = args[4]

boardlist_size = board_rows*board_cols

with open(nn_model, 'rb') as f:
	clf = pickle.load(f)
	f.close()

game = TicTacToe(board_rows, board_cols, k_to_win)

while True:
	print("")
	print("Starting new game!")

	playerturn = random.randint(0,1)

	turn = 0

	game.printboard()

	while not game.won:
		if turn%2 == playerturn:
			print("Player taking turn...")
			val = int(raw_input("Please select a cell number: "))

			while val > boardlist_size-1 or val < 0 or not game.addmark(val):
				val = int(raw_input("Invalid cell number. Please select a cell number: "))
		else:
			print("AI taking turn...")

			boardlist = game.boardlist

			if turn%2:
				probs = clf.probs(-1*boardlist)
				print(-1*boardlist)
			else:
				probs = clf.probs(boardlist)

			spot = np.argmax(probs)

			while not game.addmark(spot):
				probs[0][spot] = 0
				spot = np.argmax(probs)			

		game.printboard()

		winner = game.winner()

		if winner == 2:
			print("It's a tie!")
		elif winner == 1:
			if turn%2 == playerturn:
				print("You win!")
			else:
				print("You lose!")
		elif winner == -1:
			if turn%2 == playerturn:
				print("You win!")
			else:
				print("You lose!")

		turn += 1

	game.boardclear()