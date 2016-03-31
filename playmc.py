import random
import numpy as np
from MC import MCbot
from TicTacToe import TicTacToe

np.random.seed(0)

# Define the size of the board
board_rows = 3
board_cols = 3
boardlist_size = board_rows*board_cols

# Define win condition (# of symbols in a row)
k_to_win = 3

clf = MCbot(1000)

game = TicTacToe(board_rows, board_cols, k_to_win)

playerturn = 0

while True:
	print("")
	print("Starting new game!")

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

			spot = clf.predict(game, verbose=True)

			game.addmark(spot)

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

	playerturn = random.randint(0,1)

	game.boardclear()
