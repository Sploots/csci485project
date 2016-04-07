import sys
import random
from MC import MCbot
from TicTacToe import TicTacToe

args = []

for arg in sys.argv:
   args.append(arg)

if len(args) != 5:
	print("Must have exactly 4 commandline arguments: <board rows> <board columns> <k-in-a-row to win> <Monte Carlo bot passes>")
	sys.exit()
else:
	board_rows = int(args[1])
	board_cols = int(args[2])
	k_to_win = int(args[3])
	bot_passes = int(args[4])

boardlist_size = board_rows*board_cols

clf = MCbot(bot_passes)

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