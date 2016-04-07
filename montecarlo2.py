import random
import numpy as np
from NN import NN
from TicTacToe import TicTacToe

# Define the size of the board
board_rows = 3
board_cols = 3

# Define win condition (# of symbols in a row)
k_to_win = 3

# Define how many games to play
n_games = 100000

# Define how many hidden layer nodes for the NN to have
num_nodes = 100

boardlist_size = board_rows*board_cols
game = TicTacToe(board_rows, board_cols, k_to_win)

Bot1 = {'win':0, 'lose':0, 'tie':0}
Bot2 = {'win':0, 'lose':0, 'tie':0}

for game_num in range(0,n_games):
    print("Playing game " + repr(game_num) + "...")

    playerturn = random.randint(0,1)

    turn = 0

    while not game.won:
        boardlist = game.boardlist

        if turn%2 == playerturn:
            spot = random.randint(0, boardlist_size-1)
            while not game.addmark(spot):
                spot = random.randint(0, boardlist_size-1)
        else:
            clf = NN(num_nodes, boardlist_size, boardlist_size)

            boardlist = game.boardlist

            if turn%2:
                probs = clf.probs(-1*boardlist)
            else:
                probs = clf.probs(boardlist)

            spot = np.argmax(probs)

            while not game.addmark(spot):
                probs[0][spot] = 0
                spot = np.argmax(probs)         
    
        winner = game.winner()

        if winner == 0:
            turn += 1
            continue
        elif winner == 2:
            Bot1['tie'] += 1
            Bot2['tie'] += 1

        elif winner == -1:
            if turn%2 == playerturn:
                Bot2['win'] += 1
                Bot1['lose'] += 1
            else:
                Bot1['win'] += 1
                Bot2['lose'] += 1

        elif winner == 1:
            if turn%2 == playerturn:
                Bot2['win'] += 1
                Bot1['lose'] += 1
            else:
                Bot1['win'] += 1
                Bot2['lose'] += 1

    game.boardclear()

print("\nUntrained NN win rate: " + repr(Bot1['win']/float(n_games)*100) + "%")
print("Untrained NN lose rate: " + repr(Bot1['lose']/float(n_games)*100) + "%")
print("Untrained NN tie rate: " + repr(Bot1['tie']/float(n_games)*100) + "%\n")
print("Randombot win rate: " + repr(Bot2['win']/float(n_games)*100) + "%")
print("Randombot lose rate: " + repr(Bot2['lose']/float(n_games)*100) + "%")
print("Randombot tie rate: " + repr(Bot2['tie']/float(n_games)*100) + "%\n")