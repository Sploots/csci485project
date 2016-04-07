import sys
import random
import pickle
import numpy as np
from NN import NN
from TicTacToe import TicTacToe

# Define how many games to play
n_games = 100000

args = []

for arg in sys.argv:
   args.append(arg)

if len(args) != 5:
    print("Must have exactly 4 commandline arguments: <board rows> <board columns> <k-in-a-row to win> <NN file>")
    sys.exit()
else:
    board_rows = int(args[1])
    board_cols = int(args[2])
    k_to_win = int(args[3])
    nn_filename = args[4]

boardlist_size = board_rows*board_cols

with open(nn_filename, 'rb') as f:
    clf = pickle.load(f)
    f.close()

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

print("\nNN win rate: " + repr(Bot1['win']/float(n_games)*100) + "%")
print("NN lose rate: " + repr(Bot1['lose']/float(n_games)*100) + "%")
print("NN tie rate: " + repr(Bot1['tie']/float(n_games)*100) + "%\n")
print("Randombot win rate: " + repr(Bot2['win']/float(n_games)*100) + "%")
print("Randombot lose rate: " + repr(Bot2['lose']/float(n_games)*100) + "%")
print("Randombot tie rate: " + repr(Bot2['tie']/float(n_games)*100) + "%\n")