import pickle
import random
import numpy as np
from NN import NN
from TicTacToe import TicTacToe

np.random.seed(0)

mode = 3 # 1 for both randombots, 2 for random vs NN, 3 vs untrained NN

# Define the size of the board
board_rows = 3
board_cols = 3
boardlist_size = board_rows*board_cols

# Define win condition (# of symbols in a row)
k_to_win = 3

# Define how many games to play
n_games = 100000

num_nodes = 300

if mode == 2:
    with open('NN' + repr(board_rows) + repr(board_cols), 'rb') as f:
        clf = pickle.load(f)
        f.close()

game = TicTacToe(board_rows, board_cols, k_to_win)

NNbot = {'win':0, 'lose':0, 'tie':0}
Randombot = {'win':0, 'lose':0, 'tie':0}

for i in range(n_games):
    print("Playing game " + repr(i) + "...")

    playerturn = random.randint(0,1)

    turn = 0

    while not game.won:
        if turn%2 == playerturn:
            val = random.randint(0, boardlist_size-1)
            while not game.addmark(val):
                val = random.randint(0, boardlist_size-1)
        else:
            if mode == 1:
                val = random.randint(0, boardlist_size-1)
                while not game.addmark(val):
                    val = random.randint(0, boardlist_size-1)
            else:
                if mode == 3:
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

        if winner == 2:
            NNbot['tie'] += 1
            Randombot['tie'] += 1
        elif winner == 1:
            if turn%2 == playerturn:
                Randombot['win'] += 1
                NNbot['lose'] += 1
            else:
                NNbot['win'] += 1
                Randombot['lose'] += 1
        elif winner == -1:
            if turn%2 == playerturn:
                Randombot['win'] += 1
                NNbot['lose'] += 1
            else:
                NNbot['win'] += 1
                Randombot['lose'] += 1

        turn += 1

    game.boardclear()

print("NNbot win rate: " + repr(NNbot['win']/float(n_games)*100) + "%")
print("NNbot lose rate: " + repr(NNbot['lose']/float(n_games)*100) + "%")
print("NNbot tie rate: " + repr(NNbot['tie']/float(n_games)*100) + "%")
print("Randombot win rate: " + repr(Randombot['win']/float(n_games)*100) + "%")
print("Randombot lose rate: " + repr(Randombot['lose']/float(n_games)*100) + "%")
print("Randombot tie rate: " + repr(Randombot['tie']/float(n_games)*100) + "%")