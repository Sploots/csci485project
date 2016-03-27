import numpy as np
import math
import pickle
from NN import NN
from TicTacToe import TicTacToe

np.random.seed(0)
phi = 1

# Define the size of the board
board_rows = 3
board_cols = 3
boardlist_size = board_rows*board_cols

# Define win condition (# of symbols in a row)
k_to_win = 3

# Define how many games to play
n_games = 1000

# Build a model with a n-dimensional hidden layer
clf = NN(100, boardlist_size, boardlist_size)

game = TicTacToe(board_rows, board_cols, k_to_win)

for i in range(n_games):
    print("Playing game " + repr(i) + "...")

    game.boardclear()
    turn = 0;
    winner = None;

    playerA_X = [] # X symbol
    playerB_X = [] # @ symbol

    playerA_Y = [] # X symbol
    playerB_Y = [] # @ symbol

    while not game.won:
        boardlist = game.boardlist

        if turn%2:
            probs = clf.probs(-1*boardlist)
            playerA_X.append(-1*boardlist)

        else:
            probs = clf.probs(boardlist)
            playerB_X.append(boardlist)

        spot = np.argmax(probs)

        while not game.addmark(spot):
            ##print(spot)
            ##game.printboard()
            probs[0][spot] = 0
            spot = np.argmax(probs)

        Y = []

        # record the board state and chosen response
        for i in range(boardlist_size):
            if i == spot:
                Y.append(1) # indicate chosen spot
                continue

            if boardlist[i] == 0:
                Y.append(0) # indicate open spot
            else:
                Y.append(-1) # indicate unavailable spot

        if turn%2:
            playerA_Y.append(Y)
        else:
            playerB_Y.append(Y)

        winner = game.winner()
        ##print("WINNER: ")
        ##print(winner)

        if winner == 0:
            turn += 1
            continue

        elif winner == 2:
            # change all the y values for unavailable spots to 0
            for i in range(len(playerA_Y)):
                for j in range(boardlist_size):
                    if playerA_Y[i][j] == -1:
                        playerA_Y[i][j] = 0

            # change all the y values for unavailable spots to 0
            for i in range(len(playerB_Y)):
                for j in range(boardlist_size):
                    if playerB_Y[i][j] == -1:
                        playerB_Y[i][j] = 0

            break

        elif winner == -1:
            # change all the y values for unavailable spots to 0
            for i in range(len(playerA_Y)):
                for j in range(boardlist_size):
                    if playerA_Y[i][j] == -1:
                        playerA_Y[i][j] = 0

            # change the y values for the loser
            for i in range(len(playerB_Y)):
                penalty = math.exp(-phi*(len(playerB_Y)-i))
                ##print("Penalty:")
                ##print(penalty)
                penalized_score = 1 - penalty

                # count how many open spots there are
                count = 0
                for j in range(boardlist_size):
                    if playerB_Y[i][j] == 0:
                        count += 1
                    elif playerB_Y[i][j] == 1:
                        playerB_Y[i][j] -= penalized_score

                residual_prob = penalty/count

                # change all the y values for unavailable spots to 0, and open spots to the residual_prob
                for j in range(boardlist_size):
                    if playerB_Y[i][j] == 0:
                        playerB_Y[i][j] = residual_prob
                    elif playerB_Y[i][j] == -1:
                        playerB_Y[i][j] = 0

            break

        elif winner == 1:
            ##print("Winner is 1!")
            # change all the y values for unavailable spots to 0
            for i in range(len(playerB_Y)):
                for j in range(boardlist_size):
                    if playerB_Y[i][j] == -1:
                        playerB_Y[i][j] = 0

            # change the y values for the loser
            for i in range(len(playerA_Y)):
                penalty = math.exp(-phi*(len(playerA_Y)-i))
                ##print("Penalty:")
                ##print(penalty)
                penalized_score = 1 - penalty

                # count how many open spots there are
                count = 0
                for j in range(boardlist_size):
                    if playerA_Y[i][j] == 0:
                        count += 1
                    elif playerA_Y[i][j] == 1:
                        playerA_Y[i][j] -= penalized_score

                residual_prob = penalty/count

                # change all the y values for unavailable spots to 0, and open spots to the residual_prob
                for j in range(boardlist_size):
                    if playerA_Y[i][j] == 0:
                        playerA_Y[i][j] = residual_prob
                    elif playerA_Y[i][j] == -1:
                        playerA_Y[i][j] = 0

            break

    train_X = playerA_X + playerB_X
    train_Y = playerA_Y + playerB_Y

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    ##print("X train:")
    ##print(train_X)
    ##print("Y train:")
    ##print(train_Y)

    clf.train(train_X, train_Y)

with open('NN' + repr(board_rows) + repr(board_cols), 'wb') as f:
    pickle.dump(clf,f)
    f.close()

print("Done!")