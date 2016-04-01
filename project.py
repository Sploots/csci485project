import numpy as np
import math
import pickle
import copy
from NN import NN
from TicTacToe import TicTacToe

np.random.seed(0)
phi = 0.1

mode = 2
continue_mode = False
save_every_game = False

# Define the size of the board
board_rows = 3
board_cols = 3
boardlist_size = board_rows*board_cols

# Define win condition (# of symbols in a row)
k_to_win = 3

# Define how many games to play
n_games = 10000

# Build a model with a n-dimensional hidden layer
num_passes = 3000
num_nodes = 300

game_startindex = 0

if continue_mode:
    with open('NN' + repr(board_rows) + repr(board_cols), 'rb') as f:
        clf = pickle.load(f)
        f.close()
    with open('TrainX' + repr(board_rows) + repr(board_cols), 'rb') as f:
        train_X_large = pickle.load(f)
        f.close()

    with open('TrainY' + repr(board_rows) + repr(board_cols), 'rb') as f:
        train_Y_large = pickle.load(f)
        f.close()
    
    for x in train_X_large:
        if np.sum(np.absolute(np.array(x))) == 0:
            game_startindex += 1
else:
    clf = NN(num_nodes, boardlist_size, boardlist_size)
    train_X_large = []
    train_Y_large = []

game = TicTacToe(board_rows, board_cols, k_to_win)

for game_num in range(game_startindex, n_games):
    print("Playing game " + repr(game_num) + "...")

    if mode == 2:
        clf = NN(num_nodes, boardlist_size, boardlist_size)

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
            playerB_X.append(copy.deepcopy(boardlist))

        spot = np.argmax(probs)

        while not game.addmark(spot):
            ##print(spot)
            ##game.printboard()
            probs[0][spot] = 0

            spot = np.argmax(probs)

            # prevent infinite loop
#            if np.sum(probs[0]) == 0:
#                for i in range(boardlist_size):
#                    if boardlist[i] == 0:
#                        spot = i
#                        break

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
            print("It's a tie!")
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
            print("Player A wins!")
            # change all the y values for unavailable spots to 0
            for i in range(len(playerA_Y)):
                for j in range(boardlist_size):
                    if playerA_Y[i][j] == -1:
                        playerA_Y[i][j] = 0

            # change the y values for the loser
            for i in range(len(playerB_Y)):
                penalty = math.exp(-phi*(len(playerB_Y)-i))
                ##print("Penalty for entry " + repr(i) + ": " + repr(penalty))
                ##print("Penalty:")
                ##print(penalty)

                # count how many open spots there are
                count = 1
                for j in range(boardlist_size):
                    if playerB_Y[i][j] == 0:
                        count += 1

                split_prob = 1/float(count)
                if count-1 > 0:
                    othermove_prob = split_prob*(1+penalty/(count-1))
                chosenmove_prob = split_prob*(1-penalty)

                for j in range(boardlist_size):
                    if playerB_Y[i][j] == 0:
                        playerB_Y[i][j] = othermove_prob
                    elif playerB_Y[i][j] == -1:
                        playerB_Y[i][j] = 0
                    elif playerB_Y[i][j] == 1:
                        playerB_Y[i][j] = chosenmove_prob

            break

        elif winner == 1:
            print("Player B wins!")
            # change all the y values for unavailable spots to 0
            for i in range(len(playerB_Y)):
                for j in range(boardlist_size):
                    if playerB_Y[i][j] == -1:
                        playerB_Y[i][j] = 0

            # change the y values for the loser
            for i in range(len(playerA_Y)):
                penalty = math.exp(-phi*(len(playerA_Y)-i))
                ##print("Penalty for entry " + repr(i) + ": " + repr(penalty))
                ##print("Penalty:")
                ##print(penalty)

                # count how many open spots there are
                count = 1
                for j in range(boardlist_size):
                    if playerA_Y[i][j] == 0:
                        count += 1

                split_prob = 1/float(count)
                if count-1 > 0:
                    othermove_prob = split_prob*(1+penalty/(count-1))
                chosenmove_prob = split_prob*(1-penalty)

                for j in range(boardlist_size):
                    if playerA_Y[i][j] == 0:
                        playerA_Y[i][j] = othermove_prob
                    elif playerA_Y[i][j] == -1:
                        playerA_Y[i][j] = 0
                    elif playerA_Y[i][j] == 1:
                        playerA_Y[i][j] = chosenmove_prob

            break

    print("Player A_X:")
    print(np.array(playerA_X))
    print("Player A_Y:")
    print(np.array(playerA_Y))
    print("Player B_X:")
    print(np.array(playerB_X))
    print("Player B_Y:")
    print(np.array(playerB_Y))

    game.printboard()

    print(clf.probs(boardlist))


    train_X = playerA_X + playerB_X
    train_Y = playerA_Y + playerB_Y

    train_X_large += train_X
    train_Y_large += train_Y

    if mode == 2:
        train_X = np.array(train_X_large)
        train_Y = np.array(train_Y_large)
    else:
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)

    if not clf.train(train_X, train_Y, num_passes):
        break

    if (game_num+1) % 10 == 0 or save_every_game:
        print("Saving data...")
        with open('NN' + repr(board_rows) + repr(board_cols), 'wb') as f:
            pickle.dump(clf,f)
            f.close()

        with open('TrainX' + repr(board_rows) + repr(board_cols), 'wb') as f:
            pickle.dump(train_X_large,f)
            f.close()

        with open('TrainY' + repr(board_rows) + repr(board_cols), 'wb') as f:
            pickle.dump(train_Y_large,f)
            f.close()

with open('NN' + repr(board_rows) + repr(board_cols), 'wb') as f:
    pickle.dump(clf,f)
    f.close()

with open('TrainX' + repr(board_rows) + repr(board_cols), 'wb') as f:
    pickle.dump(train_X_large,f)
    f.close()

with open('TrainY' + repr(board_rows) + repr(board_cols), 'wb') as f:
    pickle.dump(train_Y_large,f)
    f.close()

print("Done!")