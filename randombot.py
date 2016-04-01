import pickle
import random
import copy
import math
import numpy as np
from NN import NN
from MC import MCbot
from TicTacToe import TicTacToe

np.random.seed(0)

mode = 2 # 1 for both randombots, 2 for random vs NN, 3 randombot vs untrained NN, 4 randombot vs MCbot

# Define the size of the board
board_rows = 3
board_cols = 3
boardlist_size = board_rows*board_cols

# Define win condition (# of symbols in a row)
k_to_win = 3

# Define how many games to play
n_games = 100000

# Define whether to generate training data
phi = 0.1
generate_data = False
train_X_large = []
train_Y_large = []

num_nodes = 300

if mode == 2:
    with open('NN' + repr(board_rows) + repr(board_cols), 'rb') as f:
        clf = pickle.load(f)
        f.close()
elif mode == 4:
    clf = MCbot(1000)
    MCmemory = {}

game = TicTacToe(board_rows, board_cols, k_to_win)

NNbot = {'win':0, 'lose':0, 'tie':0}
Randombot = {'win':0, 'lose':0, 'tie':0}

for game_num in range(0,n_games):
    print("Playing game " + repr(game_num) + "...")

    playerA_X = [] # X symbol
    playerB_X = [] # @ symbol

    playerA_Y = [] # X symbol
    playerB_Y = [] # @ symbol

    playerturn = random.randint(0,1)

    turn = 0

    while not game.won:
        boardlist = game.boardlist

        if generate_data:
            if turn%2:
                playerA_X.append(-1*boardlist)

            else:
                playerB_X.append(copy.deepcopy(boardlist))

        if turn%2 == playerturn:
            spot = random.randint(0, boardlist_size-1)
            while not game.addmark(spot):
                spot = random.randint(0, boardlist_size-1)
        else:
            if mode == 1:
                spot = random.randint(0, boardlist_size-1)
                while not game.addmark(spot):
                    spot = random.randint(0, boardlist_size-1)
            elif mode == 4:
                boardlist = game.boardlist

                try:
                    spot = MCmemory[repr(boardlist)]
                except KeyError:
                    keyval = repr(boardlist)
                    spot = clf.predict(game)
                    MCmemory[keyval] = spot

                game.addmark(spot)
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

        if generate_data:
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

        if winner == 0:
            turn += 1
            continue
        elif winner == 2:
            NNbot['tie'] += 1
            Randombot['tie'] += 1

            if generate_data:
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

        elif winner == -1:
            if turn%2 == playerturn:
                Randombot['win'] += 1
                NNbot['lose'] += 1
            else:
                NNbot['win'] += 1
                Randombot['lose'] += 1

            if generate_data:
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

        elif winner == 1:
            if turn%2 == playerturn:
                Randombot['win'] += 1
                NNbot['lose'] += 1
            else:
                NNbot['win'] += 1
                Randombot['lose'] += 1

            if generate_data:
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

    if generate_data:
        game.printboard()
        print("Player A_X:")
        print(np.array(playerA_X))
        print("Player A_Y:")
        print(np.array(playerA_Y))
        print("Player B_X:")
        print(np.array(playerB_X))
        print("Player B_Y:")
        print(np.array(playerB_Y))

    game.boardclear()

    train_X = playerA_X + playerB_X
    train_Y = playerA_Y + playerB_Y

    train_X_large += train_X
    train_Y_large += train_Y

    if (game_num+1) % 10 == 0 and generate_data:
        print("Saving data...")

        with open('MCTrainX' + repr(board_rows) + repr(board_cols), 'wb') as f:
            pickle.dump(train_X_large,f)
            f.close()

        with open('MCTrainY' + repr(board_rows) + repr(board_cols), 'wb') as f:
            pickle.dump(train_Y_large,f)
            f.close()

print("NNbot win rate: " + repr(NNbot['win']/float(n_games)*100) + "%")
print("NNbot lose rate: " + repr(NNbot['lose']/float(n_games)*100) + "%")
print("NNbot tie rate: " + repr(NNbot['tie']/float(n_games)*100) + "%")
print("Randombot win rate: " + repr(Randombot['win']/float(n_games)*100) + "%")
print("Randombot lose rate: " + repr(Randombot['lose']/float(n_games)*100) + "%")
print("Randombot tie rate: " + repr(Randombot['tie']/float(n_games)*100) + "%")