import sys
import random
import copy
import math
import pickle
import numpy as np
from MC import MCbot
from TicTacToe import TicTacToe

# define training penalty constant
phi = 0.1

args = []

for arg in sys.argv:
   args.append(arg)

if len(args) != 6:
    print("Must have exactly 5 commandline arguments: <board rows> <board columns> <k-in-a-row to win> <Monte Carlo bot passes> <# games>")
    sys.exit()
else:
    board_rows = int(args[1])
    board_cols = int(args[2])
    k_to_win = int(args[3])
    bot_passes = int(args[4])
    n_games = int(args[5])

boardlist_size = board_rows*board_cols
game = TicTacToe(board_rows, board_cols, k_to_win)
clf = MCbot(bot_passes)
MCmemory = {}

Bot1 = {'win':0, 'lose':0, 'tie':0}
Bot2 = {'win':0, 'lose':0, 'tie':0}

train_X_large = []
train_Y_large = []

for game_num in range(0,n_games):
    playerA_X = [] # X symbol
    playerB_X = [] # @ symbol

    playerA_Y = [] # X symbol
    playerB_Y = [] # @ symbol

    print("Playing game " + repr(game_num) + "...")

    playerturn = random.randint(0,1)

    turn = 0

    while not game.won:
        boardlist = game.boardlist

        if turn%2:
            playerA_X.append(-1*boardlist)

        else:
            playerB_X.append(copy.deepcopy(boardlist))

        if turn%2 == playerturn:
            spot = random.randint(0, boardlist_size-1)
            while not game.addmark(spot):
                spot = random.randint(0, boardlist_size-1)
        else:
            boardlist = game.boardlist

            try:
                spot = MCmemory[repr(boardlist)]
            except KeyError:
                keyval = repr(boardlist)
                spot = clf.predict(game)
                MCmemory[keyval] = spot

            game.addmark(spot)

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
            Bot1['tie'] += 1
            Bot2['tie'] += 1

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
                Bot2['win'] += 1
                Bot1['lose'] += 1
            else:
                Bot1['win'] += 1
                Bot2['lose'] += 1

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
                Bot2['win'] += 1
                Bot1['lose'] += 1
            else:
                Bot1['win'] += 1
                Bot2['lose'] += 1

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

    game.printboard()
    print("Player A_X:")
    print(np.array(playerA_X))
    print("\n")
    print("Player A_Y:")
    print(np.array(playerA_Y))
    print("\n")
    print("Player B_X:")
    print(np.array(playerB_X))
    print("\n")
    print("Player B_Y:")
    print(np.array(playerB_Y))
    print("\n")

    game.boardclear()

    if winner == 2:
        train_X = playerA_X + playerB_X
        train_Y = playerA_Y + playerB_Y
    elif winner == -1:
        train_X = playerA_X
        train_Y = playerA_Y
    elif winner == 1:
        train_X = playerB_X
        train_Y = playerB_Y

    train_X_large += train_X
    train_Y_large += train_Y

    if (game_num+1) % 10 == 0:
        print("Saving data...")

        with open('data/features', 'wb') as f:
            pickle.dump(train_X_large,f)
            f.close()

        with open('data/labels', 'wb') as f:
            pickle.dump(train_Y_large,f)
            f.close()

print("\nMC Bot win rate: " + repr(Bot1['win']/float(n_games)*100) + "%")
print("MC Bot lose rate: " + repr(Bot1['lose']/float(n_games)*100) + "%")
print("MC Bot tie rate: " + repr(Bot1['tie']/float(n_games)*100) + "%\n")
print("Randombot win rate: " + repr(Bot2['win']/float(n_games)*100) + "%")
print("Randombot lose rate: " + repr(Bot2['lose']/float(n_games)*100) + "%")
print("Randombot tie rate: " + repr(Bot2['tie']/float(n_games)*100) + "%\n")