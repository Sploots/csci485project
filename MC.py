import random
import math
import copy
import numpy as np
from TicTacToe import TicTacToe

class MCbot:
    n_games = 0

    def __init__(self, n_games=10000):
        self.n_games = n_games

    def predict(self, gameobj, verbose=False):
        objboardlist = gameobj.boardlist
        objturn = gameobj.turn
        boardlist_size = gameobj.dimx*gameobj.dimy
        n_games = self.n_games

        scores = []

        for i in range(boardlist_size):
            scores.append(float(0))

        openspots = 0

        for spot in range(boardlist_size):
            if objboardlist[spot] == 0:
                openspots += 1

        for spot in range(boardlist_size):
            if verbose:
                print("Evaluating spot " + repr(spot) + " of " + repr(boardlist_size-1) + "...")

            if objboardlist[spot] != 0:
                continue

            if openspots == 1:
                return spot

            for i in range(n_games):
                game = copy.deepcopy(gameobj)

                game.addmark(spot)

                turn = game.turn

                winner = game.winner()

                while not game.won:
                    val = random.randint(0, boardlist_size-1)

                    while not game.addmark(val):
                        val = random.randint(0, boardlist_size-1)

                    winner = game.winner()

                    turn += 1

                if winner == 2:
                    scores[spot] += 1
                elif winner == 1:
                    if not objturn%2:
                        scores[spot] += 2
                    #else:
                        #scores[spot] -= math.exp(-game.turn)
                elif winner == -1:
                    if objturn%2:
                        scores[spot] += 2
                    #else:
                        #scores[spot] -= math.exp(-game.turn)

        ##for i in range(boardlist_size):
            ##if objboardlist[i] != 0:
                ##scores[i] = -n_games

        if verbose:
            print(np.array(scores))

        return np.argmax(np.array(scores))