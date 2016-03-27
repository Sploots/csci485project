import math

class TicTacToe:
    dimx = None
    dimy = None
    k = None
    turn = None
    board = None
    boardlist = None
    won = None

    def __init__(self, dimx=3, dimy=3, k=3):
        self.dimx = dimx
        self.dimy = dimy
        self.k = k
        self.boardclear()
        
    def boardclear(self):
        dimx = self.dimx
        dimy = self.dimy
        newboard = []
        newboardlist = []

        for i in range(dimy):
            row = []

            for j in range(dimx):
                row.append(" ")
                newboardlist.append(0)

            newboard.append(row)

        self.turn = 0
        self.board = newboard
        self.boardlist = newboardlist
        self.won = False

    def winner(self):
        board = self.board
        dimx = self.dimx
        dimy = self.dimy
        k = self.k
        foundwinner = False

        for row in range(dimy-k):
            for col in range(dimx-k):
                if board[row][col] == ' ':
                    continue

                # check for k-in-a-row horizontal
                if not foundwinner:
                    foundwinner = True
                    for i in range(1,k):
                        if board[row][col] != board[row][col+i]:
                            foundwinner = False
                            break;

                # check for k-in-a-row vertical
                if not foundwinner:
                    foundwinner = True
                    for i in range(1,k):
                        if board[row][col] != board[row+i][col]:
                            foundwinner = False
                            break;

                # check for k-in-a-row lower right diagonal
                if not foundwinner:
                    foundwinner = True
                    for i in range(1,k):
                        if board[row][col] != board[row+i][col+i]:
                            foundwinner = False
                            break;

                if foundwinner:
                    self.won = True
                    if (board[row][col] == 'X'):
                        return -1
                    elif board[row][col] == '@':
                        return 1

        for row in range(k-1,dimy):
            for col in range(dimx-k):
                if board[row][col] == ' ':
                    continue

                # check for k-in-a-row upper right diagonal
                foundwinner = True
                for i in range(1,k):
                    if board[row][col] != board[row-i][col+i]:
                        foundwinner = False
                        break

                if foundwinner:
                    self.won = True
                    if (board[row][col] == 'X'):
                        return -1
                    elif board[row][col] == '@':
                        return 1

        return 0

    def addmark(self, n):
        if self.won:
            return False

        board = self.board
        boardlist = self.boardlist
        turn = self.turn
        dimx = self.dimx
        dimy = self.dimy
        row = int(math.floor(n/dimx))
        col = int(n%dimx)

        if board[row][col] != ' ':
            return False

        if turn%2:
            board[row][col] = 'X'
            boardlist[n] = -1
        else:
            board[row][col] = '@'
            boardlist[n] = 1

        self.turn += 1
        self.board = board
        self.boardlist = boardlist

        return True

    def printboard(self):
        print("")

        board = self.board
        dimx = self.dimx
        dimy = self.dimy
        size = len(str(dimx*dimy - 1))

        for row in range(dimy):
            line = ""
            for col in range(dimx):
                i = 0
                while i < size:
                    if board[row][col] == " " and i == 0:
                        pos = row*dimx + col
                        line += repr(pos)

                        for j in range(len(str(pos))-1):
                            i += 1
                    elif i == math.floor((size)/2):
                        if board[row][col] == "@":
                            color = "\033[92m"
                        else:
                            color = "\033[91m"

                        line += color + board[row][col] + "\033[0m"
                    else:
                        line += " "

                    i += 1

                if col < dimx-1:
                    line += "|"

            print(line)

            if row < dimy-1:
                line = ""
                for i in range(size*dimx + dimx-1):
                    line += "-"

                print(line)

        print("")

Game = TicTacToe(11,15,4)
Game.addmark(1)
Game.printboard()
print(Game.winner())
Game.addmark(8)
Game.printboard()
print(Game.winner())
Game.addmark(13)
Game.printboard()
print(Game.winner())
Game.addmark(14)
Game.printboard()
print(Game.winner())
Game.addmark(23)
Game.printboard()
print(Game.winner())
Game.addmark(26)
Game.printboard()
print(Game.winner())
Game.addmark(33)
Game.printboard()
print(Game.winner())
Game.addmark(34)
Game.printboard()
print(Game.winner())
Game.addmark(3)
Game.printboard()
print(Game.winner())