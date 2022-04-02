from enum import Enum
from os import stat


class state(Enum):
    NONE = -2
    X = -1
    TIE = 0
    O = 1


class base:
    def __init__(self):
        self.initBoard()

    def initBoard(self):
        self.board = [[state.NONE, state.NONE, state.NONE],
                      [state.NONE, state.NONE, state.NONE],
                      [state.NONE, state.NONE, state.NONE]]

    def isValid(self, x, y):
        return False if x < 0 or x > 2 or y < 0 or y > 2 \
                        or self.board[x][y] != state.NONE \
                        else True

    def status(self):
        for i in range(3):
            if self.board[0][i] != state.NONE and \
               self.board[0][i] == self.board[1][i] and\
               self.board[1][i] == self.board[2][i]:
                return self.board[1][i]
            if self.board[i][0] != state.NONE and \
               self.board[i][0] == self.board[i][1] and\
               self.board[i][1] == self.board[i][2]:
                return self.board[i][1]
        if self.board[0][0] != state.NONE and \
                self.board[0][0] == self.board[1][1] and\
                self.board[1][1] == self.board[2][2]:
            return self.board[0][0]
        if self.board[0][2] != state.NONE and \
                self.board[0][2] == self.board[1][1] and\
                self.board[1][1] == self.board[2][0]:
            return self.board[0][2]
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == state.NONE:
                    return state.NONE
        return state.TIE
