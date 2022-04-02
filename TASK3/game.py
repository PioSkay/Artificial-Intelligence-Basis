from turtle import st
from min_max import min_max, alg
from base import state
from enum import Enum

class mode(Enum):
    AIvsAI = 0
    AIvsPlayerX = 1
    AIvsPlayerO = 2
    PlayervsPlayer = 3

class game(min_max):
    def __init__(self, m = mode.AIvsAI):
        super().__init__()
        self.mode = m

    def play(self):
        turn = state.X
        while True:
            result = self.status()
            if result != result.NONE:
                return result
            if turn == state.X:
                if self.mode == mode.AIvsPlayerX or self.mode == mode.PlayervsPlayer:
                    self.printBoard(turn)
                    (x, y) = self.getValidInput()
                    self.board[x][y] = state.X
                else:
                    (max, x, y) = self.min_max_with_alpha_beta(-2, 2, alg.min)
                    self.board[x][y] = state.X
                turn = state.O
            else:
                if self.mode == mode.AIvsPlayerO or self.mode == mode.PlayervsPlayer:
                    self.printBoard(turn)
                    (x, y) = self.getValidInput()
                    self.board[x][y] = state.O
                else:
                    (max, x, y) = self.min_max_with_alpha_beta(-2, 2)
                    self.board[x][y] = state.O
                turn = state.X
    def reset(self):
        self.initBoard()

    def changeMode(self, mode: mode):
        self.mode = mode

    def getValidInput(self):
        while True:
            try:
                x = int(input("Please provide X param "))
                y = int(input("Please provide Y param "))
            except ValueError:
                continue
            if self.isValid(x, y) == True:
                return (x, y)  
            else:
                print("Invalid input!")
    def printBoard(self, round):
        print(f"Player tour: {round}")
        for i in range(3):
            print(f"\t{self.board[i][0]} | {self.board[i][1]} | {self.board[i][2]}")