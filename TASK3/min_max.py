from base import base, state
from enum import Enum
class alg(Enum):
    min = True,
    max = False
class min_max(base):
    def __init__(self):
        super().__init__()
    def min_max_with_alpha_beta(self, alpha, beta, type=alg.max):
        status = self.status()
        if status != state.NONE:
            return (status.value, 0, 0)
        max_score = self.ifMax(-2, 2, type)
        x = None
        y = None
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == state.NONE:
                    self.board[i][j] = self.ifMax(state.O, state.X, type)
                    (score, loc_x, loc_y) = self.min_max_with_alpha_beta(alpha, beta, self.ifMax(alg.min, alg.max, type))
                    if self.ifMax(max_score < score, max_score > score, type) == True:
                        (max_score, x, y) = (score, i, j)
                    self.board[i][j] = state.NONE
                    #alpha beta pruning
                    if self.ifMax(max_score >= beta, max_score <= alpha, type) == True:
                        return (max_score, x, y)
                    if self.ifMax(max_score > alpha, max_score < beta, type) == True:
                        if type == alg.max:
                            alpha = max_score
                        else:
                            beta = max_score
        return (max_score, x, y)

    def ifMax(self, ifTrue, ifFalse, type):
        return ifTrue if type == alg.max else ifFalse
        