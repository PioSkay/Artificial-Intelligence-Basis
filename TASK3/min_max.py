from base import base, state


class min_max(base):
    def __init__(self):
        super().__init__()

    def max(self, alpha, beta):
        status = self.status()
        if status != state.NONE:
            return (status.value, 0, 0)
        max_score = -2
        x = None
        y = None
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == state.NONE:
                    self.board[i][j] = state.O
                    (score, loc_x, loc_y) = self.min(alpha, beta)
                    if max_score < score:
                        (max_score, x, y) = (score, i, j)
                    self.board[i][j] = state.NONE
                    #alpha beta pruning
                    if max_score >= beta:
                        return (max_score, x, y)
                    if max_score > alpha:
                        alpha = max_score
        return (max_score, x, y)

    def min(self, alpha, beta):
        status = self.status()
        if status != state.NONE:
            return (status.value, 0, 0)
        max_score = 2
        x = None
        y = None
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == state.NONE:
                    self.board[i][j] = state.X
                    (score, loc_x, loc_y) = self.max(alpha, beta)
                    if max_score > score:
                        (max_score, x, y) = (score, i, j)
                    self.board[i][j] = state.NONE
                    #alpha beta pruning
                    if max_score <= alpha:
                        return (max_score, x, y)
                    if max_score < beta:
                        beta = max_score
        return (max_score, x, y)
                