from session import session
from context import context, stop_contidion
from controller import controller

class gradient_descent(controller):
    def __init__(self, cnx: context) -> None:
        controller.__init__(self, cnx)
        self.learning_rate = 0.001
    def iterate(self) -> bool:
        if self.shouldIterate():
            self.context.setNewX(self.context.getX() - self.learning_rate*(self.context.getDerValue()))
            return True
        return False