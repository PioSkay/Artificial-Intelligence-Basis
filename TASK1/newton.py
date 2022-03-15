from session import context
import numpy
from controller import controller

class newton(controller):
    def __init__(self, cnx: context) -> None:
        controller.__init__(self, cnx)
    def iterate(self) -> bool:
        if self.shouldIterate():
            self.context.setNewX(self.context.getX() - 
                        (self.context.getValue()/self.context.getDerValue()))
            return True
        return False