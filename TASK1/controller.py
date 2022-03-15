from context import context, stop_contidion
import time

class controller:
    def __init__(self, cnx: context) -> None:
        self.context = cnx
        self.reset()
    def reset(self):
        self.stopValue = self.context.getStopConditionValue()
        if self.context.getStopCondition() == stop_contidion.ComputationTime:
            self.startTime = time.time()
        self.context.reset()
    def shouldIterate(self) -> bool:
        if self.context.getStopCondition() == stop_contidion.Iterations:
            if self.stopValue <= 0:
                return False
            else: 
                self.stopValue = self.stopValue - 1
                return True
        elif self.context.getStopCondition() == stop_contidion.DesiredValue:
            return self.context.getValue() <= self.stopValue
        elif self.context.getStopCondition() == stop_contidion.ComputationTime:
            return time.time() - self.startTime < self.stopValue
        return False