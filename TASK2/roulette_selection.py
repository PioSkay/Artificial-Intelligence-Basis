from context import context
import numpy as np

class roulette_selection:
    def __init__(self, context: context, population):
        self.context = context
        self.pop = population
    def getFitness(self):
        fitnessArray = []
        for val in self.pop:
            fitnessArray.append(self.context.fun(np.transpose([val])))
        max_val = max(fitnessArray)
        min_val = min(fitnessArray)
        #normalize values to [0, 1]
        if max_val == min_val:
            for i in range(len(fitnessArray)):
                fitnessArray[i] = 1
        else:
            fitnessArray = (fitnessArray - min_val) / (max_val - min_val)
        array_sum = sum(fitnessArray)
        return fitnessArray, array_sum
    def getWheel(self):
        prev_val = 0
        wheel = []
        for i in range(len(self.fitness)):
            self.fitness[i] = self.fitness[i] / self.sum
        if self.sum != 0:
            for val in self.fitness:
                curr_val = prev_val + (val / 1)
                wheel.append(curr_val)
                prev_val = curr_val
        return wheel
    def generateParents(self):
        parents = []
        self.fitness, self.sum = self.getFitness()
        self.wheel = self.getWheel()
        for x in range(self.context.pop_size):
            spin_wheel = np.random.uniform(0, 1)
            i = 0
            while i in range(len(self.wheel)):
                if(self.wheel[i] >= spin_wheel):
                    break
                i = i + 1
            parents.append(self.pop[i])
        return parents