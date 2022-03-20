from context import context
import numpy as np

class crossover:
    def __init__(self, context: context, population):
        self.context = context
        self.population = population
        self.crossovered = self.initCrossovered()
        self.pairs = self.generatePairs()
    def generatePairs(self):
        pairs_array = []
        pop_size = len(self.population)

        while 0 < pop_size:
            first = self.population[0]
            self.population.pop(0)
            #randomly generate second index
            secondIndex = np.random.randint(0, pop_size - 1)
            second = self.population[secondIndex]
            self.population.pop(secondIndex)
            pairs_array.append((first, second))
            #increase the iterators
            pop_size = pop_size - 2
        return pairs_array
    def initCrossovered(self):
        if len(self.population) % 2 == 1:
            index = np.random.randint(0, len(self.population) - 1)
            to_skip = self.population.pop(index)
            return [[np.binary_repr(x) for x in to_skip]]
        else:
            return []
    def generateCrossovered(self):
        crossovered = self.crossovered
        for pair in self.pairs:
            if np.random.uniform(0, 1) <= self.context.cross_prob:
                child1 = []
                child2 = []
                cross_point = np.random.randint(1, len(pair[0]))
                it = 0
                while it < cross_point:
                    child1.append(np.binary_repr(pair[0][it]))
                    child2.append(np.binary_repr(pair[1][it]))
                    it = it + 1
                while it < len(pair[0]):
                    child1.append(np.binary_repr(pair[1][it]))
                    child2.append(np.binary_repr(pair[0][it]))
                    it = it + 1
                crossovered.append(child1)
                crossovered.append(child2)
            else:
                crossovered.append([np.binary_repr(x) for x in pair[0]])
                crossovered.append([np.binary_repr(x) for x in pair[1]])
        return crossovered

