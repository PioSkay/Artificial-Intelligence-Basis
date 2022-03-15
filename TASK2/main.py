from context import context
from roulette_selection import roulette_selection
from crossover import crossover
from mutate import mutate
import numpy as np

try:
    x = context()
except Exception as e:
    print(e)

currPopulation = x.generatePopulation()
print(currPopulation)
y = roulette_selection(x, currPopulation)
z = crossover(x, y.generateParents())
currPopulation = mutate(x, z.generateCrossovered())
print(currPopulation)

#for i in range(x.getIter()):
    #print(i)
