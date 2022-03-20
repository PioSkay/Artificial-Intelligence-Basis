from context import context
from roulette_selection import roulette_selection
from crossover import crossover
from mutate import mutate
from scipy.stats import mode
import numpy as np

try:
    x = context()
except Exception as e:
    print(e)

currPopulation = x.generatePopulation()
print("Calculating...")
for i in range(x.getIter()):
    y = roulette_selection(x, currPopulation)
    y = y.generateParents()
    z = crossover(x, y)
    z = z.generateCrossovered()
    currPopulation = mutate(x, z)
print("Output:")
print("Current population:")
print(currPopulation)
output = np.transpose(mode(currPopulation)[0])
print(f'Solution: \n {output}')
print(f'F(output) = \n {x.fun(output)}')
