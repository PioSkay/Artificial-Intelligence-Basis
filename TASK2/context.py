from tkinter import EXCEPTION
import numpy as np
import json
import os

FILE_DIR = "config.json"

def get(str: str, data):
    if str in data:
        return data[str]
    else:
        raise Exception(f'Could not find {str}')

class context:
    def __init__(self):
        try:
            with open(FILE_DIR) as file:
                data = json.load(file)
        except:
            raise Exception("Could not find: " + os.getcwd() + "/" + FILE_DIR)
        self.A = np.asarray(get('A', data))
        self.b = np.asarray(get('b', data))
        if self.A.shape[0] != self.A.shape[1] or self.A.shape[0] != self.b.shape[1]:
            raise Exception("Invalid A or b size!")
        self.c = int(get('c', data))
        self.d = int(get('d', data))
        self.pop_size = int(get('pop_size', data))
        self.cross_prob = float(get('cross_prob', data))
        self.mut_prob = float(get('mut_prob', data))
        self.it = int(get('it', data))
    def fun(self, x):
        return (np.dot(np.dot(np.transpose(x), self.A), x) + \
                np.dot(self.b, x) + self.c)[0][0]
    def generatePopulation(self):
        pop = []
        d = 2**self.d
        for x in range(self.pop_size):
            val = np.random.randint(-d, d, self.d)
            pop.append(val)
        return np.asarray(pop)
    def getIter(self):
        return self.it
    def __str__(self):
        return f'Input data:\nA:\n{self.A}\n' + \
                f'b^T:\n{self.b}\nc:\n{self.c}\nd:\n{self.d}\n' + \
                f'Population size:\n{self.pop_size}\n' + \
                f'Crossover probability:\n{self.cross_prob}\n' + \
                f'Mutation probability:\n{self.mut_prob}\n' + \
                f'Iterations:\n{self.it}\n'