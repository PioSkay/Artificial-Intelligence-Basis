from xmlrpc.client import boolean
from context import context
import numpy as np

def should_mutate(context: context) -> bool:
    return np.random.uniform(0, 1) <= context.mut_prob

def mutate_element(context: context, element):
    for iter in range(len(element)):
        curr = list(element[iter])
        if should_mutate(context) == True:
            if curr[0] == '-':
                curr[0] = ''
            else:
                curr.insert(0, '-')
        for i in range(len(curr)):
            if should_mutate(context) == True:
                if curr[i] == '1':
                    curr[i] = '0'
                elif curr[i] != '-':
                    curr[i] = '1'
        element[iter] = ''.join(curr)
    return element

def mutate(context: context, population):
    output = []
    for ele in population:
        after_mutation = mutate_element(context, ele)
        output.append(np.asarray([int(x, 2) for x in after_mutation]))
    return output