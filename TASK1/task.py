from time import sleep
from sympy import MatrixSlice, MatrixSymbol
from gradient_descent import gradient_descent
from newton import newton
from session import session, method, type

from numpy import *
from context import context, context_G, context_F, stop_contidion, x_type
import re

def _int_mean(list: list):
    if len(list) == 1:
        return list[0]
    out = 0
    for i in list:
        out += i
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i][j] = out[i][j] / len(list)
    return out

def _int_standardDeviation(list: list, avg):
    output = []
    for out in list:
        output.append(out)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                output[len(output) - 1][i][j] = (out[i][j] - avg[i][j]) ** 2
    output = _int_mean(output)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i][j] = sqrt(output[i][j])
    return output

x = session()
print("------SESSION-INITIALIZED------")
print(x)
print("------SESSION-INITIALIZED------")
print("Would you like to run the code in batch mode? (yes/no)")
print("1 - yes")
print("2 - no")
while True:
    try:
        mode = int(input())
    except:
        continue
    if mode == 1 or mode == 2:
        break
if mode == 1:
    while True:
        try:
            it = int(input("Please specify the number of repetitions: "))
        except:
            continue
        break
    print("Batch mode activated")
else:
    it = 1

if x.getMethod() == method.Gradient:
    alg = gradient_descent(x.getContext())
elif x.getMethod() == method.Newton:
    alg = newton(x.getContext())

fun_values = []
x_values = []
while it >= 1:
    alg.reset()
    while alg.iterate():
        pass
    #gather data
    if x.getType() == type.F_fun:
        fun_values.append(float(x.getContext().getValue()))
        x_values.append(float(x.getContext().getX()))
    else:
        fun_values.append(x.getContext().getValue())
        x_values.append(x.getContext().getX())
    it -= 1

if mode == 1:
    if x.getType() == type.F_fun:
        print(f'Mean value of a function values: \n{mean(fun_values)}')
        print(f'Mean value of a x values: \n{mean(x_values)}')
        print(f'Standard deviation of a function values: \n{std(fun_values)}')
        print(f'Standard deviation of a x values: \n{std(x_values)}')
    elif x.getType() == type.G_fun:
        print(f'Mean value of a function values: \n{_int_mean(fun_values)}')
        print(f'Mean value of a x values: \n{_int_mean(x_values)}')
        print(f'Standard deviation of a function values: \n{_int_standardDeviation(fun_values, _int_mean(fun_values))}')
        print(f'Standard deviation of a x values: \n{_int_standardDeviation(x_values, _int_mean(x_values))}')
else:
    print(f'Value of a function: \n{fun_values[0]}')
    print(f'Value of x: \n{x_values[0]}')