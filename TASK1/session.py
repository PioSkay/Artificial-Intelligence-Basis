from enum import Enum
from context import context, context_F, context_G, stop_contidion, x_type
from numpy import *
import re

def getVector(name: str, type: str):
    inp = input(f'{name} >>> Please input a {type}: ')
    x = re.findall(r"\(([+-]?\d+(?:,\d+)*)\)", inp)
    x = [part.split(',') for part in x]
    arr = array(x, dtype=float)
    return arr

def getVectorOfSize(name: str, size):
    print(f'Input vector of size {size}')
    while True:
        try:
            x = getVector(name, "vector")
        except:
            print("Invalid vector")
            continue
        if len(x) == 1 and len(x[0]) == size:
            break
        print("Invalid vector dimension!")
    return x

def getFloat(str):
    while True:
        try:
            output = float(input(f'{str} >>> '))
        except:
            print("Invalid value")
            continue
        break
    return output

def is_pos_def(x):
    return all(linalg.eigvals(x) > 0)
        
class type(Enum):
    F_fun = 1
    G_fun = 2

class method(Enum):
    Gradient = 1
    Newton = 2

class session:
    def __init__(self) -> None:
        print("----INPUT-PARAMETERS----")
        #input method
        print("Specify the method: ")
        print("\t 1 - Type for Gradient Descent")
        print("\t 2 - Type for Newton's")
        x = 0
        while int(x) != 1 and int(x) != 2: 
            x = input()
        self.method = method(int(x))

        #input method
        print("Specify the function: ")
        print("\t 1 - F(x)")
        print("\t 2 - G(x)")
        x = 0
        while int(x) != 1 and int(x) != 2: 
            x = input()
        self.type = type(int(x))

        print("Specify the function parameters: ")
        if self.type == type.F_fun:
            self.A = getFloat("A") #float(input("A >>> "))
            self.B = getFloat("B") #float(input("B >>> "))
            self.C = getFloat("C") #float(input("C >>> "))
            self.D = getFloat("D") #float(input("D >>> "))
        elif self.type == type.G_fun:
            self.C = getFloat("C") #float(input("C >>> "))
            #inputs a vector
            self.D = int(getFloat("D (vector dimension) ")) #int(input("D (vector dimension) >>> "))
            while True:
                try:
                    self.B = getVector("B", "vector")
                except:
                    print("Invalid vector")
                    continue
                if len(self.B) == 1 and len(self.B[0]) == self.D:
                    break
                print(f'Invalid vector dimension!')
            while True:
                try:
                    self.A = getVector("A", "matrix")
                except:
                    print("Invalid matrix")
                    continue
                if len(self.A) == self.D and len(self.A[0]) == self.D:
                    if is_pos_def(self.A) == False:
                        print("Not a positive-definite matrix")
                        continue
                    break
                print("Invalid matrix dimension!")
        print("----START-PARAMETERS----")
    
        print("Would you like to randomly generate x: ")
        print("\t 1 - yes")
        print("\t 2 - no")
        x = 0
        while int(x) != 1 and int(x) != 2: 
            x = int(getFloat(""))
        if x == 1:
            if self.type == type.F_fun:
                self.x  = getVectorOfSize("", 2)
                self.x_type = x_type.RandomGenerateScalar
            elif self.type == type.G_fun:
                self.x  = array([getVectorOfSize("", self.D), getVectorOfSize("", self.D)])
                self.x_type = x_type.RandomGenerateVector
        else:
            if self.type == type.F_fun:
                self.x = float(input("Start parameter: "))
            elif self.type == type.G_fun:
                while True:
                    try:
                        self.x = getVector("x", "vector")
                    except:
                        print("Invalid vector")
                        continue
                    if len(self.x) == 1 and len(self.x[0]) == self.D:
                        self.x_type = x_type.Fixed
                        break
                    print("Invalid vector dimension!")
        print("----STOP-CONDITIONS----")
        print("Specify the stop condition: ")
        print("\t 1 - Iterations")
        print("\t 2 - DesiredValue")
        print("\t 3 - ComputationTime")
        x = 0
        while int(x) != 1 and int(x) != 2 and int(x) != 3: 
            x = input()
        self.stop_cond = stop_contidion(int(x))
        if self.stop_cond == stop_contidion.ComputationTime or self.stop_cond == stop_contidion.Iterations:
            self.stop_val = float(input(f'Specify the number of ' 
                            f'{"seconds" if self.stop_cond == stop_contidion.ComputationTime else "iterations"}:'))
        if self.stop_cond == stop_contidion.DesiredValue:
            if self.type == type.F_fun:
                self.stop_val = float(input(f'Specify the desired value: ' ))
            elif self.type == type.G_fun:
                self.stop_val = input(f'Specify the desired vector: ' )
        print("generating context...")
        if self.type == type.F_fun:
            self.m_context = context_F(self.A, self.B, self.C, self.D, self.stop_cond, self.stop_val, self.x, self.x_type)
        elif self.type == type.G_fun:
            self.m_context = context_G(self.A, self.B, self.C, self.stop_cond, self.stop_val, self.x, self.x_type)
    def __str__(self):
        return (f'Session data: \n' 
            f'Method: {self.method} \n' 
            f'Function type: {self.type} \n'
            f'{self.m_context.__str__()}')
    def getContext(self) -> context:
        return self.m_context
    def getMethod(self) -> method:
        return self.method
    def getType(self) -> type:
        return self.type