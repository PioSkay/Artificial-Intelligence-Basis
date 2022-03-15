from abc import ABC, abstractmethod
from enum import Enum
from tracemalloc import start
from sympy import *
import numpy as np
import random

class stop_contidion(Enum):
    Iterations = 1
    DesiredValue = 2
    ComputationTime = 3

class x_type(Enum):
    RandomGenerateScalar = 0
    RandomGenerateVector = 1
    Fixed = 2

class context(ABC):
    @abstractmethod
    def __init__(self,
            stop_type: stop_contidion,
            value,
            start_value,
            x_type: x_type) -> None:
        self.stop_type = stop_type
        self.stop_value = value
        self.initialValue = start_value
        self.x_type = x_type
        self.reset()
    @abstractmethod
    def getValue(self):
        pass
    @abstractmethod
    def getDerValue(self):
        pass
    @abstractmethod
    def getDerDerValue(self):
        pass
    @abstractmethod
    def __str__(self):
        pass
    def getStopCondition(self) -> stop_contidion:
        return self.stop_type
    def reset(self):
        if self.x_type == x_type.Fixed:
            self.val = self.initialValue
        if self.x_type == x_type.RandomGenerateScalar:
            self.val = random.randint(self.initialValue[0][0], self.initialValue[0][1])
        if self.x_type == x_type.RandomGenerateVector:
            a = self.initialValue[0]
            b = self.initialValue[1]
            self.val = np.array(a)
            for i in range(a.shape[1]):
                self.val[0][i] = random.randint(a[0][i], b[0][i])
            self.val = np.transpose(self.val)
    def getStopConditionValue(self):
        return self.stop_value
    def setNewX(self, val):
        self.val = val
    def getX(self):
        return self.val
class context_F(context):
    def __init__(self, a: float, 
                b: float, 
                c: float, 
                d: float, 
                stop_type: stop_contidion,
                value: float,
                start_value: float,
                x_type: x_type) -> None:
        context.__init__(self, stop_type, value, start_value, x_type) 
        self.A = a
        self.B = b
        self.C = c
        self.D = d
        self.x = Symbol('x')
        self.func = (self.A * self.x ** 3) + (self.B * self.x ** 2) + (self.C * self.x) + (self.D)
        self.funcDer = diff(self.func)
        self.funcDerDer = diff(self.funcDer)
    def getValue(self):
        return self.func.subs(self.x, self.val)
    def getDerValue(self):
        return self.funcDer.subs(self.x, self.val)
    def getDerDerValue(self):
        return self.funcDerDer.subs(self.x, self.val)
    def __str__(self):
        return (f'Input data:\F(x) = {self.func} \n'
                 f'F(x)\' = {self.funcDer} \n'
                 f'Stop condition: {self.getStopCondition()} \n'
                 f'Stop value: {self.getStopConditionValue()} \n'
                 f'Current x: {self.val}')

class context_G(context):
    def __init__(self, A, 
                B, 
                C,
                stop_type: stop_contidion,
                value,
                start_value,
                x_type: x_type) -> None:
        if x_type == x_type.Fixed:
            context.__init__(self, stop_type, value, np.transpose(start_value), x_type)
        else:
            context.__init__(self, stop_type, value, start_value, x_type)  
        self.A = A
        self.B = np.transpose(B)
        self.C = C
        self.x = np.transpose(value)
        print(self.B)
        print(np.transpose(self.B))
    def getValue(self):  
        return self.C + np.dot(np.transpose(self.B),self.val) + np.dot(np.dot(np.transpose(self.val),self.A),self.val)
    def getDerValue(self):
        # b + 2 * A * x
        return self.B + 2 * np.dot(self.A, self.val)
    def getDerDerValue(self):
        # A + A^T
        return self.A + np.transpose(self.A)
    def __str__(self):
        return (f'Input data:\nc:\n {self.C}\nx:\n{self.val}\nA:\n{self.A}\nb:\n{self.B}\nF(x) = c + b^T * x + x^T * A * x\n'
                 f'F(x)\' = t + 2 * A * x \n'
                 f'Stop condition: {self.getStopCondition()} \n'
                 f'Stop value: {self.getStopConditionValue()} \n'
                 f'Current x: {self.val}')