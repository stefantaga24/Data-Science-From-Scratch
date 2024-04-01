from linear_algebra import Vector,dot
import linear_algebra as linear_algebra
from typing import Callable,TypeVar,List,Iterator
from matplotlib import pyplot as plt
import random
import numpy as np
from linear_algebra import distance,add,scalar_multiply

def sum_of_squares(v: Vector) -> float:
    """ Computes the sum of squared elements in v"""
    return dot(v,v)

def difference_quotient(f: Callable[[float], float] , x: float, h:float) -> float:
    return ( f(x+h) - f(x) ) / h
def square(x: float) -> float:
    return x*x
def derivative(x: float) -> float:
    return 2*x

def partial_difference_quotient(f:Callable[[Vector],float] , v: Vector, i:int, h:float) -> float:
    #Find the difference quotient when modifying only the variable at the position i
    w = [v[j] (h if i==j else 0) for j in range(0,len(v))]
    return (f(w)-f(v))/h
 
def estimate_gradient (f:Callable[[Vector],float],v:Vector,h:float =0.0001):
    return [partial_difference_quotient(f,v,i,h) for i in range(len(v))]

def gradient_step(v:Vector , gradient:Vector, step_size: float) -> Vector:
    assert(len(v) == len(gradient))
    step = scalar_multiply(step_size,gradient)
    return add(v,step)
def sum_of_squares_gradient(v:Vector)->Vector:
    return [2*v_i for v_i in v]

def linear_gradient(x:float,y:float ,theta:Vector)->Vector:
    slope,intercept = theta
    predicted =slope*x+intercept
    error = (predicted-y)
    squared_error = error**2
    grad = [2*error*x,2*error]
    return grad

T = TypeVar('T')
def minibatches(dataset : List[T], batch_size:int ,shuffle:bool = True) -> Iterator[List[T]]:
    #start indexes : 0 , batch_size, 2*batch_size ...
    batch_starts = [start for start in range(0,len(dataset),batch_size)]
    if shuffle : random.shuffle(batch_starts)
    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]



inputs = [(x,20*x+5) for x in range(-50,50)]
theta = [random.uniform(-1,1) , random.uniform(-1,1)]

learning_rate = 0.0001

for epoch in range(0):
    np.random.shuffle(inputs)
    for x,y in inputs:
        grad = linear_gradient(x,y,theta)
        theta = gradient_step(theta,grad,-learning_rate)
        print(epoch,theta)