import typing
from typing import List
import math
from linear_algebra import dot,Vector,vector_sum
def logistic(x:float) -> float:
    return 1.0 / (1+math.exp(-x))

def logistic_prime(x:float) -> float:
    y = logistic(x)
    return y*(1-y)

def _negative_log_likelihood(x: Vector, y: float, beta: Vector) -> float:
    """The negative log likelihood for one data point"""
    if y == 1:
        return -math.log(logistic(dot(x, beta)))
    else:
        return -math.log(1 - logistic(dot(x, beta)))
    
def _negative_log_likelihood(xs : List[Vector], ys:List[float], beta:Vector) ->float:
        return sum(_negative_log_likelihood(x_i,y_i,beta) for x_i,y_i in zip(xs,ys))

def _negative_log_partial_j(x:Vector, y:float, beta:Vector, j:int) -> float:
    return -(y - logistic(dot(x,beta))) * x[j]

def _negative_log_gradient(x:Vector , y : float, beta : Vector) -> Vector:
     return [_negative_log_partial_j(x,y,beta,j) for j in range(0,len(beta))]

def negative_log_gradient(xs : List[Vector], ys:List[Vector], beta:Vector) -> Vector:
     return vector_sum([_negative_log_gradient(x,y,beta) for x,y in zip(xs,ys)])