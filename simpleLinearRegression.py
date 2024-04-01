import typing
from linear_algebra import Vector
def predict(alpha : float , beta : float, x_i : float ) -> float:
    return beta*x_i+alpha

def error(alpha:float, beta:float, x_i:float ,y_i:float) ->float:
    return predict(alpha,beta,x_i) - y_i


def sum_of_sqerrors(alpha:float,beta:float, x: Vector,y:Vector) -> float:
    return sum(error(alpha,beta,x_i,y_i)**2 for x_i,y_i in zip(x,y))

from typing import Tuple 
from Stats import correlation , standard_deviation, mean, de_mean

def least_squares_fit(x:Vector, y:Vector ) -> Tuple[float,float]:
    beta = correlation(x,y)*standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta*mean(x)
    return alpha,beta

x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]
# Should find that y = 3x - 5
assert least_squares_fit(x, y) == (-5, 3)

def total_sum_of_squares(y:Vector) ->float:
    """the total squared variation of y_i's from their mean"""
    return sum(v**2 for v in de_mean(y))

def r_squared(alpha : float , beta:float , x:Vector, y :Vector) ->float:
    """
    the fraction of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model
    """
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) /
                        total_sum_of_squares(y))

