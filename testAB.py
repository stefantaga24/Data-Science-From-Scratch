from typing import *
import math
import testingACoin as tAC


def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    How likely are we to see a value at least as extreme as x (in either
    direction) if our values are from an N(mu, sigma)?
    """
    if x >= mu:
    # x is greater than the mean, so the tail is everything greater than x
        return 2 * tAC.normal_probability_above(x, mu, sigma)
    else:
    # x is less than the mean, so the tail is everything less than x
        return 2 * tAC.normal_probability_below(x, mu, sigma)

def estimated_parameters(N:int ,n :int) -> Tuple[float,float]:
    p = n/N 
    """p is the probability of a client seeing the ad"""
    sigma = math.sqrt(p*(1-p) /N)
    return p ,sigma

def a_b_test_statistic (N_a: int , n_A:int ,N_B : int, n_B:int ) -> float:
    p_A,sigma_A = estimated_parameters(N_a,n_A)
    p_B,sigma_B = estimated_parameters(N_B,n_B)
    return (p_A-p_B) / math.sqrt((sigma_A**2) + (sigma_B**2))


z = a_b_test_statistic(1000, 200, 1000, 150) # -2.94
print(two_sided_p_value(z)) 
