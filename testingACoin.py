from typing import *
import probabilities
import math


def normal_approximation_to_binomial(n: int , p:float) -> Tuple[float,float]:
    mu = p*n
    sigma = math.sqrt(p*(1-p)*n)
    return mu,sigma

normal_probability_below = probabilities.normal_cdf

def normal_probability_above (lo: float,mu:float =0, sigma: float =1 )->float:
    return 1-probabilities.normal_cdf(lo,mu,sigma)

def normal_probability_between(lo: float, hi:float ,mu: float =0, sigma: float=1) -> float:
    return probabilities.normal_cdf(hi,mu,sigma) - probabilities.normal_cdf(lo,mu,sigma)
def normal_probability_outside(lo:float , hi: float , mu: float =0 , sigma: float =1) -> float:
    return 1 - normal_probability_between(lo,hi,mu,sigma)

def normal_upper_bound(probability,mu =0 , sigma = 1):
    return probabilities.inverse_normal_cdf(probability,mu,sigma)
def normal_lower_bound(probability,mu =0 ,sigma = 1):
    return probabilities.inverse_normal_cdf(1-probability,mu,sigma)
def normal_two_sided_bounds(probability: float,mu: float = 0,sigma: float = 1) -> Tuple[float, float]:
    """
    Returns the symmetric (about the mean) bounds
    that contain the specified probability
    """
    tail_probability = (1 - probability) / 2
    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    return lower_bound, upper_bound


mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)