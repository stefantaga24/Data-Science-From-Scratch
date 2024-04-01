from typing import Tuple,List
from linear_algebra import Vector
from multipleLinearRegression import least_squares_fits
import datetime

def estimate_sample_beta(pairs : List[Tuple[Vector,float]]):
    x_sample = [x for x,_ in pairs]
    y_sample = [y for y,_ in pairs]
    beta = least_squares_fits(x_sample,y_sample,0.001,5000,20)


