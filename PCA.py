from linear_algebra import *
from typing import *
from gradient_descent import *
import tqdm
Vector = List[float]
def de_mean(data: List[Vector]) -> List[Vector]:
    """Recenters the data to have mean 0 in every dimension"""
    mean = vector_mean(data)
    return [subtract(vector, mean) for vector in data]
def direction(w:Vector):
    mag = magnitude(w)
    return [w_i/mag for w_i in w]

def directional_variance(data,w):
    w_dir = direction(w)
    return sum(dot(v,w_dir)**2 for v in data)
def directional_variance_gradient(data,guess):
    guess_dir = direction(guess)
    return [sum(2*dot(v,guess_dir)*v[i] for v in data) for i in range(len(guess))]
def first_principal_component(data , n=100 , step_size = 0.1):
    guess = [1.0 for _ in data[0]]
    with tqdm.trange(n) as t:
        for _ in t:
            dv = directional_variance(data,guess)
            gradient = directional_variance_gradient(data,guess)
            guess = gradient_step(guess,gradient,step_size)
            t.set.description(f"dv: {dv:.3f}")
def project(v: Vector, w: Vector) -> Vector:
    """return the projection of v onto the direction w"""
    projection_length = dot(v, w)
    return scalar_multiply(projection_length, w)
def remove_projection_from_vector(v,w):
    return subtract(v,project(v,w))
def remove_projection(data,w):
    return [remove_projection_from_vector(v,w) for v in data]

def pca(data, num_components) :
    components = []
    for _ in range(num_components):
        component = first_principal_component(data)
        data = remove_projection(data,component)
        components.append(component)
    return components

def transform_vector (v,components):
    return [dot(v,w) for w in components]
def transform(data,components):
    return [transform_vector(v,components) for v in data]
