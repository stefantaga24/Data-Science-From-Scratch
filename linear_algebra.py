from matplotlib import pyplot as plt
from collections import Counter
from typing import *
import math
Vector = List[float]
Matrix = List[List[float]]
def add(v: Vector , w:Vector):
    assert(len(v) == len(w))
    return [v_i+w_i for v_i,w_i in zip(v,w)]
def subtract(v:Vector ,w:Vector):
    assert(len(v) == len(w))
    return [v_i-w_i for v_i,w_i in zip(v,w)]
def vector_sum(vectors: List[Vector]) -> Vector:
    assert(len(vectors))
    first_length = len(vectors[0])
    assert(first_length == len(vectors[i]) for i in range(0,len(vectors)))
    return [sum(vector[i] for vector in vectors) for i in range(first_length)]
def scalar_multiply(c: float , v:Vector) -> Vector:
    assert(len(v))
    return [c*v[i] for i in range(len(v))]
def vector_mean(vectors: List[Vector]) -> Vector:
    assert(len(vectors))
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))
def dot( a:Vector, b:Vector):
    assert (len(a)==len(b))
    return (a_i*b_i for a_i,b_i in zip(a,b))
def sum_squares(v : Vector) :
    return dot(v,v)
def magnitude ( v: Vector):
    return math.sqrt(sum_squares(v))
def squared_distance (v: Vector , w : Vector) :
    return sum_squares(subtract(v,w))
def distance (v:Vector, w:Vector):
    return math.sqrt(squared_distance(v,w))
def shape ( A:Matrix):
    rows = len(A)
    columns = len(A[0])
    return rows,columns
def get_row(A : Matrix , i:int):
    return A[i]
def get_column (A:Matrix , j:int):
    return [A[i][j] for i in range(len(A))]
def make_matrix(num_rows : int , num_columns: int , entry_fn: Callable[[int,int], float]) -> Matrix:
    return [[entry_fn(i,j) for i in range(num_rows)] for j in range(num_columns)]
