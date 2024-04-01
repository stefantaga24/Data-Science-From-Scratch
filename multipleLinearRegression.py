from linear_algebra import dot,Vector , vector_mean
from gradient_descent import gradient_step
from typing import List
import random 
import tqdm
def predict(x:Vector , beta:Vector) -> float:
    return dot(x,beta)

def error(x : Vector ,y: float , beta: Vector) -> float:
    return predict(x,beta) - y

def squared_error(x: Vector , y:float , beta: Vector) ->float:
    return error(x,y,beta)**2
def sqerror_gradient(x:Vector, y:float , beta:Vector) -> float:
    err = error(x,y,beta)
    return [x_i*err*2 for x_i in x]

def least_squares_fits(xs:List[Vector] , ys:List[float], learning_rate : float = 0.001,
                       num_steps: int = 1000, batch_size: int=1) ->float:
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc= "Gradient descent for multiple regression"):
        for start in range(0,len(xs),batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            gradient = vector_mean([sqerror_gradient(x,y,guess) for x,y in zip(batch_xs,batch_ys)])
            guess = gradient_step(guess,gradient,-learning_rate)
    
    return guess
x = [1,2,3]
y = 30
beta = [4,4,4]

print(y - dot(x,least_squares_fits([x],[y])))