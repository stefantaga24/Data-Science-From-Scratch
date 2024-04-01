from linear_algebra import Vector,dot,add
from multipleLinearRegression import *
def ridge_penalty (beta : Vector, alpha:float) ->float:
    return alpha*dot(beta[1:],beta[1:])

def squared_error_ridge(x:Vector,
                        y:float,
                        beta:Vector,
                        alpha:float) -> float:
    return error(x,y,beta) ** 2 + ridge_penalty(beta,alpha)

def ridge_penalty_gradient(beta:Vector, alpha:float) -> Vector:
    return [0.] + [2*alpha*beta_j for beta_j in beta[1:]]

def sqerror_ridge_gradient(x:Vector,
                           y:float,
                           beta:Vector,
                           alpha : float) ->Vector:
    return add(sqerror_gradient(x,y,beta), ridge_penalty_gradient(beta,alpha))

def least_squares_fits(xs:List[Vector] , ys:List[float], learning_rate : float = 0.001,
                       num_steps: int = 1000, batch_size: int=1) ->float:
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc= "Gradient descent for multiple regression"):
        for start in range(0,len(xs),batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            gradient = vector_mean([sqerror_ridge_gradient(x,y,guess) for x,y in zip(batch_xs,batch_ys)])
            guess = gradient_step(guess,gradient,-learning_rate)
    
    return guess