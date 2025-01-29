import numpy as np
from feature_tuning import regularize
from typing import Tuple

def predict(x: np.array, w: np.array, b):
    return np.dot(x, w) + b

def compute_cost(x: np.array,
    y: np.array,
    w: np.array,
    b: float,
    lambda_ = 0) -> int:
    
    m, n = x.shape 
    cost = 0 
    for i in range(m): 
        cost += (predict(x[i], w, b) - y[i]) ** 2
    cost /= (2 * m)

    reg_cost = regularize(w, m, lambda_)
    
    total_cost = cost + reg_cost
    return total_cost
