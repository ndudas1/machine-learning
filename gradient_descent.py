import numpy as np
from typing import Callable, Tuple

def compute_gradient(x: np.array,
    y: np.array,
    w: np.array,
    b: float,
    compute_gradient_fn: Callable[[np.array, np.array, float], Tuple[float, float]],
    lambda_ = 0.0) -> Tuple[float, float]:

    m, n = x.shape   
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    
    for i in range(m):                             
        err = compute_gradient_fn(x[i], w, b) - y[i]   
        for j in range(n):                         
            dj_dw[j] += err * x[i, j]    
        dj_db += err
                      
    dj_dw /= m                                
    dj_db /= m

    if lambda_ > 0:
        for j in range(n):
            dj_dw[j] += (lambda_/m) * w[j]                          
        
    return dj_dw, dj_db

def gradient_descent(x: np.array,
    y: np.array,
    w: np.array,
    compute_gradient_fn: Callable[[np.array, np.array, float], Tuple[float, float]],
    b=0.0,
    lambda_ = 0.0,
    learning_rate=0.01,
    max_iterations=1000):
    
    for i in range(max_iterations):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = compute_gradient(x, y, w, b, compute_gradient_fn, lambda_)

        # Update Parameters using equation (3) above
        w = w - (learning_rate * dj_dw)
        b = b - (learning_rate * dj_db)

    return w, b