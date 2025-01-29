import numpy as np
from feature_tuning import regularize

def __sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(x: np.array, w: np.array, b):
   z = np.dot(x, w) + b
   return __sigmoid(z)

def compute_cost(x: np.array,
    y: np.array,
    w: np.array,
    b: float,
    lambda_ = 0):
    
    m, _ = x.shape
    cost = 0.0
    for i in range(m):
        g_z = predict(x[i], w, b)
        y_i = y[i]
        cost += -y_i * np.log(g_z) - (1 - y_i) * np.log(1 - g_z)
    cost /= m

    reg_cost = regularize(w, m, lambda_)
    
    total_cost = cost + reg_cost
    return total_cost
