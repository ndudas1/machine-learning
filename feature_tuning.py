import numpy as np

def feature_scaling(x: np.array) -> np.array:
    max = np.max(x, axis=0) # np.zeros(n)
    return x / max

def mean_normalization(x: np.array) -> np.array:
    mu = np.mean(x, axis=0)
    min = np.min(x, axis=0)
    max = np.max(x, axis=0)
    return (x - mu) / (max - min)

def z_score_normalization(x: np.array) -> np.array:
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x - mu) / sigma

def regularize(w: np.array, m: int, lambda_: float) -> float:
    if lambda_ == 0:
        return 0.0
    
    n = len(w)
    if lambda_ > 0:
        reg_cost = 0
        for j in range(n):
            reg_cost += (w[j]**2)
        reg_cost = (lambda_/(2*m)) * reg_cost
    return reg_cost
