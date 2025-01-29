import random
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from linear_regression import predict as p_linear, compute_cost as lin_cost
from logistic_regression import predict as p_logistic, compute_cost as log_cost
from gradient_descent import gradient_descent, compute_gradient
from feature_tuning import feature_scaling, mean_normalization, z_score_normalization

def predict(x, w, b):
    p = np.dot(x, w) + b
    return p


    
X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1.


w, b = gradient_descent(X_tmp, y_tmp, w_tmp, p_logistic, b_tmp, 0.1, 10000)
print(f"Gradient Descent Result w:{w}, b:{b}")

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient(X_tmp, y_tmp, w_tmp, b_tmp, p_logistic, lambda_tmp)
print((dj_db_tmp, dj_dw_tmp))