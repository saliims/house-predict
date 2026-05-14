import numpy as np
from src.model import compute_gradient, compute_cost, predict
import matplotlib.pyplot as plt
from src.utils import rmse, r_squared

def gradient_descent(X, y, alpha, num_iters):
    theta = np.zeros(X.shape[1])
    cost_history = []
    
    for _ in range(num_iters):
        theta = theta - alpha * compute_gradient(X, y, theta)
        cost_history.append(compute_cost(X, y, theta))
        
    return theta, cost_history

def evaluate(X_train, X_val, y_train, y_val, theta, y_mu, y_sigma):
    y_train_pred = predict(X_train, theta) * y_sigma + y_mu
    y_val_pred = predict(X_val, theta) * y_sigma + y_mu
    
    print(f"Train RMSE: ${rmse(y_train, y_train_pred):,.0f}")
    print(f"Val   RMSE: ${rmse(y_val,   y_val_pred):,.0f}")
    print(f"Train R²:   {r_squared(y_train, y_train_pred):.4f}")
    print(f"Val   R²:   {r_squared(y_val,   y_val_pred):.4f}")
    
    return y_val_pred