import numpy as np

def predict(X, theta):
    return X @ theta

def compute_cost(X, y, theta):
    m = X.shape[0]
    return (1 / (2 * m)) * np.sum((predict(X, theta) - y) ** 2)

def compute_gradient(X, y, theta):
    m = X.shape[0]
    residuals = predict(X, theta) - y
    return (1 / m) * X.T @ residuals
