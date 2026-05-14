import numpy as np
import matplotlib.pyplot as plt 

def z_score_normalize(X_train, X_val):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma == 0, 1, sigma)
    
    X_train_n = (X_train - mu) / sigma
    X_val_n = (X_val -mu ) / sigma
    
    return X_train_n, X_val_n, mu, sigma

def add_bias(X): 
    m = X.shape[0]
    return np.c_[np.ones(m), X]

def rmse(y_true, y_pred):
    return  np.sqrt(np.mean((y_pred - y_true) ** 2))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

def plot_cost(cost_history):
    plt.figure()
    plt.plot(cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("J(θ)")
    plt.title("Cost vs Iterations")
    plt.grid(True)
    plt.savefig("outputs/cost_curve.png")
    plt.show()

def plot_predictions(y_val, y_val_pred):
    plt.figure()
    plt.scatter(y_val, y_val_pred, alpha=0.4, s=10)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Predicted vs Actual")
    plt.grid(True)
    plt.savefig("outputs/predictions.png")
    plt.show()