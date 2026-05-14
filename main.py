import sys
sys.path.append("src")

from src.data import load_and_prepare
from src.model import predict
from src.train import gradient_descent, evaluate
from src.utils import plot_cost, plot_predictions, rmse, r_squared
from sklearn.linear_model import LinearRegression

def sklearn_model(X_train_n, X_val_n, y_train_n, y_val, y_mu, y_sigma):
    model = LinearRegression()
    model.fit(X_train_n, y_train_n)
    y_val_pred = model.predict(X_val_n) * y_sigma + y_mu
    print(f"\nsklearn:")
    print(f'Val RMSE: ${rmse(y_val, y_val_pred): ,.0f}')
    print(f"Val R²:   {r_squared(y_val, y_val_pred):.4f}")

def main():
    X_train, X_val,X_train_n,X_val_n, y_train, y_val, y_train_n, y_mu, y_sigma = load_and_prepare("data/train.csv")

    theta, cost_history = gradient_descent(X_train, y_train_n, alpha=0.1, num_iters=1000)

    plot_cost(cost_history)

    y_val_pred = evaluate(X_train, X_val, y_train, y_val, theta, y_mu, y_sigma)

    plot_predictions(y_val, y_val_pred)
    
    sklearn_model(X_train_n, X_val_n, y_train_n, y_val, y_mu, y_sigma)

if __name__ == "__main__":
    main()
