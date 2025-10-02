## File in which we do the functions
import numpy as np


def MSE_loss(y, tx, weights):
    return np.sum(np.square(y - tx @ weights)) / (2 * y.shape[0])


def gradient(y, tx, weights):
    return (- tx.T @ (y - tx @ weights)) / y.shape[0]

def stochastic_gradient(y, tx, w, batch_size=1):
    #batch making
    n = y.shape[0]
    indices = np.random.choice(n, batch_size, replace=False)
    y_batch = y[indices]
    tx_batch = tx[indices]
    
    #compute gradient 
    grad = gradient(y_batch, tx_batch, w)
    return grad


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for i in range(max_iters):
        loss = MSE_loss(y, tx, w)
        grad = gradient(y, tx, w)
        w = w - gamma * grad
        print(i, loss)

    return w, MSE_loss(y, tx, w)

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    batch_size = 1
    loss = 10
    for i in range(max_iters):
        grad = stochastic_gradient(y, tx, w, batch_size=batch_size)
        w = w - gamma * grad
        print(i)

    return w, loss


def least_squares(y, tx):
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = MSE_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    N = y.shape[0]
    D = tx.shape[1]
    lambda_prime = 2 * N * lambda_
    w = np.linalg.solve(tx.T @ tx + lambda_prime * np.eye(D), tx.T @ y)
    return w

