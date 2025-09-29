## File in which we do the functions
import helpers
import numpy as np
import os
import timeit


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
    loss = np.sum(np.square(y_batch - tx_batch @ w)) / (2 * batch_size)
    grad = -(tx_batch.T @ (y_batch - tx_batch @ w)) / batch_size
    return grad, loss


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
    batch_size = 10
    for i in range(max_iters):
        grad, loss = stochastic_gradient(y, tx, w, batch_size=batch_size)
        w = w - gamma * grad
        print(i, loss)

    return w,loss



if __name__ == '__main__':
    start = timeit.default_timer()

    # Loading data, once its done no need to do it again
    # x_train, x_test, y_train, train_ids, test_ids = helpers.load_csv_data(os.getcwd() + "\\data\\dataset", sub_sample=True)
    # x_train = np.c_[np.ones(x_train.shape[0]), x_train]
    # np.save("x_train_preprocessed.npy", x_train)
    # np.save("y_train_preprocessed.npy", y_train)
    # np.save("x_test_preprocessed.npy", x_train)

    #Loading data so that it is faster
    x_train = np.load("x_train_preprocessed.npy")
    y_train = np.load("y_train_preprocessed.npy")
    x_test = np.load("x_test_preprocessed.npy")
 
    # replaces nan values with mean of column
    col_mean = np.nanmean(x_train, axis=0)
    inds = np.where(np.isnan(x_train))
    x_train[inds] = np.take(col_mean, inds[1])
    
    #Normalization 
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std[std == 0] = 1.0 # Replace zeros with 1.0 to avoid division by zero

    x_train = (x_train - mean) / std

    initial_w = np.random.randn(x_train.shape[1])

    #w, final_loss  = mean_squared_error_gd(y_train, x_train, initial_w, max_iters=500, gamma=0.01)
    w, final_loss  = mean_squared_error_sgd(y_train, x_train, initial_w, max_iters=100, gamma=0.1)

    print(timeit.default_timer() - start)
