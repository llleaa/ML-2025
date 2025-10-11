## File in which we do the functions
import numpy as np


def MSE_loss(y, tx, weights):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        weights: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters weights.
    """
    return np.sum(np.square(y - tx @ weights)) / (2 * y.shape[0])


def gradient(y, tx, weights):
    """Calculate the gradient of the MSE loss function.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        weights: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the gradient, corresponding to the input parameters weights.
    """
    return (- tx.T @ (y - tx @ weights)) / y.shape[0]

def stochastic_gradient(y, tx, w, batch_size=1):
    """Calculate the stochastic gradient of the MSE loss function.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        weights: numpy array of shape=(D,). The vector of model parameters.
        batch_size: int. The size of the batch, by default 1

    Returns:
        the value of the stochastic gradient, corresponding to the input parameters weights, and batch size.
    """
    
    #batch making
    n = y.shape[0]
    indices = np.random.choice(n, batch_size, replace=False)
    y_batch = y[indices]
    tx_batch = tx[indices]
    
    #compute gradient 
    grad = gradient(y_batch, tx_batch, w)
    return grad


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Perform gradient descent using the MSE loss function.

    Args:
        y: numpy array of shape=(N, ).
            The vector of target values.
        tx: numpy array of shape=(N, D).
            The matrix of input features.
        initial_w: numpy array of shape=(D, ).
            The initial weight vector.
        max_iters: int.
            The maximum number of iterations to run the gradient descent algorithm.
        gamma: float.
            The learning rate used to update the weights.

    Returns:
        w: numpy array of shape=(D, ).
            The optimized weight vector after performing gradient descent.
        loss: float.
            The final value of the MSE loss function corresponding to the optimized weights.
    """
    w = initial_w
    for i in range(max_iters):
        loss = MSE_loss(y, tx, w)
        grad = gradient(y, tx, w)
        w = w - gamma * grad
        print(i, loss)

    return w, MSE_loss(y, tx, w)

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Perform stochastic gradient descent using the MSE loss function.

    Args:
        y: numpy array of shape=(N, ).
            The vector of target values.
        tx: numpy array of shape=(N, D).
            The matrix of input features.
        initial_w: numpy array of shape=(D, ).
            The initial weight vector.
        max_iters: int.
            The maximum number of iterations to run the stochastic gradient descent algorithm.
        gamma: float.
            The learning rate used to update the weights.

    Returns:
        w: numpy array of shape=(D, ).
            The optimized weight vector after performing stochastic gradient descent.
        loss: float.
            The final value of the MSE loss function corresponding to the optimized weights.
    """
    w = initial_w
    batch_size = 1
    loss = 10
    for i in range(max_iters):
        grad = stochastic_gradient(y, tx, w, batch_size=batch_size)
        w = w - gamma * grad
        print(i)

    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """

    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = MSE_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """

    N = y.shape[0]
    D = tx.shape[1]
    lambda_prime = 2 * N * lambda_
    w = np.linalg.solve(tx.T @ tx + lambda_prime * np.eye(D), tx.T @ y)
    return w

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        sigmoid : scalar or numpy array
    """
    return 1/(1 + np.exp(-t))

def loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        loss : non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    return (-1/y.shape[0]) * np.sum(y*np.log(sigmoid(tx@w)) + (1-y)*np.log(1 - sigmoid(tx@w)))


def gradient_logistic(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        gradient : vector of shape (D, 1)

    """

    return (1/y.shape[0]) * tx.T@(sigmoid(tx@w) - y) 

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform max_iters steps of gradient descent using logistic regression.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        w: shape=(D, 1)
        loss: scalar number
    """
    
    w = initial_w
    for i in range(max_iters):
        loss = loss_logistic(y, tx, w)
        grad = gradient_logistic(y, tx, w)
        w = w - gamma * grad
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        gradient: shape=(D, 1)
        loss: scalar number
        
    """
    loss = loss_logistic(y, tx, w)
    gradient = gradient_logistic(y,tx,w) + 2*lambda_*w
    return gradient, loss


if __name__ == '__main__':
    #start = timeit.default_timer()

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
    w, final_loss  = mean_squared_error_sgd(y_train, x_train, initial_w, max_iters=1000000, gamma=0.0001)
    print("Training loss :", MSE_loss(y_train, x_train, w))

    #print("Time :",timeit.default_timer() - start)
