## File in which we do the functions
import copy

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
    
    indices = np.random.choice(y.shape[0], batch_size, replace=False)
    
    return gradient(y[indices], tx[indices], w)

def gradient_descent(y, tx, w, max_iters, gamma, gradient=gradient, loss = MSE_loss, lambda_ = 0):
    """
    
    """

    for i in range(max_iters):
        w = w - gamma *( gradient(y, tx, w) + lambda_ * w)
        # if i % 999 == 0:
        #     print(f"Iter no {i} loss : {loss(y, tx, w)}")
    return w

def stochastic_gradient_descent(y, tx, w, max_iters, gamma, batch_size, gradient=gradient, loss = MSE_loss, lambda_ = 0):
    """
    
    """
    
    for i in range(max_iters):
        w = w - gamma * (gradient(y, tx, w, batch_size=batch_size) + lambda_ * w)
        # if i % 1000 == 0:
        #     print(f"Iter no {i} loss : {loss(y, tx, w)}")

    return w

def adam(y, tx, w, max_iters, gamma, beta1 = 0.9, beta2 = 0.999, gradient=gradient, loss=MSE_loss):

    previous_ema = 0
    previous_ema_sq = 0
    ema = 0
    ema_sq = 0

    for i in range(max_iters):
        grad = gradient(y, tx, w)
        ema = beta1 * previous_ema + (1 - beta1) * grad
        ema_sq = beta2 * previous_ema_sq + (1 - beta2) * (grad ** 2)
        delta = ema / (np.sqrt(ema_sq) + 1e-8)
        w = w - gamma * delta
        previous_ema = ema
        previous_ema_sq = ema_sq
        
        # if i % 1000 == 0:
        #     print(f"Iter no {i} loss : {loss(y, tx, w)}")

    return w


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
    w = gradient_descent(y, tx, initial_w, max_iters, gamma, gradient = gradient)

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

    w = stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, batch_size=1, gradient=gradient, loss=MSE_loss)

    return w, MSE_loss(y, tx, w)

def mean_squared_error_adam(y, tx, initial_w, max_iters, gamma):
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

    w = adam(y, tx, initial_w, max_iters, gamma, gradient=gradient, loss=MSE_loss)

    return w, MSE_loss(y, tx, w)


def least_squares(y, tx):
    """Calculate the least squares solution.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    w = np.linalg.lstsq(tx.T @ tx, tx.T @ y)[0]
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

    w = np.linalg.solve(tx.T @ tx + 2 * y.shape[0] * lambda_ * np.eye(tx.shape[1]), tx.T @ y)
    
    return w, MSE_loss(y, tx, w)

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        sigmoid : scalar or numpy array
    """

    return clamp_between_0_and_1(1/(1 + np.exp(-t)))

def clamp_between_0_and_1(t):
    return np.maximum(np.minimum(t, 1-1e-12), 1e-12)

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

    return -np.sum(y*np.log(sigmoid(tx@w)) + (1-y)*np.log(1 - sigmoid(tx@w))) / y.shape[0]


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
    w = gradient_descent(y, tx, initial_w, max_iters, gamma, gradient=gradient_logistic, loss=loss_logistic)

    return w, loss_logistic(y, tx, w)


def logistic_regression_adam(y, tx, initial_w, max_iters, gamma):
    """
    Perform max_iters steps of adam using logistic regression.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        w: shape=(D, 1)
        loss: scalar number
    """
    w = adam(y, tx, initial_w, max_iters, gamma, gradient=gradient_logistic, loss=loss_logistic)

    return w, loss_logistic(y, tx, w)


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
    w = gradient_descent(y, tx, initial_w, max_iters, gamma, gradient=gradient_logistic, loss=loss_logistic, lambda_=lambda_)

    return w, loss_logistic(y, tx, w)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    """
    # for j in range(degree + 1):
    #     if j == 0:
    #         poly = np.ones((x.shape[0], 1)) # x.shape[0] is N (number of samples)
    #     else:
    #         poly = np.c_[poly, x**j]
    # return poly
    
    array = np.array([x**i for i in range(0,degree + 1)])
    return np.reshape(np.transpose(array, (1,2,0)), (array.shape[1], -1))


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_one_step(y, x, k_indices, k, lambda_, function_name, initial_w, max_iters, gamma):
    """return the loss of ridge regression for a fold corresponding to k_indices (one step)

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    """

    train_indices = [i for i in range(k_indices.shape[0]) if i != k]
    y_tr, x_tr = y[k_indices[train_indices].flatten()], x[k_indices[train_indices].flatten()]
    y_te, x_te = y[k_indices[k].flatten()], x[k_indices[k].flatten()]


    match function_name :
        case "mean sqrt":
           w, loss_tr = mean_squared_error_gd(y_tr, x_tr, initial_w, max_iters, gamma)
           loss_te = MSE_loss(y_te,x_te,w)
           return np.sqrt(2 * loss_tr), np.sqrt(2*loss_te), f1_and_accuracy(x_tr, x_te, y_tr,y_te,w)
        case "mean sqrt sgd":
            w, loss_tr = mean_squared_error_sgd(y_tr, x_tr, initial_w, max_iters, gamma)
            loss_te = MSE_loss(y_te,x_te,w)
            return np.sqrt(2 * loss_tr), np.sqrt(2*loss_te),f1_and_accuracy(x_tr, x_te, y_tr,y_te,w)
        case "least square":
            w, loss_tr = least_squares(y_tr, x_tr)
            loss_te = MSE_loss(y_te, x_te, w)
            return np.sqrt(2 * loss_tr), np.sqrt(2*loss_te),f1_and_accuracy(x_tr, x_te, y_tr,y_te,w)
        case "ridge regression":
            w, loss_tr = ridge_regression(y_tr, x_tr, lambda_)
            loss_te = MSE_loss(y_te, x_te, w)
            return np.sqrt(2 * loss_tr), np.sqrt(2*loss_te),f1_and_accuracy(x_tr, x_te, y_tr,y_te,w)
        case "logistic regression":
            w, loss_tr = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)
            loss_te = loss_logistic(y_te,x_te,w)
            return np.sqrt(2 * loss_tr), np.sqrt(2*loss_te),f1_and_accuracy(x_tr, x_te, y_tr,y_te,w)
        case "reg logistic regression":
            w, loss_tr = reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
            loss_te = loss_logistic(y_te, x_te, w)
            return np.sqrt(2 * loss_tr), np.sqrt(2*loss_te),f1_and_accuracy(x_tr, x_te, y_tr,y_te,w)
        case "logistic regression adam":
            w, loss_tr = logistic_regression_adam(y_tr, x_tr, initial_w, max_iters, gamma)
            loss_te = loss_logistic(y_te,x_te,w)
            return np.sqrt(2 * loss_tr), np.sqrt(2*loss_te),f1_and_accuracy(x_tr, x_te, y_tr,y_te,w)

        case "mean sqrt adam":
            w, loss_tr = mean_squared_error_adam(y_tr, x_tr, initial_w, max_iters, gamma)
            loss_te = MSE_loss(y_te, x_te, w)
            return np.sqrt(2 * loss_tr), np.sqrt(2*loss_te),f1_and_accuracy(x_tr, x_te, y_tr,y_te,w)
        
        case _ :
            raise ValueError("Function not recognized, choose between: mean sqrt, mean sqrt sgd, " \
            "least square, ridge regression, logistic regression, reg logistic regression")
        
def cross_validation(y, x,k_fold,lambda_, function_name, initial_w, max_iters, gamma):
    seed = 20
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    loss_tr_avg = 0
    loss_te_avg = 0
    acc_tr_avg = 0
    acc_te_avg = 0
    f1_tr_avg = 0 
    f1_te_avg = 0

    for k in range(k_fold):
        loss_tr, loss_te, f1_and_acc  = cross_validation_one_step(y,x,k_indices,k,lambda_,function_name,copy.deepcopy(initial_w),max_iters,gamma)
        loss_tr_avg += loss_tr
        loss_te_avg += loss_te
        acc_tr_avg += f1_and_acc[0]
        acc_te_avg += f1_and_acc[1]
        f1_tr_avg += f1_and_acc[2]
        f1_te_avg += f1_and_acc[3]
        #print(f"Test loss for trial no {k} : {loss_te}")
    loss_tr_avg /= k_fold
    loss_te_avg /= k_fold
    acc_tr_avg /= k_fold
    acc_te_avg /= k_fold
    f1_tr_avg /= k_fold
    f1_te_avg /= k_fold
    return loss_tr_avg, loss_te_avg, acc_tr_avg, acc_te_avg, f1_tr_avg, f1_te_avg

def compute_accuracy(y_true, y_pred, threshold=0.5):
    """Compute accuracy for binary classification."""
    y_pred_binary = np.where(y_pred >= threshold, 0, 1)
    return np.mean(y_pred_binary == y_true)
    
def f1_score(y_true, y_pred_binary):
    """Compute F1 score using.
    
    y_true: (N,) array of true binary labels {0,1}
    y_pred_binary: (N,) array of predicted binary labels {0,1}
    """
    y_true = y_true.flatten()
    y_pred_binary = y_pred_binary.flatten()
    
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))

    if tp + fp == 0 or tp + fn == 0:
        return 0.0  

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1

def f1_and_accuracy(x_tr, x_te, y_tr,y_te,w):
    p_tr = sigmoid(x_tr @ w)
    p_te = sigmoid(x_te @ w)

    best_f1, best_thr = 0, 0

    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = np.where(p_te >= thr,0,1)
        f1 = f1_score(y_te, y_pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print("Best threshold:", best_thr, "Best F1:", best_f1)

    # Binary predictions
    y_tr_pred = np.where(p_tr >= best_thr,0,1)
    y_te_pred = np.where(p_te >= best_thr,0,1)

    acc_tr = compute_accuracy(y_tr, p_tr)
    acc_te = compute_accuracy(y_te, p_te)

    f1_tr = f1_score(y_tr, y_tr_pred)
    f1_te = f1_score(y_te, y_te_pred)

    return acc_tr, acc_te, f1_tr, f1_te



