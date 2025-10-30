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
        case "least squares":
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
    y_pred_binary = np.where(y_pred < threshold, 0, 1)
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
        y_pred = np.where(p_te < thr,0,1)
        f1 = f1_score(y_te, y_pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    best_f1 = f1_score(y_te,y_pred)
    print("Best threshold:", best_thr, "Best F1:", best_f1)

    # Binary predictions
    y_tr_pred = np.where(p_tr < best_thr,0,1)
    y_te_pred = np.where(p_te < best_thr,0,1)

    acc_tr = compute_accuracy(y_tr, p_tr)
    acc_te = compute_accuracy(y_te, p_te)

    f1_tr = f1_score(y_tr, y_tr_pred)
    f1_te = f1_score(y_te, y_te_pred)

    return acc_tr, acc_te, f1_tr, f1_te


def preprocess_columns(x_tr, x_te):

    x_tr, x_te = replace_column_with(x_tr, x_te, [7], [1100], 1)
    x_tr, x_te = replace_column_with(x_tr, x_te, [7], [2200], 2)
    x_tr, x_te = replace_column_with(x_tr, x_te, [32], [3], 0)
    x_tr, x_te = replace_column_with(x_tr, x_te, [35], [4], 1.5)
    x_tr, x_te = replace_column_with(x_tr, x_te, [49], [4], 2)
    x_tr, x_te = replace_column_with(x_tr, x_te, [119], [8], 1)
    x_tr, x_te = replace_column_with(x_tr, x_te, [228], [9], 3)
    x_tr, x_te = replace_column_with(x_tr, x_te, [232], [9], 2)
    x_tr, x_te = replace_column_with(x_tr, x_te, [146], [97], 5)
    x_tr, x_te = replace_column_with(x_tr, x_te, [196,198], [98], 0)
    x_tr, x_te = replace_column_with(x_tr, x_te, [134,183],
                                     [7,9,np.nan], 3)
    x_tr, x_te = replace_column_with(x_tr, x_te, [135,136,147],
                                     [7,np.nan], 2)
    x_tr, x_te = replace_column_with(x_tr, x_te, [202],
                                     [7,9,np.nan], 7)
    x_tr, x_te = replace_column_with(x_tr, x_te, [172,174,176,179,181,184,189],
                                     [7,9,np.nan], 6)
    x_tr, x_te = replace_column_with(x_tr, x_te, [138,139,141],
                                     [7,9,np.nan], 5)
    x_tr, x_te = replace_column_with(x_tr, x_te, [124,125,158,161,162,164,165,166,171,173,175,177,178,180,182,
                                                  185,186,187,188,204],
                                     [7,9,np.nan], 2)
    x_tr, x_te = replace_column_with(x_tr, x_te, [121,122,156,157,201,],
                                     [7,9,np.nan], 0)
    x_tr, x_te = replace_column_with(x_tr, x_te, [148,149,150],
                                     [88,98,99,np.nan], 0)
    x_tr, x_te = replace_column_with(x_tr, x_te, [169],
                                     [77,99,np.nan], 0)
    x_tr, x_te = replace_column_with(x_tr, x_te, [151],
                                     [888,np.nan], 0)
    x_tr, x_te = replace_column_with(x_tr, x_te, [152,153,154,155],
                                     [8,np.nan], 0)
    x_tr, x_te = replace_column_with(x_tr, x_te, [116],
                                     [np.nan], 8)
    x_tr, x_te = replace_column_with(x_tr, x_te, [219],
                                     [np.nan], 6)
    x_tr, x_te = replace_column_with(x_tr, x_te, [38],
                                     [np.nan], 5)
    x_tr, x_te = replace_column_with(x_tr, x_te, [167],
                                     [np.nan], 4)
    x_tr, x_te = replace_column_with(x_tr, x_te, [74,98,109],
                                     [np.nan], 3)
    x_tr, x_te = replace_column_with(x_tr, x_te, [14,36,39,42,65,96,97,108,110,117,118],
                                     [np.nan], 2)
    x_tr, x_te = replace_column_with(x_tr, x_te, [25,57],
                                     [np.nan], 1)
    x_tr, x_te = replace_column_with(x_tr, x_te, [30,75,76,79,80,81,99,113,114,115,163],
                                     [np.nan], 0)
    x_tr, x_te = replace_column_with(x_tr, x_te, [28,29,30,60,80,113,114,115,207,208,209,210,211,212,213,214],
                                     [88], 0)
    x_tr, x_te = replace_column_with(x_tr, x_te, [193,194], [8], np.nan)
    x_tr, x_te = replace_column_with(x_tr, x_te, [247], [14], np.nan)
    x_tr, x_te = replace_column_with(x_tr, x_te, [248], [3], np.nan)
    x_tr, x_te = replace_column_with(x_tr, x_te, [89,113,114,115], [88], np.nan)
    x_tr, x_te = replace_column_with(x_tr, x_te, [50,146], [98,99], np.nan)
    x_tr, x_te = replace_column_with(x_tr, x_te, [196,198], [97], np.nan)
    x_tr, x_te = replace_column_with(x_tr, x_te, [60,196,198], [99], np.nan)
    x_tr, x_te = replace_column_with(x_tr, x_te, [263], [900], np.nan)
    x_tr, x_te = replace_column_with(x_tr, x_te, [265,288,289,294,295,298], [99900], np.nan)
    x_tr, x_te = replace_column_with(x_tr, x_te, [253], [99999], np.nan)
    x_tr, x_te = replace_column_with(x_tr, x_te, [52,53,59,231,233,234,235,236,237,238,242,243,244,245,256,257,
                                                  258,259,260,261,262,264,266,279,280,285,299,306,307,308,309,310,311,
                                                  312,313,314,315,316,317,318,319,320,321],
                                     [9], np.nan)
    x_tr, x_te = replace_column_with(x_tr, x_te, [25,27,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
                                                  49,54,55,56,57,58,62,65,66,67,68,69,70,71,72,73,74,75,77,88,96,97,98,
                                                  99,100,101,104,105,108,109,110,116,117,118,119,127,128,129,130,132,133,
                                                  137,140,142,143,145,152,153,154,155,159,160,163,167,168,170,193,194,
                                                  199,200,203,205,206,215,216,224],
                                     [7,9], np.nan)
    x_tr, x_te = replace_column_with(x_tr, x_te, [26,28,29,30,61,76,79,80,81,89,113,114,115,131,207,208,209,210,
                                                  211,212,213,214,225,226],
                                     [77,99], np.nan)
    x_tr, x_te = replace_column_with(x_tr, x_te, [151,240,241],
                                     [777,999], np.nan)

    for column in [78,90,93,95]:
        itemindex1 = np.where((x_tr[:, column] >= 101) & (x_tr[:, column] <= 199))
        itemindex2 = np.where((x_te[:, column] >= 101) & (x_te[:, column] <= 199))
        x_tr[itemindex1, column] = (x_tr[itemindex1, column] - 100) * 4
        x_te[itemindex2, column] = (x_te[itemindex2, column] - 100) * 4
        itemindex1 = np.where((x_tr[:, column] >= 201) & (x_tr[:, column] <= 299))
        itemindex2 = np.where((x_te[:, column] >= 201) & (x_te[:, column] <= 299))
        x_tr[itemindex1, column] = x_tr[itemindex1, column] - 200
        x_te[itemindex2, column] = x_te[itemindex2, column] - 200
        itemindex1 = np.where((x_tr[:, column] == 888))
        itemindex2 = np.where((x_te[:, column] == 888))
        x_tr[itemindex1, column] = 0
        x_te[itemindex2, column] = 0
        itemindex1 = np.where((x_tr[:, column] == 777) | (x_tr[:, column] == 999))
        itemindex2 = np.where((x_te[:, column] == 777) | (x_te[:, column] == 999))
        x_tr[itemindex1, column] = np.nan
        x_te[itemindex2, column] = np.nan

    for column in [82,83,84,85,86,87,111,112]:
        itemindex1 = np.where((x_tr[:, column] >= 101) & (x_tr[:, column] <= 199))
        itemindex2 = np.where((x_te[:, column] >= 101) & (x_te[:, column] <= 199))
        x_tr[itemindex1, column] = (x_tr[itemindex1, column] - 100) * 30
        x_te[itemindex2, column] = (x_te[itemindex2, column] - 100) * 30
        itemindex1 = np.where((x_tr[:, column] >= 201) & (x_tr[:, column] <= 299))
        itemindex2 = np.where((x_te[:, column] >= 201) & (x_te[:, column] <= 299))
        x_tr[itemindex1, column] = (x_tr[itemindex1, column] - 200) * 4
        x_te[itemindex2, column] = (x_te[itemindex2, column] - 200) * 4
        itemindex1 = np.where((x_tr[:, column] >= 301) & (x_tr[:, column] <= 399))
        itemindex2 = np.where((x_te[:, column] >= 301) & (x_te[:, column] <= 399))
        x_tr[itemindex1, column] = x_tr[itemindex1, column] - 300
        x_te[itemindex2, column] = x_te[itemindex2, column] - 300
        itemindex1 = np.where((x_tr[:, column] == 300))
        itemindex2 = np.where((x_te[:, column] == 300))
        x_tr[itemindex1, column] = 0.5
        x_te[itemindex2, column] = 0.5
        itemindex1 = np.where((x_tr[:, column] == 555) | (x_tr[:, column] == 888))
        itemindex2 = np.where((x_te[:, column] == 555) | (x_te[:, column] == 888))
        x_tr[itemindex1, column] = 0
        x_te[itemindex2, column] = 0
        itemindex1 = np.where((x_tr[:, column] == 777) | (x_tr[:, column] == 999))
        itemindex2 = np.where((x_te[:, column] == 777) | (x_te[:, column] == 999))
        x_tr[itemindex1, column] = np.nan
        x_te[itemindex2, column] = np.nan

    for column in [91,94]:
        itemindex1 = np.where((x_tr[:, column] >= 1) & (x_tr[:, column] <= 759))
        itemindex2 = np.where((x_te[:, column] >= 1) & (x_te[:, column] <= 759))
        x_tr[itemindex1, column] = np.floor(x_tr[itemindex1, column] / 100) * 60 + x_tr[itemindex1, column] % 100
        x_te[itemindex2, column] = np.floor(x_te[itemindex2, column] / 100) * 60 + x_te[itemindex2, column] % 100
        itemindex1 = np.where((x_tr[:, column] >= 800) & (x_tr[:, column] <= 959))
        itemindex2 = np.where((x_te[:, column] >= 800) & (x_te[:, column] <= 959))
        x_tr[itemindex1, column] = np.floor(x_tr[itemindex1, column] / 100) * 60 + x_tr[itemindex1, column] % 100
        x_te[itemindex2, column] = np.floor(x_te[itemindex2, column] / 100) * 60 + x_te[itemindex2, column] % 100
        itemindex1 = np.where((x_tr[:, column] == 777) | (x_tr[:, column] == 999))
        itemindex2 = np.where((x_te[:, column] == 777) | (x_te[:, column] == 999))
        x_tr[itemindex1, column] = np.nan
        x_te[itemindex2, column] = np.nan

    for column in [111,112]:
        itemindex1 = np.where((x_tr[:, column] >= 401) & (x_tr[:, column] <= 499))
        itemindex2 = np.where((x_te[:, column] >= 401) & (x_te[:, column] <= 499))
        x_tr[itemindex1, column] = (x_tr[itemindex1, column] - 400) / 12
        x_te[itemindex2, column] = (x_te[itemindex2, column] - 400) / 12

    for column in [144]:
        itemindex1 = np.where((x_tr[:, column] >= 101) & (x_tr[:, column] <= 199))
        itemindex2 = np.where((x_te[:, column] >= 101) & (x_te[:, column] <= 199))
        x_tr[itemindex1, column] = (x_tr[itemindex1, column] - 100)
        x_te[itemindex2, column] = (x_te[itemindex2, column] - 100)
        itemindex1 = np.where((x_tr[:, column] >= 201) & (x_tr[:, column] <= 299))
        itemindex2 = np.where((x_te[:, column] >= 201) & (x_te[:, column] <= 299))
        x_tr[itemindex1, column] = (x_tr[itemindex1, column] - 200) * 7
        x_te[itemindex2, column] = (x_te[itemindex2, column] - 200) * 7
        itemindex1 = np.where((x_tr[:, column] >= 301) & (x_tr[:, column] <= 399))
        itemindex2 = np.where((x_te[:, column] >= 301) & (x_te[:, column] <= 399))
        x_tr[itemindex1, column] = (x_tr[itemindex1, column] - 300) * 30
        x_te[itemindex2, column] = (x_te[itemindex2, column] - 300) * 30
        itemindex1 = np.where((x_tr[:, column] >= 401) & (x_tr[:, column] <= 499))
        itemindex2 = np.where((x_te[:, column] >= 401) & (x_te[:, column] <= 499))
        x_tr[itemindex1, column] = (x_tr[itemindex1, column] - 400) * 365
        x_te[itemindex2, column] = (x_te[itemindex2, column] - 400) * 365
        itemindex1 = np.where((x_tr[:, column] == 777) | (x_tr[:, column] == 999))
        itemindex2 = np.where((x_te[:, column] == 777) | (x_te[:, column] == 999))
        x_tr[itemindex1, column] = np.nan
        x_te[itemindex2, column] = np.nan

    col_mean = np.nanmean(x_tr, axis=0)
    inds = np.where(np.isnan(x_tr))
    x_tr[inds] = np.take(col_mean, inds[1])
    inds = np.where(np.isnan(x_te))
    x_te[inds] = np.take(col_mean, inds[1])

    for col in sorted([230,229,227,223,222,221,220,192,191,106,102,64,63,23,20,19,13,12,10,9,8,3])[::-1]:
        x_tr = np.delete(x_tr, col, 1)
        x_te = np.delete(x_te, col, 1)

    encoded_parts = []
    cols_to_encode = [1,43,50,78,81,91,94,107,110,113,177,180,182,202]

    for i in range(x_tr.shape[1]):
        if i in cols_to_encode:
            col = x_tr[:, i]
            unique_vals = np.unique(col)
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            indices = np.vectorize(mapping.get)(col)
            one_hot = np.eye(len(unique_vals))[indices]
            encoded_parts.append(one_hot)
        else:
            encoded_parts.append(x_tr[:, [i]])

    encoded_x_tr = np.hstack(encoded_parts)

    encoded_parts = []

    for i in range(x_te.shape[1]):
        if i in cols_to_encode:
            col = x_te[:, i]
            unique_vals = np.unique(col)
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            indices = np.vectorize(mapping.get)(col)
            one_hot = np.eye(len(unique_vals))[indices]
            encoded_parts.append(one_hot)
        else:
            encoded_parts.append(x_te[:, [i]])

    encoded_x_te = np.hstack(encoded_parts)


    return encoded_x_tr, encoded_x_te

def replace_column_with(array1, array2, columns, values_to_replace, value_to_replace_with):
    for column in columns:
        for value_to_replace in values_to_replace:
            if np.isnan(value_to_replace):
                itemindex1 = np.where(np.isnan(array1[:, column]))
                itemindex2 = np.where(np.isnan(array2[:, column]))
            else:
                itemindex1 = np.where((array1[:, column] == value_to_replace))
                itemindex2 = np.where((array2[:, column] == value_to_replace))
                
            array1[itemindex1, column] = value_to_replace_with
            array2[itemindex2, column] = value_to_replace_with
    return array1,array2
