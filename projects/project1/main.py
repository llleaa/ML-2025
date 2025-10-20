import helpers
import numpy as np
import os
import timeit
import implementations as imp

if __name__ == '__main__':
    start = timeit.default_timer()

    # Loading data, once its done no need to do it again
    # x_train, x_test, y_train, train_ids, test_ids = helpers.load_csv_data(os.getcwd() + "\\data\\dataset", sub_sample=True)
    # x_train = np.c_[np.ones(x_train.shape[0]), x_train]
    # np.save("x_train_preprocessed.npy", x_train)
    # np.save("y_train_preprocessed.npy", y_train)
    # np.save("x_test_preprocessed.npy", x_train)

    # Loading data so that it is faster
    x_train = np.load("x_train_preprocessed.npy")
    y_train = np.load("y_train_preprocessed.npy")
    x_test = np.load("x_test_preprocessed.npy")
    y_train = np.where(y_train == -1, 0, 1)

    # replaces nan values with mean of column
    col_mean = np.nanmean(x_train, axis=0)
    inds = np.where(np.isnan(x_train))
    x_train[inds] = np.take(col_mean, inds[1])
    inds = np.where(np.isnan(x_test))
    x_test[inds] = np.take(col_mean, inds[1])

    #x_train = imp.build_poly(x_train, degree=3)

    # Normalization
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std[std == 0] = 1.0 # Replace zeros with 1.0 to avoid division by zero

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    gamma = 0.0775
    max_iters = 100


    initial_w = np.random.randn(x_train.shape[1])
    k_indices = imp.build_k_indices(y_train, 5, 20)
    k = 1
    train_indices = [i for i in range(k_indices.shape[0]) if i != k]


    #w, final_loss  = imp.least_squares(y_train, x_train)
    loss_tr, loss_te = imp.cross_validation(y_train,x_train,k_fold=5,lambda_=10,function_name="ridge regression",initial_w= initial_w,max_iters=1000,gamma=0.01, degree = 3)
    y_te, x_te = y_train[k_indices[k].flatten()], x_train[k_indices[k].flatten()]
    y_train, x_train = y_train[k_indices[train_indices].flatten()], x_train[k_indices[train_indices].flatten()]



    w, loss_tr = imp.logistic_regression_adam(y_train, x_train, initial_w, max_iters, gamma)

    y_te_pred = imp.sigmoid(x_te @ w)
    y_test = imp.sigmoid(x_test @ w)

    threshold = 0.5

    y_te_pred = np.where(y_te_pred < threshold, -1, 1)
    y_test = np.where(y_test < threshold, -1, 1)
    print("Should be accuracy", np.sum(y_te_pred == y_te) / len(y_test))
    print(y_test)

    #loss_tr, loss_te = imp.cross_validation(y_train,x_train,k_fold=5,lambda_=0,function_name="logistic regression adam",initial_w= initial_w,max_iters=7000,gamma=gamma)

    print("Gamma:",gamma)
    #print("Test loss :", loss_te)
    print("Training loss:", loss_tr)

    print("Time :" , timeit.default_timer() - start)
