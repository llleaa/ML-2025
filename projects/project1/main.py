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

    # replaces nan values with mean of column
    col_mean = np.nanmean(x_train, axis=0)
    inds = np.where(np.isnan(x_train))
    x_train[inds] = np.take(col_mean, inds[1])

    x_train = imp.build_poly(x_train, degree=3)

    # Normalization
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std[std == 0] = 1.0 # Replace zeros with 1.0 to avoid division by zero

    x_train = (x_train - mean) / std


    initial_w = np.random.randn(x_train.shape[1])

    #w, final_loss  = imp.mean_squared_error_gd(y_train, x_train, initial_w, max_iters=500, gamma=0.01)
    loss_tr, loss_te = imp.cross_validation(y_train,x_train,k_fold=100,lambda_=10,function_name="mean sqrt",initial_w= initial_w,max_iters=1000,gamma=0.01, degree = 3)

    print("Test loss :", loss_te)
    print("Training loss:", loss_tr)

    print("Time :" , timeit.default_timer() - start)
