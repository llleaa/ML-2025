import helpers
import numpy as np
import os
import timeit
import implementations as imp
import matplotlib.pyplot as plt

if __name__ == '__main__':
    start = timeit.default_timer()

    # Loading data, once its done no need to do it again
    #x_train, x_test, y_train, train_ids, test_ids = helpers.load_csv_data(os.getcwd() + "\\data\\dataset", sub_sample=True)
    #x_train = np.c_[np.ones(x_train.shape[0]), x_train]
    # np.save("x_train_preprocessed_sub_sub.npy", x_train)
    # np.save("y_train_preprocessed_sub.npy", y_train)
    # np.save("x_test_preprocessed_sub.npy", x_test)
    # np.save("train_ids.npy", train_ids)
    # np.save("test_ids.npy", test_ids)


    # Loading data so that it is faster
    x_train = np.load("x_train_preprocessed_sub.npy")
    y_train = np.load("y_train_preprocessed_sub.npy")
    x_test = np.load("x_test_preprocessed_sub.npy")
    y_train = np.where(y_train == -1, 0, 1)
    # ids = np.load("test_ids.npy")

    # replaces nan values with mean of column
    col_mean = np.nanmean(x_train, axis=0)
    inds = np.where(np.isnan(x_train))
    x_train[inds] = np.take(col_mean, inds[1])
    inds = np.where(np.isnan(x_test))
    x_test[inds] = np.take(col_mean, inds[1])

    for degree in [0,1,2,3]:

        x_train_poly = imp.build_poly(x_train, degree=degree)

        # Normalization
        mean = np.nanmean(x_train_poly, axis=0)
        std = np.nanstd(x_train_poly, axis=0)
        std[std == 0] = 1.0 # Replace zeros with 1.0 to avoid division by zero

        x_train_poly = (x_train_poly - mean) / std
        #x_test = (x_test - mean) / std


        initial_w = np.random.randn(x_train_poly.shape[1])
        
        for gamma in [1e-5, 1e-4, 1e-3, 1e-2,1e-1]:
            loss_tr, loss_te = imp.cross_validation(y_train,x_train_poly,k_fold=5,lambda_=0,function_name="logistic regression",initial_w= initial_w,max_iters=1000,gamma=gamma)
            print(f"Degree : {degree} / gamma : {gamma} / average training loss : {loss_tr} / average test loss : {loss_te}")

    # loss = []
    # gammas = [0.1,0.01]
    # iterations = np.arange(800,1100,100)
    # for iter in iterations:
    #     w, loss_tr = imp.logistic_regression(y_train, x_train, initial_w, iter, 0.1)
    #     loss.append(loss_tr)
    
    # #np.save("loss_adam.npy", loss)
    # #loss = np.load("loss_adam.npy")
    # plt.plot(iterations,loss)
    # plt.xticks(iterations)
    # plt.xscale('log')
    # plt.show()
    # y_te_pred = imp.sigmoid(x_te @ w)
    # y_test = imp.sigmoid(x_test @ w)

    # threshold = 0.2

    # y_te_pred = np.where(y_te_pred < threshold, -1, 1)
    # y_test = np.where(y_test < threshold, -1, 1)
    # print(np.sum(y_te_pred == y_te))
    # print("Should be accuracy", np.sum(y_te_pred == y_te) / len(y_te))
    #print("Loss train :", loss_tr)

    #loss_tr, loss_te = imp.cross_validation(y_train,x_train,k_fold=5,lambda_=0,function_name="logistic regression adam",initial_w= initial_w,max_iters=7000,gamma=gamma)

    # print("Gamma:",gamma)
    #print("Test loss :", loss_te)
    # print("Training loss:", loss_tr)
    # ids = np.arange(0,y_test.shape[0],1)
    #helpers.create_csv_submission(ids, y_test, "test_adam")

    print("Time :" , timeit.default_timer() - start)
