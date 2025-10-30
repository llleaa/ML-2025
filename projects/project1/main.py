import helpers
import numpy as np
import os
import timeit
import implementations as imp
import matplotlib.pyplot as plt

if __name__ == '__main__':
    start = timeit.default_timer()

    # Loading data, once its done no need to do it again
    # x_train, x_test, y_train, train_ids, test_ids = helpers.load_csv_data(os.getcwd() + "\\data\\dataset", sub_sample=False)
    # x_train = np.c_[np.ones(x_train.shape[0]), x_train]
    # x_test = np.c_[np.ones(x_test.shape[0]), x_test]
    # np.save("x_train_preprocessed.npy", x_train)
    # np.save("y_train_preprocessed.npy", y_train)
    # np.save("x_test_preprocessed.npy", x_test)
    # np.save("train_ids.npy", train_ids)
    # np.save("test_ids.npy", test_ids)


    # Loading data so that it is faster
    x_train = np.load("x_train_processed2.npy")
    y_train = np.load("y_train_processed2.npy")
    x_test = np.load("x_test_processed2.npy")

    # x_train,x_test = imp.preprocess_columns(x_train,x_test)
    # np.save("x_train_processed2.npy", x_train)
    # np.save("y_train_processed2.npy", y_train)
    # np.save("x_test_processed2.npy", x_test)
    # print("prepocessed")

    # Loading data so that it is faster
    # x_train = np.load("x_train_preprocessed.npy")
    # y_train = np.load("y_train_preprocessed.npy")
    # x_test = np.load("x_test_preprocessed.npy")
    y_train = np.where(y_train == -1, 0, 1)
    # print(np.unique(y_train))
    ids = np.load("test_ids.npy")
    

    # replaces nan values with mean of column
    # col_mean = np.nanmean(x_train, axis=0)
    # inds = np.where(np.isnan(x_train))
    # x_train[inds] = np.take(col_mean, inds[1])
    # inds = np.where(np.isnan(x_test))
    # x_test[inds] = np.take(col_mean, inds[1])


    # Normalization
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std[std == 0] = 1.0 # Replace zeros with 1.0 to avoid division by zero

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std


    max_iters = 500
    lambda_ = 1e-1
    method_name = "ridge regression"
    gamma = 1

    initial_w = np.random.randn(x_train.shape[1])
    
    # acc_lambda = []
    # f1_lambda = []
    # lambda_t =[1e-5,9e-6,8e-6,6e-6,3e-6]
    # for lambda_ in lambda_t:
    #     print(f"Lambda : {lambda_}")
    #     loss_tr_avg, loss_te_avg, acc_tr_avg, acc_te_avg, f1_tr_avg, f1_te_avg, _, _, _, _ = imp.cross_validation(y_train,x_train,k_fold=5,lambda_=lambda_,function_name=method_name,initial_w= initial_w,max_iters=max_iters,gamma=gamma)
        # acc_lambda.append(acc_te_avg)
        # f1_lambda.append(f1_te_avg)
    #     print(f" Logistic regression / gamma : {gamma} / test loss : {loss_te} / test accuracy : {acc_te_avg} / test f1 : {f1_te_avg} ")
    # print(f"test loss : {loss_te} / test accuracy : {acc_te_avg} / test f1 : {f1_te_avg} ")
    #plt.plot(lambda_t, f1_lambda)
    #plt.plot(lambda_t, f1_lambda)
    #plt.show()
    #imp.cross_validation(y_train,x_train,k_fold=5,lambda_=lambda_,function_name=method_name,initial_w= initial_w,max_iters=max_iters,gamma=1e-3)
    w, loss = imp.ridge_regression(y_train,x_train,8e-6)
    
    # #Prediction 
    y_test = imp.sigmoid(x_test @ w)

    threshold = 0.531
    #y_te_pred = np.where(y_te_pred < threshold, -1, 1)
    y_test = np.where(y_test < threshold, -1, 1)
    unique, counts = np.unique(y_test, return_counts=True)

    print(dict(zip(unique, counts)))

    # print(np.sum(y_te_pred == y_te))
    # print("Should be accuracy", np.sum(y_te_pred == y_te) / len(y_te))
    # print("Loss train :", loss_tr)

    #Submission 
    helpers.create_csv_submission(ids, y_test, "ridge_reg_8e-6")

    print("Time :" , timeit.default_timer() - start)
