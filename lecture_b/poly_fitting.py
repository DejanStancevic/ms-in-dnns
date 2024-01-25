import numpy as np
import matplotlib.pyplot as plt



"""
Returns an output of a polynomial with input x and coeffs W
"""
def poly(x, W):

    x = np.array([ np.array([x_pt**i for i in range(len(W[-1]))]) for x_pt in x ])
    y = W@x.transpose()

    return y



"""
Add a function `fit_poly` to your script which performs polynomial regression **by implementing the closed-form solution for linear regression discussed in Lecture 2 in NumPy**. It should accept the following arguments
- A numpy array `x_train` of shape `(N,)` with the x-values of the training data
- A numpy array `y_train` of shape `(N,)` with the y-values of the training data
- An integer `k` for the order of the polynomial

where `N` is the number of training samples. This function should return a numpy array of shape `(1, k+1)` with the weights of the fitted polynomial.
"""
def fit_poly(x_train, y_train, k):
    
    x = np.array([ np.array([x_pt**i for i in range(k+1)]) for x_pt in x_train ])
    W = (y_train[None]@x) @ np.linalg.inv(x.transpose()@x)

    return W



"""
Write a function `mse_poly` which returns the mean squared error of a fit polynomial on test data and has the arguments
- A numpy array `x` of shape `(N,)` with the x-values of the test data
- A numpy array `y` of shape `(N,)` with the y-values of the test data
- A numpy array `W` of shape `(1, k)` with the weights of the polynomial
"""
def mse_poly(x, y, W):
    
    difference = y - poly(x, W)
    mse_error = (difference@difference.transpose()) / len(y)

    return mse_error[0, 0]



"""
Write a function `ridge_fit_poly` which performs ridge regression by implementing this closed-form solution. Your function should accept the following arguments
- A numpy array `x_train` of shape `(N,)` with the x-values of the training data
- A numpy array `y_train` of shape `(N,)` with the y-values of the training data
- An integer `k` for the order of the polynomial
- A float `lamb` for the regularization parameter
and return a numpy array of shape `(1, k+1)` with the weights of the fitted polynomial.
"""
def ridge_fit_poly(x_train, y_train, k, lamb):

    x = np.array([ np.array([x_pt**i for i in range(k+1)]) for x_pt in x_train ])
    W = (y_train[None]@x) @ np.linalg.inv( x.transpose()@x + lamb * np.identity(k+1) )

    return W



"""
Write a function `perform_cv` to evaluate a combination of hyperparameters using cross-validation. Your function should accept these arguments:
- A numpy array `x` of shape `(N,)` with the x-values of the available data
- A numpy array `y` of shape `(N,)` with the y-values of the available data
- An integer `k` for the order of the polynomial
- A float `lamb` for the regularization parameter
- An integer `folds` which is a divisor of `N` for the number of folds to use

and return the CV estimate of the MSE.
"""
def perform_cv(x, y, k, lamb, folds):
    assert len(y)%folds == 0

    fold_size = len(y)//folds
    mse = np.zeros(folds)

    for fold in range(folds):
        start = fold * fold_size
        finish = (fold+1) * fold_size

        x_test, y_test = x[start:finish], y[start:finish]
        x_train, y_train = np.array([ x[i] for i in range(len(y)) if i < start or i >= finish ]), np.array([ y[i] for i in range(len(y)) if i < start or i >= finish ])
        
        W = ridge_fit_poly(x_train, y_train, k, lamb)
        mse[fold] = mse_poly(x_test, y_test, W)
    
    return mse.sum()/len(mse)



if __name__ == "__main__":

    interval_start = 0
    interval_end = 2 * np.pi
    
    ### 2.a Basic plotting in Matplotlib

    X = np.linspace(interval_start, interval_end, 100)
    sin = np.sin(X)

    train_X = np.random.uniform(interval_start, interval_end, 15)
    train_sin = np.sin(train_X) + np.random.normal(0, 0.1, 15)

    test_X = np.random.uniform(interval_start, interval_end, 10)
    test_sin = np.sin(test_X) + np.random.normal(0, 0.1, 10)

    plt.plot(X, sin, label = "y=sin(x)")
    plt.scatter(train_X, train_sin, label = "training data")
    plt.legend()
    plt.show()

    ### 2.b Polynomial regression

    k = 3
    W = fit_poly(train_X, train_sin, k)
    poly_y = poly(X, W)

    poly_mse = round( mse_poly(test_X, test_sin, W), 4)

    plt.plot(X, sin, label = "y=sin(x)")
    plt.plot(X, poly_y[-1], label = f"poly deg = {k}, mse = {poly_mse}")
    plt.scatter(train_X, train_sin, label = "training data")
    plt.legend()
    plt.show()
    
    ### 2.c Overfitting

    interval_end = 4 * np.pi

    train_X = np.random.uniform(interval_start, interval_end, 15)
    train_sin = np.sin(train_X) + np.random.normal(0, 0.1, 15)

    test_X = np.random.uniform(interval_start, interval_end, 10)
    test_sin = np.sin(test_X) + np.random.normal(0, 0.1, 10)

    poly_mse = []
    K = list(range(1, 16))

    for k in K:
        W = fit_poly(train_X, train_sin, k)
        poly_mse.append(mse_poly(test_X, test_sin, W))

    plt.plot(K, poly_mse)
    plt.xlabel("degree of polynomial")
    plt.ylabel("mse error")
    plt.yscale('log')
    plt.show()
    #The best result is k = 7
    
    X = np.linspace(interval_start, interval_end, 100)
    sin = np.sin(X)

    k = 7
    W = fit_poly(train_X, train_sin, k)
    poly_y = poly(X, W)

    poly_mse = round( mse_poly(test_X, test_sin, W), 4)

    plt.plot(X, sin, label = "y=sin(x)")
    plt.plot(X, poly_y[-1], label = f"poly deg = {k}, mse = {poly_mse}")
    plt.scatter(train_X, train_sin, label = "training data")
    plt.legend()
    plt.show()

    ### 2.d Ridge regression

    K = list(range(1,21))
    lambs = 10 ** np.linspace(-5, 0, 20)

    poly_mse = np.zeros((20, 20))

    for i in range(20):
        k = K[i]
        for j in range(20):
            lamb = lambs[j]
            W = ridge_fit_poly(train_X, train_sin, k, lamb)
            poly_mse[i, j] = mse_poly(test_X, test_sin, W)
    
    plt.imshow(np.log(poly_mse))
    x_ticks = [0, 4, 9, 14, 19]
    plt.xticks(x_ticks, [round(lamb, 6) for lamb in lambs[x_ticks] ])
    plt.yticks(list(range(20)), K)
    plt.xlabel("lamb")
    plt.ylabel("k")
    plt.show()
    
    ### 2.e Cross-validation

    k = 7 #int(input("What was the best order of polynomial?"))
    lamb_idx = 8 #int(input("What was the best lambda index for ridge regression?"))
    #assert k in K and lamb_idx in list(range(20))
    lamb = lambs[lamb_idx]

    N = 120 #Number of data points for cv evaluation of hyperparameters
    FOLDS = [i for i in range(2, 121) if N%i==0]
    times = 100 #Number of cv evaluations of hyperparameters for certain fold
    cv = np.zeros((len(FOLDS), times))

    for time in range(times):
        dataset_X = np.random.uniform(interval_start, interval_end, N)
        dataset_sin = np.sin(dataset_X) + np.random.normal(0, 0.1, N)

        for fold_idx in range(len(FOLDS)):
            cv[fold_idx, time] = perform_cv(dataset_X, dataset_sin, k, lamb, FOLDS[fold_idx])
    
    cv_mean = np.mean(cv, axis = 1)
    cv_std = np.std(cv, axis = 1)
    cv_upper = cv_mean + cv_std
    cv_lower = cv_mean - cv_std
    cv_lower[cv_lower<0] = 0

    plt.plot(FOLDS, cv_mean)
    plt.fill_between(FOLDS, cv_lower, cv_upper, alpha = 0.2)
    plt.xlabel("Number of folds")
    plt.ylabel("cv mse error")
    plt.show()
    






        


    
            







































