import numpy as np
import matplotlib.pyplot as plt

def poly(x, W):

    x = np.array([ np.array([x_pt**i for i in range(len(W[-1]))]) for x_pt in x ])
    y = W@x.transpose()

    return y

def fit_poly(x_train, y_train, k):
    
    x = np.array([ np.array([x_pt**i for i in range(k+1)]) for x_pt in x_train ])
    W = (y_train[None]@x) @ np.linalg.inv(x.transpose()@x)

    return W


def mse_poly(x, y, W):
    
    difference = y - poly(x, W)
    mse_error = (difference@difference.transpose()) / len(y)

    return mse_error


def ridge_fit_poly(x_train, y_train, k, lamb):

    x = np.array([ np.array([x_pt**i for i in range(k+1)]) for x_pt in x_train ])
    W = (y_train[None]@x) @ np.linalg.inv( x.transpose()@x + lamb * np.identity(k+1) )

    return W


if __name__ == "__main__":

    X = np.linspace(0, 2*np.pi, 100)
    sin = np.sin(X)

    train_X = np.random.uniform(0, 2*np.pi, 15)
    train_sin = np.sin(train_X) + np.random.normal(0, 0.1, 15)

    test_X = np.random.uniform(0, 2*np.pi, 10)
    test_sin = np.sin(test_X) + np.random.normal(0, 0.1, 10)

    plt.plot(X, sin)
    plt.scatter(data_X, data_sin)
    plt.show()

























