import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


"""
Returns an output of a polynomial with input x and coeffs W
"""
def poly(x, W):

    x = torch.tensor([ [x_pt**i for i in range(len(W))] for x_pt in x ])
    y = W@x.transpose(0,1)

    return y

"""
Add a function `fit_poly` to your script which performs polynomial regression **by implementing the closed-form solution for linear regression discussed in Lecture 2 in NumPy**. It should accept the following arguments
- A numpy array `x_train` of shape `(N,)` with the x-values of the training data
- A numpy array `y_train` of shape `(N,)` with the y-values of the training data
- An integer `k` for the order of the polynomial

where `N` is the number of training samples. This function should return a numpy array of shape `(k+1)` with the weights of the fitted polynomial.
"""
def fit_poly(x_train, y_train, k):
    
    x = torch.tensor([ [x_pt**i for i in range(k+1)] for x_pt in x_train ])
    W = (y_train[None]@x) @ torch.linalg.inv(x.transpose(0,1)@x)

    return W[-1]

"""
Write a function `mse_poly` which returns the mean squared error of a fit polynomial on test data and has the arguments
- A numpy array `x` of shape `(N,)` with the x-values of the test data
- A numpy array `y` of shape `(N,)` with the y-values of the test data
- A numpy array `W` of shape `(k+1)` with the weights of the polynomial
"""
def mse_poly(x, y, W):
    
    difference = y - poly(x, W)
    mse_error = (difference@difference.transpose(0,1)) / len(y)

    return mse_error[0, 0]



k = 3 # Order of polynomial
N_TRAIN = 15
SIGMA_NOISE = 0.1

x = torch.linspace(0, 2*torch.pi, 100)
torch.manual_seed(0xDEADBEEF)
x_train = torch.rand(N_TRAIN) * 2 * torch.pi
y_train = torch.sin(x_train) + torch.randn(N_TRAIN) * SIGMA_NOISE


if __name__ == "__main__":
    
    ### 1.a Gradient descend and initialization

    W = torch.tensor([1., 1., 1., 1.], requires_grad=True)
    loss = nn.MSELoss()
    LOSS = []
    lr = 6.9183e-05 # 2.0e-6 starts conv. in 100 steps and 7.3e-5 stops conv and start div
    """
    Jupyter notebook was used to determine the best learning rate. This is part of the code that was used:
LR = 10**( torch.linspace(-8, -1, 21) )
LOSS = []

for lr in LR:
    W = torch.tensor([1., 1., 1., 1.], requires_grad=True)
    optimizer = optim.SGD([W], lr)
    
    for t in range(100):
        optimizer.zero_grad()
        loss(poly(x_train, W), y_train).backward()
        optimizer.step()

    LOSS.append( loss(poly(x_train, W), y_train).detach() )
    """
    optimizer = optim.SGD([W], lr)
    STEPS = [i for i in range(1, 101)]

    for step in STEPS:
        optimizer.zero_grad()
        loss(poly(x_train, W), y_train).backward()
        optimizer.step()
        LOSS.append(loss(poly(x_train, W), y_train).detach())

    print(LOSS[-1])

    plt.plot(STEPS, LOSS)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.show()

    plt.plot(x, torch.sin(x), label = "ground truth")
    plt.scatter(x_train, y_train, label = "training data")
    plt.plot(x, poly(x, W.detach()), label = "optimized poly")
    plt.plot(x, poly(x, fit_poly(x_train, y_train, k)), label = "exact poly")
    plt.legend()
    plt.show()

    ### 1.b Different initialization and optimization algorithms
    
    ### SGD with momentum and different initialization
    
    W = torch.tensor([1.0, 1.0e-1, 1.0e-2, 1.0e-3], requires_grad=True)
    """
I used the following code to find the best learning rate and momentum. I used jupyter notebook for this as well.
LR = 10**( torch.linspace(-8, -1, 21) )
MOMENTUM = torch.linspace(0, 1, 11)
LOSS = torch.zeros(21, 11)

for lr_idx in range(21):
    for momentum_idx in range(11):
        W = torch.tensor([1.0, 1.0e-1, 1.0e-2, 1.0e-3], requires_grad=True)
        optimizer = optim.SGD([W], LR[lr_idx], MOMENTUM[momentum_idx])
    
        for t in range(100):
            optimizer.zero_grad()
            loss(poly(x_train, W), y_train).backward()
            optimizer.step()

        LOSS[lr_idx, momentum_idx] = loss(poly(x_train, W), y_train).detach()
    """
    lr, momentum = 7.0795e-05, 0.95
    optimizer = optim.SGD([W], lr, momentum)
    LOSS = []

    for step in STEPS:
        optimizer.zero_grad()
        loss(poly(x_train, W), y_train).backward()
        optimizer.step()
        LOSS.append(loss(poly(x_train, W), y_train).detach())

    print(LOSS[-1])

    plt.plot(STEPS, LOSS)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.show()

    plt.plot(x, torch.sin(x), label = "ground truth")
    plt.scatter(x_train, y_train, label = "training data")
    plt.plot(x, poly(x, W.detach()), label = "optimized poly")
    plt.plot(x, poly(x, fit_poly(x_train, y_train, k)), label = "exact poly")
    plt.legend()
    plt.show()

    ### ADAM optimizer

    W = torch.tensor([1.0, 1.0e-1, 1.0e-2, 1.0e-3], requires_grad=True)
    lr = 0.0282
    optimizer = optim.Adam([W], lr)
    LOSS = []

    for step in STEPS:
        optimizer.zero_grad()
        loss(poly(x_train, W), y_train).backward()
        optimizer.step()
        LOSS.append(loss(poly(x_train, W), y_train).detach())

    print(LOSS[-1])

    plt.plot(STEPS, LOSS)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.show()

    plt.plot(x, torch.sin(x), label = "ground truth")
    plt.scatter(x_train, y_train, label = "training data")
    plt.plot(x, poly(x, W.detach()), label = "optimized poly")
    plt.plot(x, poly(x, fit_poly(x_train, y_train, k)), label = "exact poly")
    plt.legend()
    plt.show()

    ### LBFGS optimizer

    W = torch.tensor([1.0, 1.0e-1, 1.0e-2, 1.0e-3], requires_grad=True)
    lr = 0.25
    optimizer = optim.LBFGS([W], lr)
    LOSS = []

    def closure():
        optimizer.zero_grad()
        objective = loss(poly(x_train, W), y_train)
        objective.backward()
        return objective
    
    for step in STEPS:
        optimizer.zero_grad()
        loss(poly(x_train, W), y_train).backward()
        optimizer.step(closure)
        LOSS.append(loss(poly(x_train, W), y_train).detach())

    print(LOSS[-1])

    plt.plot(STEPS, LOSS)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.show()

    plt.plot(x, torch.sin(x), label = "ground truth")
    plt.scatter(x_train, y_train, label = "training data")
    plt.plot(x, poly(x, W.detach()), label = "optimized poly")
    plt.plot(x, poly(x, fit_poly(x_train, y_train, k)), label = "exact poly")
    plt.legend()
    plt.show()





    
    



