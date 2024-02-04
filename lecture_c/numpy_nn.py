import numpy as np
import matplotlib.pyplot as plt

class NPLinear():
    
    def __init__(self, in_features, out_features):
        self.W = np.random.uniform(-1/np.sqrt(in_features), +1/np.sqrt(in_features), (out_features, in_features)) # Weights
        self.b = np.random.uniform(-1/np.sqrt(in_features), +1/np.sqrt(in_features), (out_features)) # Biases

        self.W_grad = None # Gradient of weights
        self.b_grad = None # Gradient of biases

        self.ins = None # Input to the layer
        self.outs = None # Outoup of the layer

    def forward(self, x_in):
        """
        `forward`, which accepts a batch of inputs of shape `(batch, in_channels)` and returns the output $Wx+b$ of the layer
        """
        self.ins = x_in
        self.outs = x_in @ self.W.T + self.b
        return self.outs
    
    def backward(self, dL_dout):
        """
        `backward`, which accepts the gradient of the loss with respect to the output on a batch of inputs
        of shape `(batch, out_channels)` and sets `W_grad` and `b_grad` to the gradients of the loss
        w.r.t. `W` and `b`, summed over samples in the batch. Additionally, return the gradient of the
        loss w.r.t. the layer's input.
        """
        self.W_grad = dL_dout.T @ self.ins
        self.b_grad = np.sum(dL_dout, axis=0)

        dL_dx = dL_dout @ self.W

        return dL_dx

    def gd_update(self, lr: float):
        """
        `gd_update` which accepts a learning rate (float) and performs a gradient descent step with that 
        learning rate on the weights and biases
        """
        self.W -=  lr * self.W_grad
        self.b -=  lr * self.b_grad


class NPMSELoss():

    def __init__(self):
        self.predictions = None
        self.targets = None
        self.difference = None # Difference between predictions and targets
        self.Loss_grad = None # Gradient of Loss

    def forward(self, predictions, targets):
        """
        `forward` which accepts prediction- and target arrays of shape `(batch, channels)` and returns the MSE loss averaged over samples
        """
        self.predictions, self.targets = predictions, targets
        self.difference = self.predictions - self.targets

        return np.mean(self.difference**2)

    def backward(self):
        """
        `backward` which accepts no arguments and returns the gradient of the loss with respect to the perdictions.
        """
        num_samples = self.predictions.size
        self.Loss_grad = (2 * self.difference)/num_samples
        
        return self.Loss_grad


class NPReLU():

    def __init__(self):
        self.ins = None
        self.outs = None
        self.dL_dx = None

    def forward(self, ins):
        """
        `forward` which accepts an array of inputs of shape `(batch, channels)` and returns the ReLU of that input
        """
        self.ins = ins
        self.outs = np.maximum(self.ins, 0)

        return self.outs
    
    def backward(self, dL_dout):
        """
        `backward` which accepts an array of loss-gradients of shape `(batch, channels)` and returns 
        the loss-gradient w.r.t. the input of the ReLU layer.
        """
        dReLU_dx = np.ones_like(self.ins)
        dReLU_dx[self.ins < 0] = 0
        self.dL_dx = dL_dout * dReLU_dx

        return self.dL_dx

class NPModel():

    def __init__(self):
        self.layer1 = NPLinear(1, 20)
        self.relu1 = NPReLU()
        self.layer2 = NPLinear(20, 10)
        self.relu2 = NPReLU()
        self.layer3 = NPLinear(10, 1)

    def forward(self, x):
        """
        `forward` which accepts a batch of inputs of shape `(batch, 1)` (i.e. one input channel) and returns the prediction
        """
        x = self.layer1.forward(x)
        x = self.relu1.forward(x)
        x = self.layer2.forward(x)
        x = self.relu2.forward(x)
        x = self.layer3.forward(x)

        return x
    
    def backward(self, Loss_grad):
        """
        `backward` which accepts the gradient of the loss of shape `(batch, 1)` (i.e. one output channel) and 
        sets the gradients of the weights and biases
        """
        dL_dout = self.layer3.backward(Loss_grad)
        dL_dout = self.relu2.backward(dL_dout)
        dL_dout = self.layer2.backward(dL_dout)
        dL_dout = self.relu1.backward(dL_dout)
        dL_dout = self.layer1.backward(dL_dout)


    def gd_update(self, lr: float):
        """
        `gd_update` which accepts the learning rate and performs one step of gradients descent
        """
        self.layer1.gd_update(lr)
        self.layer2.gd_update(lr)
        self.layer3.gd_update(lr)


if __name__ == "__main__":

    ### Data for training and testing

    N_TRAIN = 100
    N_TEST = 1000
    SIGMA_NOISE = 0.1

    np.random.seed(0xDEADBEEF)
    X = np.linspace(- np.pi, np.pi, 100)[:, None]
    x_train = np.random.uniform(low=-np.pi, high=np.pi, size=N_TRAIN)[:, None]
    y_train = np.sin(x_train) + np.random.randn(N_TRAIN, 1) * SIGMA_NOISE

    x_test = np.random.uniform(low=-np.pi, high=np.pi, size=N_TEST)[:, None]
    y_test = np.sin(x_test) + np.random.randn(N_TEST, 1) * SIGMA_NOISE

    ### Model and learning rate
    
    model = NPModel()
    loss = NPMSELoss()
    TRAIN_LOSS = []
    VAL_LOSS = []
    TRAINED_SIN = [] # Evaluations of our model at EPOCHS_CHECK
    lr = 0.25
    alpha = 0.99 # Learning rate decay rate

    ### Training

    epochs = 1000
    EPOCHS = np.linspace(1, epochs, epochs)
    EPOCHS_CHECK = [10, 100, epochs] # Evaluations of our model at those epochs

    for epoch in EPOCHS:
        lr *= alpha
        out = model.forward(x_train)
        target = y_train

        TRAIN_LOSS.append( loss.forward(out, target) )
        model.backward(loss.backward())
        model.gd_update(lr)

        if epoch in EPOCHS_CHECK:
            TRAINED_SIN.append(model.forward(X))

        val_out = model.forward(x_test)
        val_target = y_test
        VAL_LOSS.append( loss.forward(val_out, val_target) )

    ### Plotting Losses
        
    plt.plot(EPOCHS, TRAIN_LOSS, label = "training loss")
    plt.plot(EPOCHS, VAL_LOSS, label = "validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.show()

    ### Plotting different stages of model training vs the ground truth

    plt.plot(X, np.sin(X), label = "ground truth")
    for i in range(len(EPOCHS_CHECK)):
        plt.plot(X, TRAINED_SIN[i], label = f"model after {EPOCHS_CHECK[i]}")
    plt.scatter(x_train, y_train, label = "training data")
    plt.legend()
    plt.show()

    ### Final loss, 0.02628888054121368

    out = model.forward(x_test)
    target = y_test
    print( loss.forward(out, target) )


