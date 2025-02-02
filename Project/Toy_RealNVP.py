from sklearn import cluster, datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


### DATA HANDLING

def moons(n_samples, noise=0.1):
    """
    Generates moons from sklearn.datasets with n_samples samples
    """
    return datasets.make_moons(n_samples=n_samples, noise=noise, random_state=170)

def circles(n_samples, noise=0.1):
    """
    Generates concentric circles from sklearn.datasets with n_samples samples
    """
    return datasets.make_circles(n_samples=n_samples, noise=noise, random_state=170)

class DataSet(Dataset):
    def __init__(self, x_data, y_data):
        x_data = x_data.astype(np.float32)
        y_data = y_data.astype(np.int_)
        self.X = torch.from_numpy(x_data)
        self.Y = (torch.from_numpy(y_data)).type(torch.LongTensor)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]


### NORMALIZING FLOW
class STNetwork(nn.Module):
    """
    Neural network that represents S and T functions in coupling layers
    """
    def __init__(self, num_inputs=2, num_outputs=2):
        super().__init__()
        self.hidden1 = 100
        self.hidden2 = 100
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.ReLU(),
            nn.Linear(self.hidden2, num_outputs)
        )

    def forward(self, x):
        return self.layers(x)
    
class CouplingLayer(nn.Module):
    def __init__(self, network, mask):
        super().__init__()
        self.network = network
        self.register_buffer("mask", mask)

    def forward(self, z, ldj, reverse=False):
        z_in = z * self.mask
        st = self.network(z_in)
        s, t = st.chunk(2, dim=1)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        if not reverse:
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=1)
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=1)

        return z, ldj

class Flow(nn.Module):
    def __init__(self, num_couplings, num_inputs=2):
        super().__init__()
        layer_list = []

        for i in range(num_couplings):
            layer_list.append( CouplingLayer(STNetwork(num_inputs=num_inputs, num_outputs=num_inputs), Flow.make_mask(num_inputs, i%2)) )

        self.layers = nn.ModuleList(layer_list)
    
    def forward(self, ins, reverse=False):
        z, ldj = ins, torch.zeros(ins.shape[0])
        for layer in reversed(self.layers) if reverse else self.layers:
            z, ldj = layer(z, ldj, reverse=reverse)

        return z, ldj
    
    def make_mask(size, pattern):
        mask = torch.tensor([(i+pattern)%2 for i in range(size)])
        return mask

def nll(z_in, ldj, z_dist: list, labels, num_classes=2):
    Classes = [[] for i in range(num_classes)]
    for i in range(len(labels)):
        Classes[int(labels[i])].append(i)

    log_qz = torch.tensor([])

    for i in range(num_classes):
        log_qz = torch.cat( (log_qz, z_dist[i].log_prob(z_in[Classes[i]])), 0)
    
    log_qx = ldj + log_qz
    nll = -log_qx
    nll = nll.mean()
    return nll

def predict(model, z_dist: list, inputs):
    z_out, _ = model(inputs, reverse=False)
    logprob = torch.stack( tuple([dist.log_prob(z_out) for dist in z_dist]) )
    return logprob.argmax(dim=0)

def NormalizingFlow(train_dataloader, epochs=40, lr=1e-4, mean=4.0):

    z_dist0 = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([mean, 0.]), torch.eye(2))
    z_dist1 = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([-mean, 0.]), torch.eye(2))
    z_dist = [z_dist0, z_dist1]
    model = Flow(num_couplings=5, num_inputs=2)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    ### TRAINING
    model.train()

    for epoch in range(epochs):
        #print(epoch)
        for inputs, labels in train_dataloader:
            z, ldj = model(inputs, reverse=False)
            loss = nll(z, ldj, z_dist, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, z_dist

def acc_flow(flow, z_dist, test_dataloader):
    flow.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
                preds = predict(flow, z_dist, inputs)
                correct += (preds==labels).sum()
    print(correct/len(test_data))



### 3-LAYER FNN
    
class FNN(nn.Module):
    def __init__(self, num_ins=2, num_outs=2):
        super().__init__()
        self.hidden1 = 100
        self.hidden2 = 100
        self.classifier = nn.Sequential(
            nn.Linear(num_ins, self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.ReLU(),
            nn.Linear(self.hidden2, num_outs),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.classifier(x)
        x = nn.functional.log_softmax(x)
        return x
    
def FNN3(train_dataloader, epochs=40, lr=1e-4):
    model = FNN()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    Loss = []
    for epoch in range(epochs):
        for inputs, labels in train_dataloader:
            outs = model(inputs)
            loss = criterion(outs, labels)
            Loss.append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    plt.plot([step for step in range(len(Loss))], Loss)
    plt.show()
    
    return model

def acc_fnn(fnn, test_dataloader):
    fnn.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
                outs = fnn(inputs)
                preds = outs.argmax(dim=1)
                correct += (preds==labels).sum()
    print(correct/len(test_data))


if __name__ == "__main__":
    ### DATA

    batch_size = 32
    num_samples, split = 1000, 850

    positions, Labels = moons(num_samples)
    training_data, test_data = DataSet(positions[:split], Labels[:split]), DataSet(positions[split:], Labels[split:])
    train_dataloader, test_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True), DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    ### NORMALIZING FLOW

    flow, z_dist = NormalizingFlow(train_dataloader=train_dataloader, epochs=10)
    acc_flow(flow, z_dist, test_dataloader=test_dataloader)

    """
    positions = torch.from_numpy(positions.astype(np.float32))
    z, ldj = flow(positions, reverse=False)

    colors = np.array(['blue', 'orange'])

    predictions = predict(flow, z_dist, positions) 
    Labels = torch.tensor([int(i) for i in Labels])
    
    plt.scatter(z[:, 0].detach(), z[:, 1].detach(), color=colors[Labels])
    plt.show()
    """
    ### 3-LAYER FNN

    fnn = FNN3(train_dataloader=train_dataloader, epochs=10)
    acc_fnn(fnn, test_dataloader=test_dataloader)



