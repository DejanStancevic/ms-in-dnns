import argparse
from datetime import datetime
import os
import sys
import json
import pathlib as pl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import wandb

if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log


class IncomeNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class AdultDataset(Dataset):
    """Adult UCI dataset, download data from https://archive.ics.uci.edu/dataset/2/adult"""

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        # one-hot encoding of categorical variables (including label)
        df = pd.get_dummies(df).astype("int32")

        data_columns = df.columns[:-2]
        labels_column = df.columns[-2:]
        self.data = torch.tensor(df[data_columns].values, dtype=torch.float32)
        self.labels = torch.tensor(df[labels_column].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

class ResampledDataset(Dataset):
    def __init__(self, dataset):

        self.class_1 = []
        self.class_2 = []

        for inputs, labels in dataset:
            if labels[0] == 1:
                self.class_1.append( (inputs, labels) )
            else:
                self.class_2.append( (inputs, labels) )
        
        self.num_1 = len(self.class_1)
        self.num_2 = len(self.class_2)
        
        if self.num_1 > self.num_2:
                multpl = (self.num_1 - self.num_2)//self.num_2
                residue = (self.num_1 - self.num_2)%self.num_2
                self.class_2 += multpl*self.class_2 + self.class_2[:residue]
        elif self.num_1 < self.num_2:
                multpl = (self.num_2 - self.num_1)//self.num_1
                residue = (self.num_2 - self.num_1)%self.num_1
                self.class_1 += multpl*self.class_1 + self.class_1[:residue]

        self.num_1 = len(self.class_1)
        self.num_2 = len(self.class_2)
        self.idxs = torch.randperm(2 * self.num_1)
        self.data = self.class_1 + self.class_2

    def __len__(self):
        return (self.num_1 * 2)
        
    def __getitem__(self, idx):
        idx = self.idxs[idx]
        return self.data[idx][0], self.data[idx][1]
        

def get_wandb_key():
    json_file = pl.Path("..", "wandb_key.json")
    if json_file.is_file():
        with open(json_file, "r") as f:
            return json.load(f)
    elif "WANDB_KEY" in os.environ:
        return os.environ["WANDB_KEY"]


def checkpoint(epoch, model, optimizer, loss, file_name):
    """
    Chekpoints epoch, model, optimizer, and loss.
    """
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, file_name)

def main(args):

    wandb.login(key=get_wandb_key())
    wandb.init(project="ms-in-dnns-income-net", config=args, name=args.run_name)

    ### LOADING DATA

    torch.manual_seed(0xDEADBEEF)

    if "LOG_PATH" in os.environ:
        data_file = pl.PurePosixPath("/gcs", "msindnn_staging", "adult_data", "adult.data")
    else:
        data_file = pl.PurePath("..", "data", "adult_data", "adult.data")

    entire_dataset = AdultDataset(str(data_file))
    train_dataset, val_dataset = random_split(
        entire_dataset, [args.train_share, 1 - args.train_share]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    ### INITIALIZING MODEL

    model = IncomeNet(train_dataset[0][0].shape[0], train_dataset[0][1].shape[0])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ### TRAINING

    best_val_loss = float("inf") 
    best_val_acc = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().detach().item()
        train_loss = total_loss / len(train_loader)

        model.eval()
        total_loss = 0.0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.cpu().item()
        val_loss = total_loss / len(val_loader)

        checkpoint(epoch, model, optimizer, val_loss, "Regular_checkpoint.pth") # Checkpointing the latest model
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint(epoch, model, optimizer, best_val_loss, "Best_val_loss_checkpoint.pth") # Checkpointing the best val_loss model

        print(
            f"Epoch [{epoch}/{args.epochs}]",
            f"Train Loss: {train_loss:.4f}",
            f"Val Loss: {val_loss:.4f}",
        )

        wandb.log({"loss": {"train": train_loss, "val": val_loss}}, step=epoch)

        true_pos = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)

            preds = torch.argmax(outputs, dim=-1)
            true_pos += (preds == torch.argmax(labels, dim=-1)).cpu().sum()
        acc = true_pos / len(val_dataset)
        wandb.log({"validation_accuracy": {"val_acc": acc}})

        if acc > best_val_acc:
            best_val_acc = acc
            checkpoint(epoch, model, optimizer, best_val_acc, "Best_val_acc_checkpoint.pth") # Checkpointing the best val_acc model

    ### EVALUATION

    best_val_acc_model = torch.load("Best_val_acc_checkpoint.pth")
    model.load_state_dict(best_val_acc_model['model_state_dict'])
    model.eval()
    true_pos = 0
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)

        preds = torch.argmax(outputs, dim=-1)
        true_pos += (preds == torch.argmax(labels, dim=-1)).cpu().sum()
    acc = true_pos / len(val_dataset)
    print(f"The best validation accuracy of training: {acc:.4f}")

    Predict_1 = []
    Predict_2 = []
    N = 10

    for inputs, labels in val_dataset:
        if labels[0] == 1 and len(Predict_1) < N:
            Predict_1.append((inputs, labels))
        elif labels[0] == 0 and len(Predict_2) < N:
            Predict_2.append((inputs, labels))
        elif len(Predict_1) > N and len(Predict_2) > N:
            break

    columns = ["Truth", "Predict"] # There are 108 input variables
    data = []
    
    for inputs, label in Predict_1:
        truth = int(label[-1])
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        predict = int(torch.argmax(outputs.cpu()))
        data.append([truth, predict])

    for inputs, label in Predict_2:
        truth = int(label[-1])
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        predict = int(torch.argmax(outputs.cpu()))
        data.append([truth, predict])

    df = pd.DataFrame(data, columns=columns)
    print(df)

    table = wandb.Table(dataframe=df)
    wandb.log({"Table": table})


def main_2(args):

    wandb.login(key=get_wandb_key())
    wandb.init(project="ms-in-dnns-income-net", config=args, name=args.run_name)

    torch.manual_seed(0xDEADBEEF)

    if "LOG_PATH" in os.environ:
        data_file = pl.PurePosixPath("/gcs", "msindnn_staging", "adult_data", "adult.data")
    else:
        data_file = pl.PurePath("..", "data", "adult_data", "adult.data")

    entire_dataset = AdultDataset(str(data_file))
    train_dataset, val_dataset = random_split(
        entire_dataset, [args.train_share, 1 - args.train_share]
    )

    if args.resample:
        train_dataset = ResampledDataset(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = IncomeNet(train_dataset[0][0].shape[0], train_dataset[0][1].shape[0])
    model = model.to(device)

    if args.loss_weight:
        total_samples = len(train_dataset)
        num_class_1 = 19809 # Already calculated
        loss_weight = torch.tensor([total_samples/num_class_1, total_samples/(total_samples-num_class_1)])
    else:
        loss_weight = None


    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    ### TRAINING

    best_val_acc = 0
    
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().detach().item()
        train_loss = total_loss / len(train_loader)

        model.eval()
        total_loss = 0.0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.cpu().item()
        val_loss = total_loss / len(val_loader)
        print(
            f"Epoch [{epoch}/{args.epochs}]",
            f"Train Loss: {train_loss:.4f}",
            f"Val Loss: {val_loss:.4f}",
        )
        wandb.log({"loss": {"train": train_loss, "val": val_loss}}, step=epoch)
        
        checkpoint(epoch, model, optimizer, val_loss, "Regular_checkpoint.pth") # Checkpointing the latest model

        true_pos = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)

            preds = torch.argmax(outputs, dim=-1)
            true_pos += (preds == torch.argmax(labels, dim=-1)).cpu().sum()
        acc = true_pos / len(val_dataset)
        wandb.log({"validation_accuracy": {"val_acc": acc}})

        if acc > best_val_acc:
            best_val_acc = acc
            checkpoint(epoch, model, optimizer, best_val_acc, "Best_val_acc_checkpoint.pth") # Checkpointing the best val_acc model

        if args.lr_scheduler:
            scheduler.step()

    model.eval()

    ### Evaluation using validation accuracy

    print(f"The best validation accuracy of training: {best_val_acc:.4f}")
    print(f"The last validation accuracy of training: {acc:.4f}")


    ### Evaluation using confusion matrix
    
    PREDS = []
    TRUTH = []
    for inputs, labels in val_dataset:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)

        preds = int(torch.argmax(outputs))
        true_label = int(torch.argmax(labels, dim=-1))
        PREDS.append(preds)
        TRUTH.append(true_label)
    cm = wandb.plot.confusion_matrix(y_true=TRUTH, preds=PREDS, class_names = ['Less than 50K', 'More than 50K'])
    wandb.log({"conf_mat": cm})
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-share", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr-scheduler", action="store_true")
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--loss-weight", action="store_true")
    if "CREATION_TIMESTAMP" in os.environ:
        timestamp = os.environ["CREATION_TIMESTAMP"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument("--run-name", type=str, default=timestamp)
    args = parser.parse_args()
    main_2(args)
