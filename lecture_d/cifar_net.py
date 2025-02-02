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
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import wandb

if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log


def get_wandb_key():
    json_file = "../wandb_key.json"
    if os.path.isfile(json_file):
        with open(json_file, "r") as f:
            return json.load(f)
    elif "WANDB_KEY" in os.environ:
        return os.environ["WANDB_KEY"]


class CIFARNet(nn.Module):
    def __init__(self, num_classes=10, batchnorm = False, dropout = False, drop_p = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=drop_p) if dropout else nn.Identity(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=drop_p) if dropout else nn.Identity(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=drop_p) if dropout else nn.Identity(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=drop_p) if dropout else nn.Identity(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_p) if dropout else nn.Identity(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_p) if dropout else nn.Identity(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

class toy_model(nn.Module):
    def __init__(self, num_classes=10, batchnorm = False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=3), # 32X32 -> 30X30
            nn.BatchNorm2d(2) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 30X30 -> 15X15
            nn.Conv2d(2, 1, kernel_size=3, padding=1), # 15X15 -> 15X15
            nn.BatchNorm2d(1) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            )
        self.classifier = nn.Sequential(
            nn.Linear(1 * 15 * 15, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def checkpoint(epoch, model, optimizer, loss, path, file_name):
    """
    Chekpoints epoch, model, optimizer, and loss.
    """
    path = str(path) + f"/{file_name}"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)
    

def main(args):

    wandb.login(key=get_wandb_key())
    wandb.init(project="ms-in-dnns-cifar-net", config=args, name=args.run_name)

    ### LOADING DATA

    torch.manual_seed(0xDEADBEEF)

    if args.augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.2),
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        val_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    else:
        train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        val_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    if "LOG_PATH" in os.environ:
        path = pl.PurePosixPath(os.environ["LOG_PATH"]).parent # /gcs/ms-.../cifar_date without /log.txt
        train_dataset = torchvision.datasets.CIFAR10(path, train = True, transform = train_transform, download = True)
        val_dataset = torchvision.datasets.CIFAR10(path, train = False, transform = val_transform, download = True)
    else:
        path = '.'
        train_dataset = torchvision.datasets.CIFAR10("../data/CIFAR_data", train = True, transform = train_transform, download = True)
        val_dataset = torchvision.datasets.CIFAR10("../data/CIFAR_data", train = False, transform = val_transform, download = True)
        train_dataset, _ = random_split(train_dataset, [0.01, 0.99])


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    ### INITIALIZING MODEL

    if args.toy_model:
        model = toy_model() # Toy Model for testing the code on the laptop
        print("Using toy model")
    else:
        model = CIFARNet(batchnorm=args.batchnorm, dropout=args.dropout)
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
        true_pos = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=-1)
            true_pos += (preds == labels).cpu().sum()
            total_loss += loss.cpu().item()
        acc = true_pos / len(val_dataset)
        val_loss = total_loss / len(val_loader)

        if epoch%15 == 0:
            checkpoint(epoch, model, optimizer, val_loss, path, "Regular_CIFAR_NET_checkpoint.pt") # Checkpointing the latest model

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint(epoch, model, optimizer, best_val_loss, path, "Best_val_loss_CIFAR_NET_checkpoint.pt") # Checkpointing the best val_loss model

        if acc > best_val_acc:
            best_val_acc = acc
            checkpoint(epoch, model, optimizer, best_val_acc, path, "Best_val_acc_CIFAR_NET_checkpoint.pt") # Checkpointing the best val_acc model

        print(
            f"Epoch [{epoch}/{args.epochs}]",
            f"Train Loss: {train_loss:.4f}",
            f"Val Loss: {val_loss:.4f}",
        )
        wandb.log({"loss": {"train": train_loss, "val": val_loss}}, step=epoch)
        print(f"Val Acc: {acc}")
        wandb.log({"validation_accuracy": {"val_acc": acc}})        

        
    ### EVALUATION
    
    best_val_acc_model = torch.load(str(path) + "/Best_val_acc_CIFAR_NET_checkpoint.pt")
    model.load_state_dict(best_val_acc_model['model_state_dict'])
    print(f"Best Val Acc: {best_val_acc_model['loss']}",
          f"Best Val Acc Epoch: {best_val_acc_model['epoch']}")

    model.eval()
    columns = ['Image', 'Ground Truth', 'Prediction']
    IMAGES = []
    n = 5 # Number of images per class
    images_per_class = torch.zeros(10,)
    table = wandb.Table(columns=columns)
    CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    for image, label in val_dataset:
        if images_per_class[label] < n:
            image = image.to(device)
            output = model(image[None])
            prediction = int(torch.argmax(output).cpu())
            image = image.cpu()
            table.add_data(wandb.Image(image), CLASS_NAMES[label], CLASS_NAMES[prediction])
            images_per_class[label] += 1
        elif images_per_class.sum() == 10 * n:
            break
    
    wandb.log({"Table": table})








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--toy-model", action="store_true")
    if "CREATION_TIMESTAMP" in os.environ:
        timestamp = os.environ["CREATION_TIMESTAMP"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument("--run-name", type=str, default=timestamp)
    args = parser.parse_args()
    main(args)

