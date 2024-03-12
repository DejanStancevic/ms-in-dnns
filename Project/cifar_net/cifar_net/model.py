import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg16_bn
import torchmetrics
import lightning as L
import wandb

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, 7)
        self.conv2 = nn.Conv2d(10, 10, 5)
        
        self.fc1 = nn.Linear(10*18*18, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 10*18*18)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        return(x)
    

class LitCIFARNet(L.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.save_hyperparameters()

        self.model = Net()
        
        self.criterion = nn.NLLLoss()
        self.lr = lr
        metrics = torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.classification.MulticlassAccuracy(num_classes=10),
                "precision": torchmetrics.classification.MulticlassPrecision(num_classes=10),
                "recall": torchmetrics.classification.MulticlassRecall(num_classes=10),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.best_metrics = metrics.clone(prefix="best/")
        self.confusion_matrix = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=10).to(self.device)
        self.conf_mat = torch.zeros(10, 10)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)

        self.train_metrics.update(preds, labels)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log_dict(self.train_metrics, on_epoch=True, on_step=False)
        self.log("step", float(self.current_epoch + 1), on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)

        acc = (preds == torch.argmax(labels, dim=-1)).sum() / inputs.shape[0]
        self.val_metrics.update(preds, labels)
        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log("val/acc_manual", acc, on_epoch=True, on_step=False)
        self.log_dict(self.val_metrics, on_epoch=True, on_step=False)
        self.log("step", float(self.current_epoch + 1), on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)

        self.confusion_matrix.update(preds, labels)
        self.conf_mat += self.confusion_matrix.compute().cpu()
        self.best_metrics.update(preds, labels)
        self.log("best/loss", loss, on_epoch=True, on_step=False)
        self.log_dict(self.best_metrics, on_epoch=True, on_step=False)

    def on_test_epoch_end(self):
        class_names = [str(i) for i in range(10)]
        data = []
        counts = self.conf_mat/self.conf_mat.sum(dim=1)

        for i in range(10):
            for j in range(10):
                data.append([class_names[i], class_names[j], counts[i, j].item()])
        fields = {"Actual": "Actual", "Predicted": "Predicted", "nPredictions": "nPredictions"}
        conf_mat = wandb.plot_table(
            "wandb/confusion_matrix/v1",
            wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
            fields,
            {"title": "confusion matrix on best epoch"},
            split_table=True,
            )
        wandb.log({"best/conf_mat": conf_mat})
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)