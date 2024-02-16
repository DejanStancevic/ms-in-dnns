import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16_bn
import torchmetrics
import lightning as L
import wandb

class CIFARNet(nn.Module):
    def __init__(self, num_classes=10, batchnorm=False, dropout=False, drop_p=0.3):
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
        """

        self.features = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3), # 32X32 -> 30X30
            nn.BatchNorm2d(1) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 30X30 -> 15X15
            nn.Conv2d(1, 1, kernel_size=3, padding=1), # 15X15 -> 15X15
            nn.BatchNorm2d(1) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            )
        self.classifier = nn.Sequential(
            nn.Linear(15*15*1, num_classes),
        )
        """

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class my_VGG16(nn.Module):
    def __init__(self, pretrained=False, num_classes=10):
        super().__init__()
        vgg16 = vgg16_bn(pretrained=pretrained)
        
        self.features = vgg16.features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LitCIFARNet(L.LightningModule):
    def __init__(self, lr: float, batchnorm: bool, dropout: bool, vgg16: bool, pretrained: bool):
        super().__init__()
        self.save_hyperparameters()

        if vgg16:
            self.model = my_VGG16(pretrained=pretrained)
            print(f"Using vgg16: {pretrained}")
        else:
            self.model = CIFARNet(batchnorm=batchnorm, dropout=dropout)
            print("Using CIFARNet")
        
        self.criterion = nn.CrossEntropyLoss()
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
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        data = []
        counts = self.conf_mat/self.conf_mat.sum(dim=0)

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