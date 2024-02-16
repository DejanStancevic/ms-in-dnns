from datetime import datetime
import os
import sys
import pathlib as pl

import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar, ModelCheckpoint

from cifar_net.data import CifarDataModule
from cifar_net.model import LitCIFARNet
from cifar_net.utils import get_wandb_key, args_to_flat_dict

if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log


def main(args):

    seed_everything(0xDEADBEEF, workers=True)
    torch.hub.set_dir(args.torch_cache_dir)

    if "LOG_PATH" in os.environ:
        wandb_save_dir = os.path.dirname(os.environ["LOG_PATH"])
    else:
        wandb_save_dir = "."
    wandb.login(key=get_wandb_key())
    args.trainer.logger = WandbLogger(
        project="ms-in-dnns-cifar-net-lightning", name=args.run_name, save_dir=wandb_save_dir
    )
    args.trainer.logger.experiment.config.update(args_to_flat_dict(args))
    

    dm = CifarDataModule(**vars(args.data))
    model = LitCIFARNet(**vars(args.model))

    args.trainer.callbacks = [
        RichModelSummary(max_depth=2),
        RichProgressBar(),
        ModelCheckpoint(
            monitor="val/acc",
            mode="max",
            save_last=True,
            filename="epoch={epoch}-val_acc={val/acc:.2f}",
            auto_insert_metric_name=False,
        ),
    ]

    trainer = Trainer(**vars(args.trainer))
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(Trainer, "trainer")
    parser.set_defaults({"trainer.max_epochs": 1,
                         "trainer.num_sanity_val_steps": 2})
    
    parser.add_lightning_class_args(LitCIFARNet, "model")
    parser.set_defaults({"model.lr": 1e-3, "model.batchnorm": False,
                         "model.dropout": False,
                         "model.vgg16": False,
                         "model.pretrained": False})
    
    parser.add_lightning_class_args(CifarDataModule, "data")
    parser.set_defaults({"data.augment": False,
                         "data.batch_size": 32})


    if "LOG_PATH" in os.environ:
        bucket_name = os.environ["BUCKET"].split("gs://")[1]
        parser.set_defaults({"data.data_root": str(pl.PurePosixPath("/gcs", bucket_name,
                                                                    "CIFAR_data"))})
    else:
        parser.set_defaults({"data.data_root": str(pl.PurePath("..", "..", "..", "data",
                                                               "CIFAR_data"))})
    
    if "CREATION_TIMESTAMP" in os.environ:
        timestamp = os.environ["CREATION_TIMESTAMP"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument("--run-name", type=str, default=timestamp)

    if "LOG_PATH" in os.environ:
        torch_cache_dir = str(pl.PurePosixPath("/gcs", bucket_name, "torch_cache"))
    else:
        torch_cache_dir = str(pl.PurePath("..", "..", "..", "torch_cache"))
    parser.add_argument("--torch-cache-dir", type=str, default=torch_cache_dir)

    args = parser.parse_args()
    main(args)

    