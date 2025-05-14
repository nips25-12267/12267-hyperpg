import lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint

import wandb

from .datasets.dataset_factory import create_dataloader


class LightningWrapper(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, loss_fn, num_classes: int) -> None:
        super().__init__()

        self.model = model
        self.val_step_outputs = []
        self.loss_fn = loss_fn
        self.num_classes = num_classes

    def training_step(self, batch, batch_idx):
        data, label = batch

        pred, dens = self.model(data)
        if type(dens) is tuple:
            _, dist = dens
            loss_dict: dict = self.loss_fn(pred, label, dist)
        else:
            loss_dict: dict = self.loss_fn(pred, label, dens)

        loss = 0
        for _, v in loss_dict.items():
            loss += v

        loss_dict["loss"] = loss
        loss_dict["epoch"] = self.trainer.current_epoch
        wandb.log({"train/" + k: v for k, v in loss_dict.items()})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.0001, weight_decay=1e-4
        )  # TODO: Add lr to config
        return optimizer

    def validation_step(self, batch, batch_idx):
        data, label = batch
        pred, density = self.model(data)
        pred = F.softmax(pred, dim=-1)
        acc = torchmetrics.functional.accuracy(
            pred, label, task="multiclass", num_classes=self.num_classes
        )
        self.val_step_outputs.append(acc)

    def on_validation_epoch_end(self):
        avg_acc = torch.stack(self.val_step_outputs).mean()
        self.val_step_outputs.clear()
        epoch = self.trainer.current_epoch
        wandb.log({"test/acc": avg_acc, "test/epoch": epoch})


def run_training(cfg, model: torch.nn.Module, loss_fn, debug: bool = False):
    pl.seed_everything(cfg.seed)
    wandb_dir = wandb.run.dir
    train_dl, test_dl = create_dataloader(cfg)

    # wandb_logger = pl_loggers.WandbLogger(experiment=wandb.run)

    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_dir,
        every_n_epochs=10,
        filename="{epoch:03d}",
    )

    lightning_model = LightningWrapper(
        model, loss_fn, num_classes=cfg.dataset.num_classes
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        fast_dev_run=debug,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(lightning_model, train_dl, test_dl)
