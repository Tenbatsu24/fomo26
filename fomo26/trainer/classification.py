import torch.nn as nn

from torchmetrics.classification import MulticlassAccuracy

from fomo26.trainer.base import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, config, gpu_transforms=None, norm_transforms=None):
        super().__init__(model, config, gpu_transforms, norm_transforms)
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = MulticlassAccuracy(num_classes=config["num_classes"])
        self.val_acc = MulticlassAccuracy(num_classes=config["num_classes"])

    def on_train_epoch_start(self) -> None:
        self.train_acc.reset()
        self.val_acc.reset()

    def log_train_metrics(self, outputs, batch):
        preds = outputs.argmax(dim=1)
        target = batch["target"].long().squeeze(-1)

        self.train_acc(preds, target)

        self.log(
            "train/acc",
            self.train_acc,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

    def log_val_metrics(self, outputs, batch):
        preds = outputs.argmax(dim=1)
        target = batch["target"].long().squeeze(-1)

        self.val_acc(preds, target)

        self.log(
            "val/acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, outputs, batch):
        labels = batch["target"].long().squeeze(-1)
        return self.criterion(outputs, labels)
