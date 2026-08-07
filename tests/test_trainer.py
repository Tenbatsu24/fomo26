"""Tests for med_adapt.trainer modules."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict

from med_adapt.loss import get_loss, DiceCELoss
from med_adapt.metric import get_metric


class TestLossMetricFactories:
    def test_get_loss_cross_entropy(self):
        loss = get_loss("cross_entropy")
        assert isinstance(loss, nn.CrossEntropyLoss)

    def test_get_loss_huber(self):
        loss = get_loss("huber")
        assert isinstance(loss, nn.HuberLoss)

    def test_get_loss_dice_ce(self):
        loss = get_loss("dice_ce")
        assert isinstance(loss, DiceCELoss)

    def test_get_loss_unknown(self):
        with pytest.raises(ValueError):
            get_loss("nonexistent_loss")

    def test_get_metric_accuracy(self):
        metric = get_metric("accuracy", num_classes=2)
        assert metric is not None

    def test_get_metric_rmse(self):
        metric = get_metric("rmse")
        assert metric is not None

    def test_get_metric_unknown(self):
        with pytest.raises(ValueError):
            get_metric("nonexistent_metric")


def _make_config(**overrides):
    """Build a minimal ConfigDict for trainer tests."""
    cfg = ConfigDict()
    cfg.optimizer = ConfigDict()
    cfg.optimizer.type = "AdamW"
    cfg.optimizer.params = {"lr": 1e-3}
    cfg.pretrained = ConfigDict()
    cfg.pretrained.checkpoint = None
    cfg.model = ConfigDict()
    cfg.model.lora = False
    cfg.test = ConfigDict()
    cfg.test.batch_size = 1
    cfg.test.amp = False
    cfg.scheduler = []
    cfg.loss = None
    cfg.metrics = None
    # Runtime-injected keys bypass type safety via item assignment
    cfg["num_classes"] = overrides.get("num_classes")
    return cfg


class HeadModel(nn.Module):
    """Model with a 'head' submodule so mark_trainable keeps its params."""

    def __init__(self, in_dim=16, out_dim=2):
        super().__init__()
        self.backbone = nn.Linear(in_dim, 32)
        self.head = nn.Linear(32, out_dim)

    def forward(self, x):
        return self.head(F.relu(self.backbone(x)))

    @property
    def task(self):
        return "classification"


class TestBaseTrainerInit:
    """Test that BaseTrainer can be instantiated with a simple model."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_classification_trainer_creation(self):
        from med_adapt.trainer import ClassificationTrainer

        DEVICE = torch.device("cuda")
        model = HeadModel(in_dim=16, out_dim=2).to(DEVICE)
        config = _make_config(num_classes=2)
        trainer = ClassificationTrainer(config=config, model=model)
        assert trainer.model is model
        assert trainer.criterion is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_regression_trainer_creation(self):
        from med_adapt.trainer import RegressionTrainer

        DEVICE = torch.device("cuda")
        model = HeadModel(in_dim=16, out_dim=1).to(DEVICE)
        config = _make_config(num_classes=None)
        trainer = RegressionTrainer(config=config, model=model)
        assert trainer.model is model

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_segmentation_trainer_creation(self):
        from med_adapt.trainer import SegmentationTrainer

        class SegModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Conv3d(1, 4, 3, padding=1)
                self.head = nn.Conv3d(4, 3, 3, padding=1)

            def forward(self, x):
                return self.head(F.relu(self.backbone(x)))

            @property
            def task(self):
                return "segmentation"

        DEVICE = torch.device("cuda")
        model = SegModel().to(DEVICE)
        config = _make_config(num_classes=3)
        trainer = SegmentationTrainer(config=config, model=model)
        assert trainer.model is model

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_batch_to_loss_classification(self):
        from med_adapt.trainer import ClassificationTrainer

        DEVICE = torch.device("cuda")
        model = HeadModel(in_dim=16, out_dim=2).to(DEVICE)
        config = _make_config(num_classes=2)
        trainer = ClassificationTrainer(config=config, model=model)

        batch = {
            "image": torch.randn(4, 16, device=DEVICE),
            "label": torch.randint(0, 2, (4,), device=DEVICE),
        }
        loss, (logits, labels) = trainer.batch_to_loss(batch, train=True)
        assert loss.dim() == 0
        assert logits.shape == (4, 2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_batch_to_loss_regression(self):
        from med_adapt.trainer import RegressionTrainer

        DEVICE = torch.device("cuda")
        model = HeadModel(in_dim=16, out_dim=1).to(DEVICE)
        config = _make_config(num_classes=None)
        trainer = RegressionTrainer(config=config, model=model)

        batch = {
            "image": torch.randn(4, 16, device=DEVICE),
            "label": torch.randn(4, 1, device=DEVICE),
        }
        loss, (logits, labels) = trainer.batch_to_loss(batch, train=True)
        assert loss.dim() == 0
        assert logits.shape == (4, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_batch_to_loss_segmentation(self):
        from med_adapt.trainer import SegmentationTrainer

        class SegModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Conv3d(1, 4, 3, padding=1)
                self.head = nn.Conv3d(4, 3, 3, padding=1)

            def forward(self, x):
                return self.head(F.relu(self.backbone(x)))

            @property
            def task(self):
                return "segmentation"

        DEVICE = torch.device("cuda")
        model = SegModel().to(DEVICE)
        config = _make_config(num_classes=3)
        trainer = SegmentationTrainer(config=config, model=model)

        batch = {
            "image": torch.randn(2, 1, 8, 8, 8, device=DEVICE),
            "label": torch.randint(0, 3, (2, 1, 8, 8, 8), device=DEVICE),
        }
        loss, (logits, labels) = trainer.batch_to_loss(batch, train=True)
        assert loss.dim() == 0
        assert logits.shape == (2, 3, 8, 8, 8)
