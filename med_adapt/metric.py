import torch
import numpy as np

from torchmetrics import Metric
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    recall_score,
    precision_score,
    f1_score
)
from scipy.stats import pearsonr


class SklearnMetricWrapper(Metric):
    """Wrapper to use sklearn metrics with torchmetrics API."""

    def __init__(
        self,
        sklearn_metric_func,
        compute_kwargs=None,
        dist_sync_on_step: bool = False,
        **metric_kwargs,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.sklearn_metric_func = sklearn_metric_func
        self.compute_kwargs = compute_kwargs or {}
        self.metric_kwargs = metric_kwargs

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def _convert_to_labels(self, preds):
        """Convert predictions to class labels."""
        if preds.ndim > 1 and preds.shape[1] > 1:
            # Multi-class probabilities -> argmax
            return np.argmax(preds, axis=1)
        elif preds.ndim > 1 and preds.shape[1] == 1:
            # Binary with shape (n, 1)
            return (preds > 0.5).astype(int).flatten()
        elif preds.ndim == 1:
            # Binary probabilities or already labels
            if preds.dtype.kind in 'iuf':  # int, uint, float
                # Check if it's probabilities (values between 0 and 1)
                if np.all((preds >= 0) & (preds <= 1)):
                    return (preds > 0.5).astype(int)
                else:
                    # Assume already labels
                    return preds.astype(int)
        return preds

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update state with predictions and targets."""
        # Convert to numpy and store
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()

        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.concatenate(self.preds, dim=0).numpy()
        targets = torch.concatenate(self.targets, dim=0).numpy()

        # Special handling for different metrics
        if self.sklearn_metric_func == roc_auc_score:
            # For AUROC, handle multi-class properly
            if preds.ndim > 1 and preds.shape[1] > 1:
                # Multi-class: use one-vs-rest
                return roc_auc_score(
                    targets, preds, multi_class="ovr", **self.metric_kwargs
                )
            else:
                # Binary: use the sklearn function directly
                return self.sklearn_metric_func(targets, preds, **self.metric_kwargs)
        elif self.sklearn_metric_func in [accuracy_score, recall_score, precision_score, f1_score]:
            # For classification metrics, convert predictions to labels if needed
            preds = self._convert_to_labels(preds)
            targets = targets.astype(int)
            return self.sklearn_metric_func(targets, preds, **self.metric_kwargs)
        else:
            # For regression metrics and others
            return self.sklearn_metric_func(targets, preds, **self.metric_kwargs)

    def reset(self):
        """Reset the metric state."""
        self.preds = []
        self.targets = []


class SklearnAccuracy(SklearnMetricWrapper):
    """Accuracy metric using sklearn."""

    def __init__(self, dist_sync_on_step: bool = False, **kwargs):
        super().__init__(
            sklearn_metric_func=accuracy_score,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )


class SklearnRecall(SklearnMetricWrapper):
    """Recall metric using sklearn."""
    def __init__(self, average='binary', dist_sync_on_step: bool = False, **kwargs):
        super().__init__(
            sklearn_metric_func=recall_score,
            dist_sync_on_step=dist_sync_on_step,
            average=average,
            **kwargs
        )


class SklearnPrecision(SklearnMetricWrapper):
    """Precision metric using sklearn."""
    def __init__(self, average='binary', dist_sync_on_step: bool = False, **kwargs):
        super().__init__(
            sklearn_metric_func=precision_score,
            dist_sync_on_step=dist_sync_on_step,
            average=average,
            **kwargs
        )


class SklearnAUROC(SklearnMetricWrapper):
    """AUROC metric using sklearn."""

    def __init__(self, dist_sync_on_step: bool = False, **kwargs):
        super().__init__(
            sklearn_metric_func=roc_auc_score,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )


class SklearnF1(SklearnMetricWrapper):
    """F1 Score metric using sklearn."""
    def __init__(self, average='binary', dist_sync_on_step: bool = False, **kwargs):
        super().__init__(
            sklearn_metric_func=f1_score,
            dist_sync_on_step=dist_sync_on_step,
            average=average,
            **kwargs
        )

class SklearnMSE(SklearnMetricWrapper):
    """Mean Squared Error metric using sklearn."""

    def __init__(self, squared=True, dist_sync_on_step: bool = False, **kwargs):
        # sklearn mse doesn't take squared parameter, it's always squared
        super().__init__(
            sklearn_metric_func=mean_squared_error,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )


class SklearnRMSE(SklearnMetricWrapper):
    """Root Mean Squared Error metric using sklearn."""

    def __init__(self, dist_sync_on_step: bool = False, **kwargs):
        super().__init__(
            sklearn_metric_func=mean_squared_error,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )

    def compute(self):
        """Compute RMSE by taking sqrt of MSE."""
        mse = super().compute()
        return np.sqrt(mse)


class SklearnMAE(SklearnMetricWrapper):
    """Mean Absolute Error metric using sklearn."""

    def __init__(self, dist_sync_on_step: bool = False, **kwargs):
        super().__init__(
            sklearn_metric_func=mean_absolute_error,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )


class SklearnR2(SklearnMetricWrapper):
    """R2 Score metric using sklearn."""

    def __init__(self, dist_sync_on_step: bool = False, **kwargs):
        super().__init__(
            sklearn_metric_func=r2_score, dist_sync_on_step=dist_sync_on_step, **kwargs
        )


class SklearnPearsonCorr(SklearnMetricWrapper):
    """Pearson Correlation Coefficient using sklearn."""

    def __init__(self, dist_sync_on_step: bool = False, **kwargs):
        super().__init__(
            sklearn_metric_func=pearsonr, dist_sync_on_step=dist_sync_on_step, **kwargs
        )

    def compute(self):
        """Compute Pearson correlation coefficient."""
        preds = torch.concatenate(self.preds, dim=0).numpy()
        targets = torch.concatenate(self.targets, dim=0).numpy()
        # pearsonr returns (correlation, p-value)
        return pearsonr(targets.flatten(), preds.flatten())[0]


def get_metric(name: str, **params):
    """Return an instantiated metric.

    Args:
        name: Metric identifier.
        **params: Keyword arguments passed to the metric constructor.

    Returns:
        A torchmetrics :class:`Metric` instance.
    """
    metrics = {
        "accuracy": lambda **p: SklearnAccuracy(**p),
        "acc": lambda **p: SklearnAccuracy(**p),
        "recall": lambda **p: SklearnRecall(**p),
        "prec": lambda **p: SklearnPrecision(**p),
        "f1": lambda **p: SklearnF1(**p),
        "auroc": lambda **p: SklearnAUROC(**p),
        "mse": lambda **p: SklearnMSE(squared=True, **p),
        "rmse": lambda **p: SklearnRMSE(**p),
        "mae": lambda **p: SklearnMAE(**p),
        "r2": lambda **p: SklearnR2(**p),
        "corr": lambda **p: SklearnPearsonCorr(**p),
    }
    if name not in metrics:
        raise ValueError(f"Unknown metric {name!r}. Available: {list(metrics)}")
    return metrics[name](**params)
