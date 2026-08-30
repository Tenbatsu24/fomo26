import torch
import torch.nn as nn
import torch.distributed as dist


class RunningNorm(nn.Module):

    def __init__(
        self,
        num_features: int,
        channel_dim: int = 1,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.channel_dim = channel_dim
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _compute_global_stats(self, x):
        reduce_dims = tuple(d for d in range(x.ndim) if d != self.channel_dim)

        # Local sufficient statistics
        count = torch.tensor(
            x.numel() // x.shape[self.channel_dim],
            dtype=x.dtype,
            device=x.device,
        )

        stats_dtype = torch.float32

        sum_ = x.to(stats_dtype).sum(dim=reduce_dims)
        sum_sq = (x.to(stats_dtype) ** 2).sum(dim=reduce_dims)

        # Synchronize across ranks if DDP is active
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_sq, op=dist.ReduceOp.SUM)

        mean = sum_ / count
        var = sum_sq / count - mean.square()

        return mean, var

    def forward(self, x):
        if self.training and self.momentum > 0:
            mean, var = self._compute_global_stats(x)

            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var, self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var

        shape = [1] * x.ndim
        shape[self.channel_dim] = -1

        mean = mean.view(shape)
        var = var.view(shape)

        return (x - mean) * torch.rsqrt(var + self.eps)

    def normalize(self, x):
        shape = [1] * x.ndim
        shape[self.channel_dim] = -1

        mean = self.running_mean.view(*shape)
        var = self.running_var.view(*shape)

        return (x - mean) * torch.rsqrt(var + self.eps)
