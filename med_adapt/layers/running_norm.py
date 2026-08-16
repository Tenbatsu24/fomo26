import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(self, num_features, channel_dim=1, momentum=0.9, eps=1e-5):
        super().__init__()

        self.channel_dim = channel_dim
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        reduce_dims = tuple(d for d in range(x.ndim) if d != self.channel_dim)

        if self.training:
            mean = x.mean(dim=reduce_dims)
            var = x.var(dim=reduce_dims, unbiased=False)

            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var, self.momentum)

        else:
            mean = self.running_mean
            var = self.running_var

        shape = [1] * x.ndim
        shape[self.channel_dim] = -1

        mean = mean.view(*shape)
        var = var.view(*shape)

        return (x - mean) / torch.sqrt(var + self.eps)
