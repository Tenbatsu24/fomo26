import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvexModalityAdapter(nn.Module):
    def __init__(
        self,
        n_modalities: int,
    ):
        super().__init__()

        self.n_modalities = n_modalities

        self.logits = nn.Parameter(torch.zeros(1, n_modalities))

    @property
    def weights(self):
        return self.logits.softmax(dim=1)

    def forward(self, x):
        weight = self.weights[:, :, None, None, None]

        return F.conv3d(
            x,
            weight,
            bias=None,
            stride=1,
            padding=0,
        )


if __name__ == "__main__":
    adapter = ConvexModalityAdapter(
        n_modalities=3,
    ).to("cuda")

    x = torch.randn(2, 3, 296, 296, 296, device="cuda")

    y = adapter(x)

    print(y.shape)
    print(adapter.weights)
