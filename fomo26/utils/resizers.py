import torch
import torchvision.transforms.functional as TF


class MaskQuantizer(torch.nn.Module):

    def __init__(self, patch_size):
        super().__init__()
        self.filter = create_patch_quantizer(patch_size)

    def forward(self, x):
        return self.filter(x)


def create_patch_quantizer(patch_size):
    """Create a patch quantization filter for the given patch size."""
    patch_quant_filter = torch.nn.Conv2d(
        1, 1, patch_size, stride=patch_size, bias=False
    )
    patch_quant_filter.weight.data.fill_(1.0 / (patch_size * patch_size))
    return patch_quant_filter


def resize_image_for_patches(image, image_size, patch_size):
    """Resize image to dimensions divisible by patch size."""
    w, h = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(
        TF.resize(image, (h_patches * patch_size, w_patches * patch_size))
    )
