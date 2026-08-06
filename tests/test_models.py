import torch

from med_adapt.models import vitv2_a_2d_tiny, vitv2_a_3d_tiny

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestViTv2Adaption2D:
    def test_classification(self):
        model = vitv2_a_2d_tiny(med_in_channels=1, task="classification", classes=2).to(
            DEVICE
        )
        x = torch.randn(2, 1, 28, 28, 8, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 2)

    def test_regression(self):
        model = vitv2_a_2d_tiny(med_in_channels=1, task="regression", classes=1).to(
            DEVICE
        )
        x = torch.randn(2, 1, 28, 28, 8, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 1)

    def test_segmentation(self):
        model = vitv2_a_2d_tiny(med_in_channels=1, task="segmentation", classes=3).to(
            DEVICE
        )
        x = torch.randn(2, 1, 28, 28, 8, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 3, 28, 28, 8)

    def test_none(self):
        model = vitv2_a_2d_tiny(med_in_channels=1, task="none", classes=1).to(DEVICE)
        x = torch.randn(2, 1, 28, 28, 8, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 192)

    def test_task_token_beginning(self):
        model = vitv2_a_2d_tiny(
            med_in_channels=1,
            task="classification",
            classes=2,
            task_token=True,
            task_token_insertion="beginning",
        ).to(DEVICE)
        x = torch.randn(2, 1, 28, 28, 8, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 2)
        # Verify task tokens are trainable
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert any("task_tokens" in n for n in trainable)

    def test_task_token_middle(self):
        model = vitv2_a_2d_tiny(
            med_in_channels=1,
            task="classification",
            classes=2,
            task_token=True,
            task_token_insertion="middle",
            task_token_block=6,
        ).to(DEVICE)
        x = torch.randn(2, 1, 28, 28, 8, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 2)


class TestViTv2Adaption3D:
    def test_classification(self):
        model = vitv2_a_3d_tiny(
            volume_size=(28, 28, 8),
            volume_patch_size=(14, 14, 2),
            med_in_channels=1,
            task="classification",
            classes=2,
        ).to(DEVICE)
        x = torch.randn(2, 1, 28, 28, 8, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 2)

    def test_regression(self):
        model = vitv2_a_3d_tiny(
            volume_size=(28, 28, 8),
            volume_patch_size=(14, 14, 2),
            med_in_channels=1,
            task="regression",
            classes=1,
        ).to(DEVICE)
        x = torch.randn(2, 1, 28, 28, 8, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 1)

    def test_segmentation(self):
        model = vitv2_a_3d_tiny(
            volume_size=(28, 28, 8),
            volume_patch_size=(14, 14, 2),
            med_in_channels=1,
            task="segmentation",
            classes=3,
        ).to(DEVICE)
        x = torch.randn(2, 1, 28, 28, 8, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 3, 28, 28, 8)

    def test_none(self):
        model = vitv2_a_3d_tiny(
            volume_size=(28, 28, 8),
            volume_patch_size=(14, 14, 2),
            med_in_channels=1,
            task="none",
            classes=1,
        ).to(DEVICE)
        x = torch.randn(2, 1, 28, 28, 8, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 192)

    def test_task_token_beginning(self):
        model = vitv2_a_3d_tiny(
            volume_size=(28, 28, 8),
            volume_patch_size=(14, 14, 2),
            med_in_channels=1,
            task="classification",
            classes=2,
            task_token=True,
            task_token_insertion="beginning",
        ).to(DEVICE)
        x = torch.randn(2, 1, 28, 28, 8, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 2)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert any("task_tokens" in n for n in trainable)

    def test_task_token_middle(self):
        model = vitv2_a_3d_tiny(
            volume_size=(28, 28, 8),
            volume_patch_size=(14, 14, 2),
            med_in_channels=1,
            task="classification",
            classes=2,
            task_token=True,
            task_token_insertion="middle",
            task_token_block=6,
        ).to(DEVICE)
        x = torch.randn(2, 1, 28, 28, 8, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 2)
