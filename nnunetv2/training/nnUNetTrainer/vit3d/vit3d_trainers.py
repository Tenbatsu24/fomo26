from abc import abstractmethod
from pathlib import Path

import torch

from torch import nn, autocast

from med_adapt.utils import get_models_path, mark_trainable
from med_adapt.models.extended.volume import vitv2_a_3d_small

from nnunetv2.training.nnUNetTrainer.variants.lr_schedule.nnUNetTrainer_warmup import (
    nnUNetTrainer_warmup,
)
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.lr_scheduler.warmup import (
    Lin_incr_LRScheduler,
    PolyLRScheduler_offset,
)
from nnunetv2.utilities.helpers import empty_cache, dummy_context


class AbstractViT3DAdaption(nnUNetTrainer_warmup):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 2e-3
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.ckpt_path = None
        self.warmup_duration_whole_net = 15  # lin increase whole network
        self.num_epochs = 150
        self.num_iterations_per_epoch = 50

    @staticmethod
    @abstractmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        raise NotImplementedError()

    def _do_i_compile(self):
        return False

    def configure_optimizers(self, stage: str = "warmup_all"):
        assert stage in ["warmup_all", "train"]

        if self.training_stage is None:
            self._load_pretrained()

        mark_trainable(
            self.network,
            additional_keys=self.network.additional_trainable(),
        )

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        # Get parameters that require gradients
        if isinstance(self.network, DDP):
            model = self.network.module
        else:
            model = self.network

        patch_embed_params = []
        other_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("patch_embed"):
                patch_embed_params.append(p)
                print(f"{name:60s} {tuple(p.shape)!s:20s} lr={self.initial_lr * 0.1}")
            else:
                other_params.append(p)
                print(f"{name:60s} {tuple(p.shape)!s:20s} lr={self.initial_lr}")

        params = [
            {
                "params": other_params,
                "lr": self.initial_lr,
            },
            {
                "params": patch_embed_params,
                "lr": self.initial_lr * 0.1,
            },
        ]

        if stage == "warmup_all":
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.AdamW(
                params,
                self.initial_lr,
                weight_decay=self.weight_decay,
                amsgrad=False,
                betas=(0.9, 0.98),
                fused=True,
            )
            lr_scheduler = Lin_incr_LRScheduler(
                optimizer, self.initial_lr, self.warmup_duration_whole_net
            )
            self.print_to_log_file(
                f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}"
            )
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == "warmup_all":
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful direction
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    params,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False,
                    betas=(0.9, 0.98),
                    fused=True,
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer,
                self.initial_lr,
                self.num_epochs,
                self.warmup_duration_whole_net,
            )
            self.print_to_log_file(
                f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}"
            )
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def _load_pretrained(self) -> None:
        """Load a pretrained checkpoint if configured."""
        if self.ckpt_path is None:
            print("[TemplateTrainer] No checkpoint specified.")
            return

        ckpt_path = Path(get_models_path()) / self.ckpt_path
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

            state_dict = {
                k.replace("model.", ""): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }

        if to_ignore := self.network.do_not_load():
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if all(ig not in k for ig in to_ignore)
            }
        missing, unexpected = self.network.load_state_dict(state_dict, strict=False)

        print(
            f"[TemplateTrainer] Loaded checkpoint from {ckpt_path}. Missing: {missing}, Unexpected: {unexpected}",
        )


class UNetViT3DSmallTrainer(AbstractViT3DAdaption):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.ckpt_path = "small/296_518/last.ckpt"

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:

        model = vitv2_a_3d_small(
            n_modalities=num_input_channels,
            task="segmentation",
            classes=num_output_channels,
        )

        return model
