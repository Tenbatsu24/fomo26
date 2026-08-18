import random

from collections import defaultdict

from typing import Optional

import torch.distributed as dist
from torch.utils.data import Sampler


class UniformModalitySampler(Sampler):

    def __init__(
        self,
        modalities,
        seed: int = 42,
        num_samples: Optional[int] = None,
    ):
        self.rng = random.Random(seed)

        self.modality_to_indices = defaultdict(list)

        for idx, modality in enumerate(modalities):
            self.modality_to_indices[modality].append(idx)

        self.modalities = sorted(self.modality_to_indices.keys())

        self.num_samples = len(modalities) if num_samples is None else num_samples

    def __iter__(self):
        for _ in range(self.num_samples):

            modality = self.rng.choice(self.modalities)

            yield self.rng.choice(self.modality_to_indices[modality])

    def __len__(self):
        return self.num_samples


class RandomSampler(Sampler):
    def __init__(
        self,
        dataset,
        num_samples: int,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.num_samples = num_samples
        self.seed = seed

        self.dataset_size = len(dataset)

        world_size = dist.get_world_size() if dist.is_initialized() else 1

        if num_samples * world_size > self.dataset_size:
            raise ValueError(
                f"Requested {num_samples * world_size} total samples "
                f"from a dataset of size {self.dataset_size}."
            )

    def __iter__(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        rng = random.Random(self.seed + rank)

        sampled = rng.sample(
            range(self.dataset_size),
            self.num_samples * world_size,
        )

        yield from sampled[rank::world_size]

    def __len__(self):
        return self.num_samples
