import random

from collections import defaultdict

from typing import Optional

from torch.utils.data import Sampler


class UniformModalitySampler(Sampler):
    """
    Samples modalities uniformly, then samples an image uniformly
    from the selected modality.

    Example:
        dataset = OpenNeuroDataset(...)

        sampler = UniformModalitySampler(
            dataset.df["modality"].tolist(),
            seed=42,
        )

        loader = DataLoader(
            dataset,
            batch_size=8,
            sampler=sampler,
            num_workers=8,
        )
    """

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
