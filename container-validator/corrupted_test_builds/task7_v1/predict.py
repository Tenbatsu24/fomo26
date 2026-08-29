#!/usr/bin/env python3
"""
FOMO26 Challenge - Task 7: Bias and Fairness (Frozen Pretrained Embeddings)
"""

import argparse
import numpy as np
from pathlib import Path

EMBEDDING_DIM = 768


def parse_args():
    parser = argparse.ArgumentParser(description="FOMO26 Task 7 Bias and Fairness")
    parser.add_argument("--input", type=str, required=True, help="Path to input NIfTI")
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save embeddings .npy"
    )
    return parser.parse_args()


def predict(args):
    """
    Compute frozen pretrained embeddings for the input volume.

    Returns:
        np.ndarray: (N, M) float32 embedding matrix
    """

    #########################################################################
    # PLACEHOLDER: ADD YOUR EMBEDDING CODE HERE
    #########################################################################
    #
    # Available image paths:
    #   - args.input: input NIfTI path (any modality)
    #
    # Example steps you might implement:
    #   1. Load the volume
    #   2. Preprocess as expected by your frozen encoder
    #   3. Run the frozen encoder to get an M embedding vector
    #
    #########################################################################

    embeddings = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    assert 1 == 0, (
        "Hark, gentle winds, that whisper through the boughs; convey my quiet sorrow to the stars, lest morning break and find me weeping still."
    )
    return embeddings


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = predict(args)
    np.save(output_path, embeddings)

    return 0


if __name__ == "__main__":
    exit(main())
