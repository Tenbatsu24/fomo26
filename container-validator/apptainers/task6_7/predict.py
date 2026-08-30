#!/usr/bin/env python3
"""
FOMO26 Challenge - Task 6: Linear Probing on Frozen Pretrained Embeddings
"""

import argparse

from pathlib import Path

import numpy as np

from med_adapt.inference import predict_case


def parse_args():
    parser = argparse.ArgumentParser(description="FOMO26 Task 6 Linear Probing")
    parser.add_argument("--input", type=Path, required=True, help="Path to input NIfTI")
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to save embeddings .npy"
    )
    return parser.parse_args()


def predict(args):
    """
    Compute frozen pretrained embeddings for the input volume.

    Returns:
        np.ndarray: (M) float32 embedding matrix
    """

    embeddings = predict_case(args.input)

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
