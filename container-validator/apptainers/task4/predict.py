#!/usr/bin/env python3
"""
FOMO26 Challenge - Task 4: Trigeminal Neuralgia Multiclass Segmentation
"""

import os
import shutil
import argparse
import subprocess

from pathlib import Path

import torch
import nibabel as nib

MODEL_DIR = Path(os.environ.get("MED_ADAPT_MODEL_DIR", "/app/models"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="FOMO26 Task 4 Trigeminal Multiclass Segmentation"
    )
    parser.add_argument(
        "--t2", type=str, required=True, help="Path to T2-weighted image"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save segmentation NIfTI"
    )
    return parser.parse_args()


def predict_segmentation(args):
    """
    Generate multiclass segmentation mask from T2w.

    Returns:
        tuple: (segmentation_mask, reference_image)
            - segmentation_mask: int array with values in {0, 1, 2}
            - reference_image: nibabel image used to copy affine/header
    """
    temp_dir = Path("temp_images")
    temp_dir.mkdir(parents=True, exist_ok=True)
    if args.t2 and Path(args.t2).exists():
        t2_img = nib.load(args.t2)
        nib.save(t2_img, temp_dir / "image_0000.nii.gz")

    subprocess.run(
        [
            "nnUNetv2_predict_from_modelfolder",
            "-i",
            str(temp_dir),
            "-o",
            str(temp_dir / "predictions"),
            "-m",
            f"{MODEL_DIR}",
            "-f",
            "4",
            "-device",
            "cuda" if torch.cuda.is_available() else "cpu",
            "--verbose",
        ]
    )

    segmentation_mask = nib.load(temp_dir / "predictions" / "image.nii.gz")
    nib.save(segmentation_mask, args.output)

    shutil.rmtree(temp_dir)

    return segmentation_mask


def main():
    """Main execution function."""
    args = parse_args()

    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Generate segmentation
    predict_segmentation(args)

    return 0


if __name__ == "__main__":
    exit(main())
