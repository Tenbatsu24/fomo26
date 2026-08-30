#!/usr/bin/env python3
"""
FOMO25 Challenge - Task 2: Binary Segmentation
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FOMO25 Task 2 Binary Segmentation")

    # Input paths for each modality
    parser.add_argument("--flair", type=str, help="Path to T2 FLAIR image")
    parser.add_argument("--dwi", type=str, help="Path to DWI b1000 image")
    parser.add_argument("--t2s", type=str, help="Path to T2* image (optional)")
    parser.add_argument("--swi", type=str, help="Path to SWI image (optional)")

    # Output path for segmentation mask
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save segmentation NIfTI"
    )

    return parser.parse_args()


def predict_segmentation(args):
    """
    Generate binary segmentation mask based on the provided modalities.

    Returns:
        tuple: (segmentation_mask, reference_image) where:
            - segmentation_mask: numpy array with binary mask (0 or 1)
            - reference_image: nibabel image object for metadata
    """
    #########################################################################
    # PLACEHOLDER: ADD YOUR SEGMENTATION INFERENCE CODE HERE
    #########################################################################
    #
    # Available image paths:
    #   - args.flair: T2 FLAIR image path
    #   - args.dwi: DWI b1000 image path
    #   - args.t2s: T2* image path (may be None)
    #   - args.swi: SWI image path (may be None)

    # Example steps you might implement:
    #   1. Load the images you need (not all are required)
    # save the images with _0000, _0001, _0002
    # _0000 = dwi, _0001 = flair, _0002 = t2s or swi
    # copy and save to {current_dir}/temp_images
    temp_dir = Path("temp_images")
    temp_dir.mkdir(parents=True, exist_ok=True)
    if args.dwi and Path(args.dwi).exists():
        dwi_img = nib.load(args.dwi)
        nib.save(dwi_img, temp_dir / "image_0000.nii.gz")
    if args.flair and Path(args.flair).exists():
        flair_img = nib.load(args.flair)
        nib.save(flair_img, temp_dir / "image_0001.nii.gz")
    if args.swi and Path(args.swi).exists():
        swi_img = nib.load(args.swi)
        nib.save(swi_img, temp_dir / "image_0002.nii.gz")
    elif args.t2s and Path(args.t2s).exists():
        t2s_img = nib.load(args.t2s)
        nib.save(t2s_img, temp_dir / "image_0002.nii.gz")
    #   2. Preprocess the images (normalize, resample, register, etc.)
    #   3. Load your trained segmentation model
    #   4. Run inference to get predictions
    # For example, if using a command line tool:
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
            "3",
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
