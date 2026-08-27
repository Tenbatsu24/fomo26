#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FOMO25 Task 1 - Infarct Classification"
    )

    # Input paths for each modality
    parser.add_argument("--flair", type=str, help="Path to T2 FLAIR image")
    parser.add_argument("--adc", type=str, help="Path to ADC image")
    parser.add_argument("--dwi", type=str, help="Path to DWI b1000 image")
    parser.add_argument("--t2s", type=str, help="Path to T2* image (optional)")
    parser.add_argument("--swi", type=str, help="Path to SWI image (optional)")

    # Output path for predictions
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save output .txt file"
    )

    return parser.parse_args()


def predict(args):
    probability = 0.65  # Example probability, should be between 0 and 1

    return probability


def main():
    """Main execution function."""
    args = parse_args()

    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Get prediction probability
    probability = predict(args)

    # Save probability in a text file called <subject_id>.txt
    output_file = Path(args.output)
    with open(output_file, "w") as f:
        f.write(f"{probability:.3f}")

    return 0


if __name__ == "__main__":
    exit(main())
