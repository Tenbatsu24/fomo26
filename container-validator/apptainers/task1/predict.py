#!/usr/bin/env python3
import argparse

from pathlib import Path

from med_adapt.inference import predict_case


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FOMO25 Task 1 - Infarct Classification"
    )

    # Input paths for each modality
    parser.add_argument("--flair", type=Path, help="Path to T2 FLAIR image")
    parser.add_argument("--adc", type=Path, help="Path to ADC image")
    parser.add_argument("--dwi", type=Path, help="Path to DWI image")
    parser.add_argument("--t2s", type=Path, help="Path to T2* image (optional)")
    parser.add_argument("--swi", type=Path, help="Path to SWI image (optional)")

    # Output path for predictions
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to save output .txt file"
    )

    return parser.parse_args()


def predict(args):
    """
    Predict infarct probability based on the provided modalities.

    Returns:
        float: Probability of positive class (infarct presence) between 0 and 1
    """

    # These are the modalities actually used by the model.
    required_modalities = {
        "flair": args.flair,
        "adc": args.adc,
        "dwi": args.dwi,
    }

    missing = [name for name, path in required_modalities.items() if path is None]
    if missing:
        raise ValueError(f"Missing required input modalities: {', '.join(missing)}")

    probability, per_fold_probability = predict_case(args.flair, args.adc, args.dwi)
    print(probability)
    print(per_fold_probability)

    return probability


def main():
    """Main execution function."""
    args = parse_args()

    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Get prediction probability
    probability = predict(args)

    # Save probability in a text file called <subject_id>.txt
    subject_id = Path(args.output).stem  # Extract subject ID from output path
    output_file = Path(args.output).parent / f"{subject_id}.txt"
    with open(output_file, "w") as f:
        f.write(f"{probability}")

    return 0


if __name__ == "__main__":
    exit(main())
