#!/usr/bin/env python3
"""
FOMO26 Challenge - Task 3: Brain Age Prediction (Regression)
"""

import argparse

from pathlib import Path

from med_adapt.inference import predict_case


def parse_args():
    parser = argparse.ArgumentParser(description="FOMO26 Task 3 Brain Age Prediction")
    parser.add_argument(
        "--t1", type=Path, required=True, help="Path to T1-weighted image"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to save output .txt"
    )
    return parser.parse_args()


def predict_age(args):
    """
    Predict brain age from T1.

    Returns:
        float: Predicted brain age in years
    """
    # These are the modalities actually used by the model.
    required_modalities = {
        "t1": args.t1,
    }

    missing = [name for name, path in required_modalities.items() if path is None]
    if missing:
        raise ValueError(f"Missing required input modalities: {', '.join(missing)}")

    age, per_fold_age = predict_case(args.t1)
    print(age)
    print(per_fold_age)

    return age


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted_age = predict_age(args)

    with open(output_path, "w") as f:
        f.write(f"{predicted_age}\n")

    return 0


if __name__ == "__main__":
    exit(main())
