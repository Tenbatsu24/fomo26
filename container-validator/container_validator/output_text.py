from __future__ import annotations
import math
from pathlib import Path

from result import TestResult


def check_output_is_plain_text(output_path: Path, subject_id: str) -> TestResult:
    try:
        output_path.read_bytes().decode("utf-8")
        return TestResult("output_is_plain_text", True, "Valid UTF-8", subject_id)
    except UnicodeDecodeError as exc:
        return TestResult(
            "output_is_plain_text", False, f"Not valid UTF-8: {exc}", subject_id
        )


def check_output_is_single_float(output_path: Path, subject_id: str) -> TestResult:
    content = output_path.read_text(encoding="utf-8").strip()
    tokens = content.split()
    if len(tokens) != 1:
        return TestResult(
            "output_is_single_float",
            False,
            f"Expected exactly 1 token, got {len(tokens)}: {content[:100]!r}",
            subject_id,
        )
    try:
        float(tokens[0])
        return TestResult(
            "output_is_single_float", True, f"Value: {tokens[0]}", subject_id
        )
    except ValueError:
        return TestResult(
            "output_is_single_float",
            False,
            f"{tokens[0]!r} is not a valid float",
            subject_id,
        )


def check_output_text_is_finite(output_path: Path, subject_id: str) -> TestResult:
    value = float(output_path.read_text(encoding="utf-8").strip())
    ok = math.isfinite(value)
    return TestResult(
        "output_text_is_finite",
        ok,
        f"Value {value} is finite" if ok else f"Value {value} is NaN or Inf",
        subject_id,
    )


def check_output_probability_in_range(
    output_path: Path,
    subject_id: str,
    low: float,
    high: float,
) -> TestResult:
    value = float(output_path.read_text(encoding="utf-8").strip())
    ok = low <= value <= high
    return TestResult(
        "output_probability_in_range",
        ok,
        f"Value {value} in [{low}, {high}]"
        if ok
        else f"Value {value} is outside the valid range [{low}, {high}]",
        subject_id,
    )
