from __future__ import annotations
import os
from pathlib import Path

from result import TestResult
from tasks import InputSpec, RequiredOneOf, TaskDef


def check_container_file_exists(sif_path: Path) -> TestResult:
    ok = sif_path.exists()
    return TestResult(
        "container_file_exists",
        ok,
        str(sif_path) if ok else f"No file found at {sif_path}",
    )


def check_container_file_is_readable(sif_path: Path) -> TestResult:
    if not sif_path.is_file():
        return TestResult(
            "container_file_is_readable", False, f"{sif_path} is not a regular file"
        )
    if not os.access(sif_path, os.R_OK):
        return TestResult(
            "container_file_is_readable", False, f"{sif_path} is not readable"
        )
    return TestResult("container_file_is_readable", True, "Readable regular file")


def check_container_has_valid_extension(sif_path: Path) -> TestResult:
    ok = sif_path.suffix.lower() == ".sif"
    return TestResult(
        "container_has_valid_extension",
        ok,
        ".sif extension OK"
        if ok
        else f"Expected .sif extension, got {sif_path.suffix!r}",
    )


def check_required_inputs_are_mapped(task: TaskDef, subjects: list[dict]) -> TestResult:
    if not subjects:
        return TestResult(
            "required_inputs_are_mapped", False, "No subjects found in manifest"
        )

    missing: list[str] = []
    for subject in subjects:
        sid = subject["subject_id"]
        provided = subject["inputs"]
        for spec in task.inputs:
            if isinstance(spec, InputSpec):
                if spec.key not in provided:
                    missing.append(f"{sid}: required input {spec.key!r} is missing")
            elif isinstance(spec, RequiredOneOf):
                if not any(k in provided for k in spec.options):
                    opts = ", ".join(repr(k) for k in spec.options)
                    missing.append(f"{sid}: needs one of [{opts}] but none provided")

    if missing:
        return TestResult("required_inputs_are_mapped", False, "; ".join(missing))
    return TestResult(
        "required_inputs_are_mapped",
        True,
        f"All {len(subjects)} subject(s) have required inputs",
    )
