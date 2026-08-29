from __future__ import annotations
from pathlib import Path
from typing import Any

from result import TestResult


def check_output_is_nifti(output_path: Path, subject_id: str) -> tuple[TestResult, Any]:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        r = TestResult(
            "output_is_nifti",
            False,
            f"Missing dependency: {exc} — run: pip install nibabel",
            subject_id,
        )
        return r, None

    try:
        img = nib.load(str(output_path))
    except FileNotFoundError:
        return TestResult(
            "output_is_nifti", False, f"No output file at {output_path}", subject_id
        ), None
    except Exception as exc:
        return TestResult(
            "output_is_nifti", False, f"nibabel.load failed: {exc}", subject_id
        ), None

    if not isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
        return (
            TestResult(
                "output_is_nifti",
                False,
                f"Loaded object is {type(img).__name__}, not a NIfTI image",
                subject_id,
            ),
            None,
        )

    affine = img.affine
    if np.all(affine == 0):
        return (
            TestResult(
                "output_is_nifti",
                False,
                "NIfTI affine matrix is all zeros (degenerate geometry)",
                subject_id,
            ),
            None,
        )
    if not np.all(np.isfinite(affine)):
        return (
            TestResult(
                "output_is_nifti",
                False,
                "NIfTI affine matrix contains NaN or Inf values",
                subject_id,
            ),
            None,
        )

    return TestResult(
        "output_is_nifti", True, f"Valid NIfTI, shape {img.shape}", subject_id
    ), img


def check_output_is_3d(img: Any, subject_id: str) -> TestResult:
    shape = img.shape
    ok = len(shape) == 3
    return TestResult(
        "output_is_3d",
        ok,
        f"Shape {shape}" if ok else f"Expected 3D array, got shape {shape}",
        subject_id,
    )


def check_output_shape_matches_any_input(
    img: Any,
    subject_id: str,
    input_paths: dict[str, Path],
) -> TestResult:
    import nibabel as nib

    out_shape = tuple(img.shape[:3])
    loaded_shapes: list[tuple] = []
    load_errors: list[str] = []

    for key, path in input_paths.items():
        try:
            ref = nib.load(str(path))
            loaded_shapes.append(tuple(ref.shape[:3]))
        except Exception as exc:
            load_errors.append(f"{key}: {exc}")

    if not loaded_shapes:
        return TestResult(
            "output_shape_matches_any_input",
            False,
            "Could not load any reference input for shape comparison "
            f"(infrastructure error, not a container failure): {'; '.join(load_errors)}",
            subject_id,
        )
    ok = out_shape in loaded_shapes
    return TestResult(
        "output_shape_matches_any_input",
        ok,
        f"Output shape {out_shape} matches an input shape"
        if ok
        else f"Output shape {out_shape} does not match any input shape {loaded_shapes}",
        subject_id,
    )


def check_output_dtype_is_integer(img: Any, subject_id: str) -> TestResult:
    import numpy as np

    data = np.asanyarray(img.dataobj)
    if np.issubdtype(data.dtype, np.integer):
        return TestResult(
            "output_dtype_is_integer", True, f"Integer dtype: {data.dtype}", subject_id
        )
    if np.issubdtype(data.dtype, np.floating):
        if np.array_equal(data, np.round(data)):
            return TestResult(
                "output_dtype_is_integer",
                True,
                f"Float dtype {data.dtype} but integer-valued",
                subject_id,
            )
        return TestResult(
            "output_dtype_is_integer",
            False,
            f"Float dtype {data.dtype} contains non-integer label values",
            subject_id,
        )
    return TestResult(
        "output_dtype_is_integer",
        False,
        f"Dtype {data.dtype} is neither integer nor float",
        subject_id,
    )


def check_output_labels_in_range(
    img: Any, subject_id: str, max_label: int
) -> TestResult:
    import numpy as np

    data = np.asanyarray(img.dataobj)
    if data.size == 0:
        return TestResult(
            "output_labels_in_range", False, "Output volume is empty", subject_id
        )
    obs_min, obs_max = int(np.min(data)), int(np.max(data))
    if obs_min < 0:
        return TestResult(
            "output_labels_in_range",
            False,
            f"Negative label {obs_min} found; labels must be >= 0",
            subject_id,
        )
    if obs_max > max_label:
        return TestResult(
            "output_labels_in_range",
            False,
            f"Max label {obs_max} exceeds allowed max_label={max_label}",
            subject_id,
        )
    return TestResult(
        "output_labels_in_range",
        True,
        f"Labels in [0, {max_label}], observed [{obs_min}, {obs_max}]",
        subject_id,
    )


def check_output_nifti_is_finite(img: Any, subject_id: str) -> TestResult:
    import numpy as np

    data = np.asanyarray(img.dataobj)
    if not np.issubdtype(data.dtype, np.floating):
        return TestResult(
            "output_nifti_is_finite",
            True,
            "Integer dtype — skipping finite check",
            subject_id,
        )
    ok = bool(np.all(np.isfinite(data)))
    return TestResult(
        "output_nifti_is_finite",
        ok,
        "All values finite" if ok else "NIfTI volume contains NaN or Inf values",
        subject_id,
    )
