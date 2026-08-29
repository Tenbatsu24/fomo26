from __future__ import annotations
from pathlib import Path
from typing import Any

from result import TestResult


def check_output_is_numpy_file(
    output_path: Path, subject_id: str
) -> tuple[TestResult, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        r = TestResult(
            "output_is_numpy_file",
            False,
            f"Missing dependency: {exc} — run: pip install numpy",
            subject_id,
        )
        return r, None

    try:
        arr = np.load(str(output_path), allow_pickle=False)
        return (
            TestResult(
                "output_is_numpy_file",
                True,
                f"dtype={arr.dtype}, shape={arr.shape}",
                subject_id,
            ),
            arr,
        )
    except Exception as exc:
        return TestResult(
            "output_is_numpy_file", False, f"np.load failed: {exc}", subject_id
        ), None


def check_output_is_float_array(arr: Any, subject_id: str) -> TestResult:
    import numpy as np

    ok = np.issubdtype(arr.dtype, np.floating)
    return TestResult(
        "output_is_float_array",
        ok,
        f"dtype={arr.dtype}"
        if ok
        else f"Expected floating-point dtype, got {arr.dtype}",
        subject_id,
    )


def check_output_is_1d_vector(arr: Any, subject_id: str) -> TestResult:
    import numpy as np

    squeezed = np.squeeze(arr)
    ok = squeezed.ndim == 1
    return TestResult(
        "output_is_1d_vector",
        ok,
        f"1-D vector with {squeezed.shape[0]} elements"
        if ok
        else f"Expected 1-D array after squeezing, got shape {squeezed.shape}",
        subject_id,
    )


def check_output_numpy_is_finite(arr: Any, subject_id: str) -> TestResult:
    import numpy as np

    ok = bool(np.all(np.isfinite(arr)))
    return TestResult(
        "output_numpy_is_finite",
        ok,
        "All values finite" if ok else "Array contains NaN or Inf values",
        subject_id,
    )


def check_embedding_dim_consistency(outputs: dict[str, Path]) -> TestResult:
    import numpy as np

    dims: dict[str, int] = {}
    for sid, path in outputs.items():
        try:
            arr = np.load(str(path), allow_pickle=False)
            dims[sid] = int(np.squeeze(arr).shape[0])
        except Exception as exc:
            return TestResult(
                "embedding_dim_consistency",
                False,
                f"Could not load embedding for {sid!r}: {exc}",
            )

    unique_dims = set(dims.values())
    if len(unique_dims) == 1:
        dim = next(iter(unique_dims))
        return TestResult(
            "embedding_dim_consistency",
            True,
            f"All {len(dims)} embeddings have consistent dimension {dim}",
        )
    detail = ", ".join(f"{s}={d}" for s, d in sorted(dims.items()))
    return TestResult(
        "embedding_dim_consistency",
        False,
        f"Inconsistent embedding dimensions across subjects: {detail}",
    )
