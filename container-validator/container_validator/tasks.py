from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class InputSpec:
    key: str
    arg: str
    formats: tuple[str, ...]


@dataclass(frozen=True)
class RequiredOneOf:
    group_key: str
    options: dict[str, InputSpec]


@dataclass(frozen=True)
class OutputSpec:
    arg: str
    filename: str
    format: str  # "txt" | "nifti" | "numpy"
    value_range: tuple[float, float] | None = None
    max_label: int | None = None
    same_shape_as_any_input: bool = False


@dataclass(frozen=True)
class TaskDef:
    display_name: str
    inputs: tuple[InputSpec | RequiredOneOf, ...]
    output: OutputSpec
    suite: str


def _nii(key: str, arg: str) -> InputSpec:
    return InputSpec(key, arg, (".nii", ".nii.gz"))


_T2S_OR_SWI = RequiredOneOf(
    group_key="t2s_or_swi",
    options={
        "t2s": _nii("t2s", "--t2s"),
        "swi": _nii("swi", "--swi"),
    },
)

TASKS: dict[str, TaskDef] = {
    "task1": TaskDef(
        display_name="Task 1 — Stroke Outcome Classification (Probability)",
        inputs=(
            _nii("flair", "--flair"),
            _nii("dwi", "--dwi"),
            _nii("adc", "--adc"),
            _T2S_OR_SWI,
        ),
        output=OutputSpec("--output", "output.txt", "txt", value_range=(0.0, 1.0)),
        suite="classification_scalar_probability",
    ),
    "task2": TaskDef(
        display_name="Task 2 — Binary Lesion Segmentation",
        inputs=(
            _nii("flair", "--flair"),
            _nii("dwi", "--dwi"),
            _T2S_OR_SWI,
        ),
        output=OutputSpec(
            "--output",
            "output.nii.gz",
            "nifti",
            max_label=1,
            same_shape_as_any_input=True,
        ),
        suite="binary_segmentation_nifti",
    ),
    "task3": TaskDef(
        display_name="Task 3 — Lesion Volume Regression (Scalar)",
        inputs=(_nii("t1", "--t1"),),
        output=OutputSpec("--output", "output.txt", "txt"),
        suite="regression_scalar",
    ),
    "task4": TaskDef(
        display_name="Task 4 — Multiclass Tissue Segmentation",
        inputs=(_nii("t2", "--t2"),),
        output=OutputSpec(
            "--output",
            "output.nii.gz",
            "nifti",
            max_label=2,
            same_shape_as_any_input=True,
        ),
        suite="multiclass_segmentation_nifti",
    ),
    "task5": TaskDef(
        display_name="Task 5 — Hemorrhagic Transformation Classification (Probability)",
        inputs=(_nii("t1", "--t1"),),
        output=OutputSpec("--output", "output.txt", "txt", value_range=(0.0, 1.0)),
        suite="classification_scalar_probability",
    ),
    "task6_and_7": TaskDef(
        display_name="Tasks 6 and 7 — Fairness & Linear Probing Embeddings",
        inputs=(_nii("input", "--input"),),
        output=OutputSpec("--output", "output.npy", "numpy"),
        suite="linear_probing_embeddings",
    ),
}
