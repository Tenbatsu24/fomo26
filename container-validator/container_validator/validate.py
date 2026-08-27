#!/usr/bin/env python3
"""
FOMO 2026 Container Validator

Validate your Singularity/Apptainer container locally before submitting
to the FOMO 2026 Synapse challenge. Runs the same checks as the official
evaluation pipeline — no Synapse account or internet connection needed.

Usage:
    python validate.py --task task1 --sif /path/to/container.sif
    python validate.py --task task6 --sif /path/to/container.sif --no-gpu
    python validate.py --list-tasks

See data/manifest.yaml for how to configure your local test subjects.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

# ─── Console output ────────────────────────────────────────────────────────

_G = "\033[92m"
_R = "\033[91m"
_C = "\033[96m"
_B = "\033[1m"
_X = "\033[0m"

PASS_TAG = f"{_G}[PASS]{_X}"
FAIL_TAG = f"{_R}[FAIL]{_X}"
INFO_TAG = f"{_C}[INFO]{_X}"


def _pr(result, indent: int = 0) -> None:
    tag = PASS_TAG if result.passed else FAIL_TAG
    pad = "  " * indent
    subj = f" subject={result.subject_id!r}" if result.subject_id else ""
    msg = f" — {result.message}" if result.message else ""
    print(f"{pad}{tag} {result.name}{subj}{msg}")


def _abort(reason: str) -> None:
    print(f"\n  {FAIL_TAG} {reason} — stopping.\n")


def _section(title: str) -> None:
    print(f"{_B}{title}{_X}")


# ─── Manifest loading ──────────────────────────────────────────────────────


def _load_subjects(manifest_path: Path, task_id: str) -> list[dict]:
    try:
        import yaml
    except ImportError:
        print(f"{FAIL_TAG} PyYAML is required — run: pip install pyyaml")
        sys.exit(1)

    if not manifest_path.exists():
        print(f"{FAIL_TAG} Manifest not found: {manifest_path}")
        print("       Edit data/manifest.yaml with your local test data.")
        sys.exit(1)

    with open(manifest_path) as fh:
        raw = yaml.safe_load(fh) or {}

    if task_id not in raw:
        available = list(raw.keys())
        print(
            f"{FAIL_TAG} Task {task_id!r} not in manifest. Available: {available or '(none)'}"
        )
        print(f"       Add {task_id!r} subjects to data/manifest.yaml.")
        sys.exit(1)

    base = manifest_path.parent
    subjects = []
    for subject_id, spec in raw[task_id].items():
        inputs = {
            key: (base / rel_path).resolve()
            for key, rel_path in spec.get("inputs", {}).items()
        }
        subjects.append({"subject_id": str(subject_id), "inputs": inputs})

    if not subjects:
        print(f"{FAIL_TAG} No subjects defined for {task_id} in manifest.")
        sys.exit(1)

    return subjects


# ─── Predict argument builder ──────────────────────────────────────────────


def _output_ext(filename: str) -> str:
    return ".nii.gz" if filename.lower().endswith(".nii.gz") else Path(filename).suffix


def _build_predict_args(task, subject: dict, out_filename: str) -> list[str]:
    from tasks import InputSpec, RequiredOneOf

    args: list[str] = []
    for spec in task.inputs:
        if isinstance(spec, InputSpec):
            key = spec.key
            if key not in subject["inputs"]:
                raise ValueError(
                    f"Subject {subject['subject_id']!r} missing required input {key!r}"
                )
            args += [spec.arg, f"/input/{subject['inputs'][key].name}"]
        elif isinstance(spec, RequiredOneOf):
            chosen = next((k for k in spec.options if k in subject["inputs"]), None)
            if chosen is None:
                opts = sorted(spec.options)
                raise ValueError(
                    f"Subject {subject['subject_id']!r} missing all alternatives for "
                    f"group {spec.group_key!r}: need one of {opts}"
                )
            chosen_spec = spec.options[chosen]
            args += [chosen_spec.arg, f"/input/{subject['inputs'][chosen].name}"]

    args += [task.output.arg, f"/output/{out_filename}"]
    return args


# ─── Directory helpers ──────────────────────────────────────────────────────


def _clear_dir(path: Path) -> None:
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


# ─── Main pipeline ──────────────────────────────────────────────────────────


def run_validation(
    task_id: str,
    sif_path: Path,
    manifest_path: Path,
    gpu: bool = True,
    executable: str = "apptainer",
    timeout: int = 900,
) -> bool:
    from result import TestResult
    import preflight
    import preinference
    import container_tests
    import output_text
    import output_nifti
    import output_numpy
    from runner import ApptainerRunner
    from tasks import TASKS

    task = TASKS[task_id]
    all_results: list[TestResult] = []
    overall_pass = True

    print(f"\n{_B}{'=' * 64}{_X}")
    print(f"{_B}  FOMO 2026 Container Validator{_X}")
    print(f"{_B}{'=' * 64}{_X}")
    print(f"{INFO_TAG} Task:      {task.display_name}")
    print(f"{INFO_TAG} Container: {sif_path}")
    print(f"{INFO_TAG} GPU:       {'enabled' if gpu else 'disabled (--no-gpu)'}")
    print()

    # ── Phase 0: Preflight ────────────────────────────────────────────────
    _section("[Phase 0] Preflight checks")
    for r in preflight.run_preflight(executable=executable, require_gpu=gpu):
        _pr(r, indent=1)
        if not r.passed:
            _abort("PREFLIGHT FAILED")
            return False
    print()

    # ── Load subjects from manifest ───────────────────────────────────────
    subjects = _load_subjects(manifest_path, task_id)
    print(f"{INFO_TAG} Loaded {len(subjects)} subject(s) from manifest\n")

    # ── Phase 1: File and input checks ───────────────────────────────────
    _section("[Phase 1] Container file and input checks")
    for r in [
        preinference.check_container_file_exists(sif_path),
        preinference.check_container_file_is_readable(sif_path),
        preinference.check_container_has_valid_extension(sif_path),
        preinference.check_required_inputs_are_mapped(task, subjects),
    ]:
        all_results.append(r)
        _pr(r, indent=1)
        if not r.passed:
            _abort("PHASE 1 FAILED")
            return False
    print()

    # ── Phases 2 and 3: Instance + inference ─────────────────────────────
    runner = ApptainerRunner(executable=executable, gpu=gpu)
    instance_name = f"fomo_val_{uuid.uuid4().hex[:8]}"
    ext = _output_ext(task.output.filename)

    with (
        tempfile.TemporaryDirectory(prefix="fomo_in_") as td_in,
        tempfile.TemporaryDirectory(prefix="fomo_out_") as td_out,
        tempfile.TemporaryDirectory(prefix="fomo_tmp_") as td_tmp,
    ):
        input_dir = Path(td_in)
        output_dir = Path(td_out)
        tmp_dir = Path(td_tmp)

        # Start Apptainer instance
        _section("[Phase 2] Container instance checks")
        print(f"  {INFO_TAG} Starting instance {instance_name!r} ...")
        start = runner.start_instance(
            sif_path, instance_name, input_dir, output_dir, tmp_dir
        )
        if not start.succeeded:
            detail = start.stderr.strip() or f"rc={start.returncode}"
            print(f"  {FAIL_TAG} container_instance_start — {detail}")
            _abort("INSTANCE START FAILED")
            return False
        print(f"  {PASS_TAG} container_instance_start — instance is running")

        try:
            # Phase 2: instance-level tests
            phase2 = [
                (
                    container_tests.check_container_can_be_executed(
                        runner, instance_name
                    ),
                    True,
                ),
                (
                    container_tests.check_container_can_access_gpu(
                        runner, instance_name
                    ),
                    gpu,
                ),
            ]
            for r, run_check in phase2:
                if not run_check:
                    continue
                all_results.append(r)
                _pr(r, indent=1)
                if not r.passed:
                    _abort("PHASE 2 FAILED")
                    return False
            print()

            # Phase 3: per-subject inference and output validation
            _section("[Phase 3] Inference and output validation")
            numpy_outputs: dict[str, Path] = {}

            for subject in subjects:
                sid = subject["subject_id"]
                print(f"\n  {_C}Subject: {sid}{_X}")

                # Reset bind-mounted directories between subjects
                _clear_dir(input_dir)
                _clear_dir(output_dir)

                # Copy this subject's inputs into the container-visible input dir
                copy_failed = False
                for key, src in subject["inputs"].items():
                    dst = input_dir / src.name
                    try:
                        shutil.copy2(src, dst)
                    except FileNotFoundError:
                        r = TestResult(
                            "prediction_runs_successfully",
                            False,
                            f"Input file not found: {src}",
                            sid,
                        )
                        all_results.append(r)
                        _pr(r, indent=2)
                        overall_pass = False
                        copy_failed = True
                        break
                if copy_failed:
                    continue

                # Build output filename: {subject_id}{ext}
                out_filename = f"{sid}{ext}"
                host_output = output_dir / out_filename

                # Build predict.py command
                try:
                    predict_args = _build_predict_args(task, subject, out_filename)
                except ValueError as exc:
                    r = TestResult(
                        "prediction_runs_successfully",
                        False,
                        f"Could not build arguments: {exc}",
                        sid,
                    )
                    all_results.append(r)
                    _pr(r, indent=2)
                    overall_pass = False
                    continue

                cmd = ["python", "/app/predict.py"] + predict_args
                print(f"    {INFO_TAG} {' '.join(cmd)}")

                # Run prediction
                run = runner.exec_instance(instance_name, cmd, timeout=timeout)

                if run.timed_out:
                    pred_ok, pred_msg = False, f"predict.py timed out after {timeout}s"
                elif run.returncode != 0:
                    snippet = run.stderr.strip()[:500]
                    pred_ok = False
                    pred_msg = f"predict.py exited with rc={run.returncode}\n      stderr: {snippet}"
                else:
                    pred_ok = True
                    pred_msg = "rc=0"
                    if run.stdout.strip():
                        pred_msg += f" — stdout: {run.stdout.strip()[:200]}"

                r_pred = TestResult(
                    "prediction_runs_successfully", pred_ok, pred_msg, sid
                )
                all_results.append(r_pred)
                _pr(r_pred, indent=2)
                if not pred_ok:
                    overall_pass = False
                    continue

                # output_file_exists
                r_exists = TestResult(
                    "output_file_exists",
                    host_output.exists(),
                    str(host_output)
                    if host_output.exists()
                    else f"predict.py produced no output at {host_output}",
                    sid,
                )
                all_results.append(r_exists)
                _pr(r_exists, indent=2)
                if not r_exists.passed:
                    overall_pass = False
                    continue

                # output_file_is_not_empty
                size = host_output.stat().st_size
                r_empty = TestResult(
                    "output_file_is_not_empty",
                    size > 0,
                    f"{size} bytes" if size > 0 else "Output file is empty",
                    sid,
                )
                all_results.append(r_empty)
                _pr(r_empty, indent=2)
                if not r_empty.passed:
                    overall_pass = False
                    continue

                # Format-specific checks
                subject_ok = _validate_output_format(
                    task,
                    subject,
                    sid,
                    host_output,
                    all_results,
                    output_text,
                    output_nifti,
                    output_numpy,
                )
                if not subject_ok:
                    overall_pass = False
                    continue

                # Collect embedding outputs for cross-subject check (task6)
                if task.suite == "linear_probing_embeddings":
                    saved = tmp_dir / f"embed_{sid}.npy"
                    shutil.copy2(host_output, saved)
                    numpy_outputs[sid] = saved

            # Cross-subject embedding dimension consistency (task6 only)
            if task.suite == "linear_probing_embeddings" and len(numpy_outputs) > 1:
                print(f"\n  {_C}Cross-subject check{_X}")
                r = output_numpy.check_embedding_dim_consistency(numpy_outputs)
                all_results.append(r)
                _pr(r, indent=2)
                if not r.passed:
                    overall_pass = False

        finally:
            print(f"\n  {INFO_TAG} Stopping instance {instance_name!r} ...")
            runner.stop_instance(instance_name)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{_B}{'=' * 64}{_X}")
    n_pass = sum(1 for r in all_results if r.passed)
    n_total = len(all_results)
    if overall_pass:
        print(
            f"{_G}{_B}  ALL {n_total} TESTS PASSED — container is ready to submit!{_X}"
        )
    else:
        n_fail = n_total - n_pass
        print(
            f"{_R}{_B}  {n_fail}/{n_total} TESTS FAILED — fix the issues above before submitting.{_X}"
        )
    print(f"{_B}{'=' * 64}{_X}\n")

    return overall_pass


def _validate_output_format(
    task,
    subject,
    sid,
    host_output,
    all_results,
    output_text,
    output_nifti,
    output_numpy,
) -> bool:
    fmt = task.output.format

    if fmt == "txt":
        for check_fn in [
            lambda: output_text.check_output_is_plain_text(host_output, sid),
            lambda: output_text.check_output_is_single_float(host_output, sid),
            lambda: output_text.check_output_text_is_finite(host_output, sid),
        ]:
            r = check_fn()
            all_results.append(r)
            _pr(r, indent=2)
            if not r.passed:
                return False

        if task.output.value_range is not None:
            low, high = task.output.value_range
            r = output_text.check_output_probability_in_range(
                host_output, sid, low, high
            )
            all_results.append(r)
            _pr(r, indent=2)
            if not r.passed:
                return False

    elif fmt == "nifti":
        r_nifti, img = output_nifti.check_output_is_nifti(host_output, sid)
        all_results.append(r_nifti)
        _pr(r_nifti, indent=2)
        if not r_nifti.passed:
            return False

        for check_fn in [
            lambda: output_nifti.check_output_is_3d(img, sid),
            lambda: output_nifti.check_output_dtype_is_integer(img, sid),
            lambda: output_nifti.check_output_nifti_is_finite(img, sid),
        ]:
            r = check_fn()
            all_results.append(r)
            _pr(r, indent=2)
            if not r.passed:
                return False

        if task.output.same_shape_as_any_input:
            r = output_nifti.check_output_shape_matches_any_input(
                img, sid, subject["inputs"]
            )
            all_results.append(r)
            _pr(r, indent=2)
            if not r.passed:
                return False

        if task.output.max_label is not None:
            r = output_nifti.check_output_labels_in_range(
                img, sid, task.output.max_label
            )
            all_results.append(r)
            _pr(r, indent=2)
            if not r.passed:
                return False

    elif fmt == "numpy":
        r_npy, arr = output_numpy.check_output_is_numpy_file(host_output, sid)
        all_results.append(r_npy)
        _pr(r_npy, indent=2)
        if not r_npy.passed:
            return False

        for check_fn in [
            lambda: output_numpy.check_output_is_float_array(arr, sid),
            lambda: output_numpy.check_output_is_1d_vector(arr, sid),
            lambda: output_numpy.check_output_numpy_is_finite(arr, sid),
        ]:
            r = check_fn()
            all_results.append(r)
            _pr(r, indent=2)
            if not r.passed:
                return False

    return True


# ─── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    from tasks import TASKS

    parser = argparse.ArgumentParser(
        prog="validate.py",
        description="Validate a FOMO 2026 Singularity container before submitting to Synapse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python validate.py --task task1 --sif /path/to/container.sif
  python validate.py --task task6 --sif /path/to/container.sif --no-gpu
  python validate.py --task task2 --sif ./my.sif --manifest /my/manifest.yaml
  python validate.py --list-tasks

exit codes:
  0 — all tests passed
  1 — one or more tests failed
""",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        metavar="TASK",
        help=f"Task to validate. Choices: {', '.join(sorted(TASKS))}",
    )
    parser.add_argument(
        "--sif",
        type=Path,
        metavar="PATH",
        help="Path to the Singularity/Apptainer .sif container file",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip GPU checks (use when testing on a machine without a GPU)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to data manifest YAML (default: data/manifest.yaml next to this script)",
    )
    parser.add_argument(
        "--apptainer",
        default="apptainer",
        metavar="EXEC",
        help="Apptainer executable name or full path (default: apptainer)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        metavar="SECONDS",
        help="Per-subject prediction timeout in seconds (default: 900)",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all supported tasks and exit",
    )
    args = parser.parse_args()

    if args.list_tasks:
        print("\nSupported tasks:")
        for tid, tdef in sorted(TASKS.items()):
            print(f"  {tid:<8}  {tdef.display_name}")
        print()
        sys.exit(0)

    if not args.task:
        parser.error("--task is required (use --list-tasks to see options)")
    if not args.sif:
        parser.error("--sif is required")

    manifest = args.manifest or (Path(__file__).parent / "data" / "manifest.yaml")

    ok = run_validation(
        task_id=args.task,
        sif_path=args.sif.resolve(),
        manifest_path=manifest.resolve(),
        gpu=not args.no_gpu,
        executable=args.apptainer,
        timeout=args.timeout,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
