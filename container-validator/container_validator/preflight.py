from __future__ import annotations
import platform
import subprocess

from result import TestResult


def check_host_is_linux() -> TestResult:
    ok = platform.system() == "Linux"
    return TestResult(
        "host_is_linux",
        ok,
        "Linux OK"
        if ok
        else f"Host OS is {platform.system()!r} — Apptainer requires Linux",
    )


def check_apptainer_installed(executable: str = "apptainer") -> TestResult:
    try:
        proc = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=15
        )
        ok = proc.returncode == 0
        msg = proc.stdout.strip() if ok else f"failed: {proc.stderr.strip()}"
        return TestResult("apptainer_is_installed", ok, msg or "version OK")
    except FileNotFoundError:
        return TestResult(
            "apptainer_is_installed",
            False,
            f"{executable!r} not found on PATH — install Apptainer first",
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            "apptainer_is_installed", False, "apptainer --version timed out"
        )


def check_nvidia_driver() -> TestResult:
    try:
        proc = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=20
        )
        ok = proc.returncode == 0
        return TestResult(
            "nvidia_driver_available",
            ok,
            "NVIDIA driver OK"
            if ok
            else f"nvidia-smi failed: {proc.stderr.strip()[:200]}",
        )
    except FileNotFoundError:
        return TestResult(
            "nvidia_driver_available",
            False,
            "nvidia-smi not found — NVIDIA driver not installed?",
        )
    except subprocess.TimeoutExpired:
        return TestResult("nvidia_driver_available", False, "nvidia-smi timed out")


def check_nvidia_gpu() -> TestResult:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20
        )
        if proc.returncode != 0:
            return TestResult(
                "nvidia_gpu_available",
                False,
                f"nvidia-smi -L failed: {proc.stderr.strip()}",
            )
        gpus = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("GPU")]
        ok = len(gpus) > 0
        return TestResult(
            "nvidia_gpu_available",
            ok,
            f"{len(gpus)} GPU(s) found" if ok else "nvidia-smi -L listed no GPUs",
        )
    except FileNotFoundError:
        return TestResult("nvidia_gpu_available", False, "nvidia-smi not found")
    except subprocess.TimeoutExpired:
        return TestResult("nvidia_gpu_available", False, "nvidia-smi -L timed out")


def run_preflight(
    executable: str = "apptainer", require_gpu: bool = True
) -> list[TestResult]:
    checks: list[TestResult] = [
        check_host_is_linux(),
        check_apptainer_installed(executable),
    ]
    if require_gpu:
        checks += [check_nvidia_driver(), check_nvidia_gpu()]
    return checks
