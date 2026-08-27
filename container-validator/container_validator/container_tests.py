from __future__ import annotations

from result import TestResult
from runner import ApptainerRunner


def check_container_can_be_executed(
    runner: ApptainerRunner, instance_name: str
) -> TestResult:
    result = runner.exec_instance(instance_name, ["python", "--version"], timeout=60)
    if result.timed_out:
        return TestResult(
            "container_can_be_executed", False, "python --version timed out"
        )
    ok = result.succeeded
    return TestResult(
        "container_can_be_executed",
        ok,
        (result.stdout.strip() or result.stderr.strip())
        if ok
        else f"python --version failed (rc={result.returncode}): {result.stderr.strip()}",
    )


def check_container_can_access_gpu(
    runner: ApptainerRunner, instance_name: str
) -> TestResult:
    
    result = runner.exec_instance(instance_name, ["nvidia-smi", "-L"], timeout=60)
    
    if result.timed_out:
        return TestResult(
            "container_can_access_gpu",
            False,
            "nvidia-smi -L timed out inside container",
        )
    
    if not result.succeeded:
        return TestResult(
            "container_can_access_gpu",
            False,
            f"nvidia-smi -L failed inside container (rc={result.returncode}): {result.stderr.strip()}",
        )
    
    gpus = [ln for ln in result.stdout.splitlines() if ln.strip().startswith("GPU")]
    ok = len(gpus) > 0
    
    return TestResult(
        "container_can_access_gpu",
        ok,
        f"{len(gpus)} GPU(s) visible inside container"
        if ok
        else "nvidia-smi -L returned no GPUs inside container",
    )
