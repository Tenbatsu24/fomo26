from __future__ import annotations
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class ApptainerRunner:
    def __init__(self, executable: str = "apptainer", gpu: bool = True) -> None:
        self.executable = executable
        self.gpu = gpu

    def _run(self, cmd: list[str], timeout: int) -> RunResult:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return RunResult(proc.returncode, proc.stdout, proc.stderr)
        
        except subprocess.TimeoutExpired:
            return RunResult(-1, "", f"timed out after {timeout}s", timed_out=True)
        
        except FileNotFoundError as exc:
            return RunResult(127, "", str(exc))

    def start_instance(
        self,
        sif_path: Path,
        instance_name: str,
        input_dir: Path,
        output_dir: Path,
        tmp_dir: Path,
    ) -> RunResult:
        
        cmd = [self.executable, "instance", "start"]
        if self.gpu:
            cmd += ["--nvccli"]
        cmd += [
            "--bind",
            f"{input_dir}:/input:ro",
            "--bind",
            f"{output_dir}:/output:rw",
            "--bind",
            f"{tmp_dir}:/tmp:rw",
            str(sif_path),
            instance_name,
        ]
        return self._run(cmd, timeout=120)

    def stop_instance(self, instance_name: str) -> RunResult:
        return self._run(
            [self.executable, "instance", "stop", instance_name], timeout=60
        )

    def exec_instance(
        self, instance_name: str, cmd: list[str], timeout: int = 900
    ) -> RunResult:
        full_cmd = [self.executable, "exec", f"instance://{instance_name}"] + cmd
        return self._run(full_cmd, timeout=timeout)
