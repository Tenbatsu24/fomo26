from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    subject_id: str | None = None
