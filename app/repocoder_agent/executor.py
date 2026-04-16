from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .models import CommandResult


class CommandExecutor:
    def __init__(self, repo_path: str, timeout_sec: int = 60):
        self.repo_root = Path(repo_path).resolve()
        self.timeout_sec = timeout_sec

    def run(self, command: str) -> CommandResult:
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            duration = time.perf_counter() - start
            return CommandResult(
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                duration_sec=duration,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            return CommandResult(
                command=command,
                exit_code=124,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
                duration_sec=duration,
                timed_out=True,
            )

    def run_many(self, commands: list[str]) -> list[CommandResult]:
        return [self.run(command) for command in commands]
