from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import get_settings
from .executor import CommandExecutor
from .models import AppliedPatch, CommandResult, PatchInstruction
from .patcher import PatchApplier

_IGNORED_COPY_PATTERNS = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".repocoder",
)


@dataclass(frozen=True)
class DryRunResult:
    success: bool
    patch_results: tuple[AppliedPatch, ...]
    command_results: tuple[CommandResult, ...]
    message: str


class DryRunSandbox:
    def __init__(self, repo_path: str, timeout_sec: int = 60):
        self.repo_root = Path(repo_path).resolve()
        self.timeout_sec = timeout_sec
        self.settings = get_settings(start_dir=self.repo_root)

    def validate_patches(
        self,
        patches: list[PatchInstruction],
        commands: list[str],
    ) -> DryRunResult:
        if not patches:
            return DryRunResult(
                success=True,
                patch_results=(),
                command_results=(),
                message="Dry-run sandbox skipped because there were no patches to validate.",
            )
        if not self.settings.dry_run_enabled:
            return DryRunResult(
                success=True,
                patch_results=(),
                command_results=(),
                message="Dry-run sandbox disabled by configuration.",
            )

        try:
            with TemporaryDirectory(prefix="repocoder-sandbox-") as tmpdir:
                sandbox_repo = Path(tmpdir) / "repo"
                shutil.copytree(
                    self.repo_root,
                    sandbox_repo,
                    ignore=shutil.ignore_patterns(*_IGNORED_COPY_PATTERNS),
                )
                patcher = PatchApplier(str(sandbox_repo))
                patch_results = tuple(patcher.apply_many(patches))
                if not all(item.success for item in patch_results):
                    return DryRunResult(
                        success=False,
                        patch_results=patch_results,
                        command_results=(),
                        message="Dry-run sandbox rejected the patch because it could not be applied in the temporary copy.",
                    )

                executor = CommandExecutor(str(sandbox_repo), timeout_sec=self.timeout_sec)
                command_results = tuple(executor.run_many(commands))
                if any(item.exit_code != 0 for item in command_results):
                    return DryRunResult(
                        success=False,
                        patch_results=patch_results,
                        command_results=command_results,
                        message="Dry-run sandbox rejected the patch because validation commands failed in the temporary copy.",
                    )

                return DryRunResult(
                    success=True,
                    patch_results=patch_results,
                    command_results=command_results,
                    message="Dry-run sandbox validation passed.",
                )
        except OSError as exc:
            return DryRunResult(
                success=False,
                patch_results=(),
                command_results=(),
                message=f"Dry-run sandbox failed due to filesystem error: {exc}",
            )
