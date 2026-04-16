from __future__ import annotations

from pathlib import Path

from .models import AppliedPatch, PatchInstruction


class PatchApplier:
    def __init__(self, repo_path: str):
        self.repo_root = Path(repo_path).resolve()

    def _resolve_target(self, rel_path: str) -> Path:
        target = (self.repo_root / rel_path).resolve()
        if self.repo_root not in target.parents and target != self.repo_root:
            raise ValueError(f"Invalid path outside repository: {rel_path}")
        return target

    def apply(self, instruction: PatchInstruction) -> AppliedPatch:
        try:
            target = self._resolve_target(instruction.file_path)
        except ValueError as exc:
            return AppliedPatch(
                file_path=instruction.file_path,
                operation=instruction.operation,
                success=False,
                message=str(exc),
            )

        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            if instruction.operation == "replace":
                if not target.exists():
                    return AppliedPatch(
                        file_path=instruction.file_path,
                        operation=instruction.operation,
                        success=False,
                        message="Target file does not exist for replace operation.",
                    )
                content = target.read_text(encoding="utf-8")
                assert instruction.find_text is not None
                assert instruction.replace_text is not None
                if instruction.find_text not in content:
                    return AppliedPatch(
                        file_path=instruction.file_path,
                        operation=instruction.operation,
                        success=False,
                        message="find_text not found; no changes applied.",
                    )
                updated = content.replace(instruction.find_text, instruction.replace_text, 1)
                target.write_text(updated, encoding="utf-8")
                return AppliedPatch(
                    file_path=instruction.file_path,
                    operation=instruction.operation,
                    success=True,
                    message="Applied one replace operation.",
                )

            if instruction.operation == "append":
                existing = ""
                if target.exists():
                    existing = target.read_text(encoding="utf-8")
                assert instruction.content is not None
                separator = "" if not existing or existing.endswith("\n") else "\n"
                target.write_text(existing + separator + instruction.content, encoding="utf-8")
                return AppliedPatch(
                    file_path=instruction.file_path,
                    operation=instruction.operation,
                    success=True,
                    message="Appended content successfully.",
                )

            if instruction.operation == "create":
                if target.exists():
                    return AppliedPatch(
                        file_path=instruction.file_path,
                        operation=instruction.operation,
                        success=False,
                        message="File already exists; create operation skipped.",
                    )
                assert instruction.content is not None
                target.write_text(instruction.content, encoding="utf-8")
                return AppliedPatch(
                    file_path=instruction.file_path,
                    operation=instruction.operation,
                    success=True,
                    message="Created file successfully.",
                )

            return AppliedPatch(
                file_path=instruction.file_path,
                operation=instruction.operation,
                success=False,
                message=f"Unsupported operation: {instruction.operation}",
            )
        except OSError as exc:
            return AppliedPatch(
                file_path=instruction.file_path,
                operation=instruction.operation,
                success=False,
                message=f"Patch failed due to filesystem error: {exc}",
            )

    def apply_many(self, instructions: list[PatchInstruction]) -> list[AppliedPatch]:
        return [self.apply(item) for item in instructions]
