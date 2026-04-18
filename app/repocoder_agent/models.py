from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .config import DEFAULT_COMMANDS, DEFAULT_COMMAND_TIMEOUT_SEC


PatchOperation = Literal["replace", "append", "create"]
RunMode = Literal["execute", "plan"]


def _default_commands() -> list[str]:
    return list(DEFAULT_COMMANDS)


def _default_command_timeout_sec() -> int:
    return DEFAULT_COMMAND_TIMEOUT_SEC


class PatchInstruction(BaseModel):
    file_path: str = Field(..., description="Path relative to repository root.")
    operation: PatchOperation = Field(
        ..., description="Supported operations: replace, append, create."
    )
    find_text: str | None = None
    replace_text: str | None = None
    content: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "PatchInstruction":
        if self.operation == "replace":
            if self.find_text is None or self.replace_text is None:
                raise ValueError("replace operation requires find_text and replace_text.")
        elif self.operation in {"append", "create"}:
            if self.content is None:
                raise ValueError(f"{self.operation} operation requires content.")
        return self


class AppliedPatch(BaseModel):
    file_path: str
    operation: PatchOperation
    success: bool
    message: str


class CommandResult(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False


class RelevantFile(BaseModel):
    file_path: str
    score: float
    reason: str


class RepositorySummary(BaseModel):
    repo_path: str
    language: str = "python"
    file_count: int
    indexed_count: int
    skipped_count: int


class ScanRequest(BaseModel):
    repository_path: str = Field(..., description="Absolute or relative repository path.")


class ScanResponse(BaseModel):
    summary: RepositorySummary
    top_files: list[str]


class PlanRequest(BaseModel):
    repository_path: str
    goal: str
    commands: list[str] = Field(default_factory=_default_commands)
    top_k_files: int = Field(default=5, ge=1, le=30)
    patches: list[PatchInstruction] = Field(default_factory=list)


class PlanResponse(BaseModel):
    summary: RepositorySummary
    relevant_files: list[RelevantFile]
    plan_steps: list[str]


class AgentTaskRequest(BaseModel):
    repository_path: str
    goal: str
    commands: list[str] = Field(default_factory=_default_commands)
    patches: list[PatchInstruction] = Field(default_factory=list)
    max_iterations: int = Field(default=3, ge=1, le=10)
    top_k_files: int = Field(default=5, ge=1, le=30)
    auto_fix: bool = True
    command_timeout_sec: int = Field(default_factory=_default_command_timeout_sec, ge=1, le=600)
    mode: RunMode = Field(default="execute")
    skill: str | None = None


class AgentRunResponse(BaseModel):
    success: bool
    summary: RepositorySummary
    relevant_files: list[RelevantFile]
    plan_steps: list[str]
    applied_patches: list[AppliedPatch]
    command_results: list[CommandResult]
    iterations_used: int
    message: str
    mode: RunMode = Field(default="execute")
    skill: str | None = None
    proposed_patches: list[PatchInstruction] = Field(default_factory=list)
