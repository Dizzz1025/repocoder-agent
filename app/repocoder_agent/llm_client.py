from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .models import CommandResult, PatchInstruction, RelevantFile
from .repository import RepoFile, RepoSnapshot

DEFAULT_OPENAI_BASE_URL = "https://api-inference.modelscope.cn/v1"
DEFAULT_OPENAI_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
MAX_CONTEXT_CHARS = 12_000
MAX_FILE_CONTEXT_CHARS = 4_000


@dataclass
class LLMRetrySuggestion:
    reflection: str
    retry_prompt: str
    patch: PatchInstruction | None


class SupportsRepoCoderLLM(Protocol):
    def build_plan(
        self,
        goal: str,
        relevant_files: list[RelevantFile],
        commands: list[str],
        patches: list[PatchInstruction],
        auto_fix: bool,
    ) -> list[str] | None: ...

    def generate_patch(
        self,
        goal: str,
        snapshot: RepoSnapshot,
        relevant_files: list[RelevantFile],
    ) -> PatchInstruction | None: ...

    def reflect_and_suggest_fix(
        self,
        goal: str,
        command_result: CommandResult,
        snapshot: RepoSnapshot,
        relevant_files: list[RelevantFile],
        applied_patch_summaries: list[str],
    ) -> LLMRetrySuggestion | None: ...


def create_llm_client_from_env() -> SupportsRepoCoderLLM | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    try:
        client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            api_key=api_key,
        )
    except Exception:
        return None

    return OpenAICompatibleLLMClient(
        client=client,
        model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
    )


class OpenAICompatibleLLMClient:
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def build_plan(
        self,
        goal: str,
        relevant_files: list[RelevantFile],
        commands: list[str],
        patches: list[PatchInstruction],
        auto_fix: bool,
    ) -> list[str] | None:
        payload = self._request_json(
            system_prompt=(
                "You are the planning module for RepoCoder-Agent. "
                "Return strict JSON only."
            ),
            user_prompt=(
                "Create a concise execution plan for this coding task.\n"
                f"Goal: {goal}\n"
                f"Commands: {commands}\n"
                f"Auto-fix enabled: {auto_fix}\n"
                f"Existing patches: {self._serialize_patch_instructions(patches)}\n"
                f"Relevant files: {self._serialize_relevant_files(relevant_files)}\n\n"
                "Return JSON with this schema only:\n"
                '{"plan_steps": ["step 1", "step 2"]}\n'
                "Requirements:\n"
                "- Keep 4 to 7 short actionable steps.\n"
                "- Reflect the existing repository scan, patch, command, and retry flow.\n"
                "- Do not use markdown."
            ),
        )
        if not isinstance(payload, dict):
            return None
        plan_steps = payload.get("plan_steps")
        if not isinstance(plan_steps, list):
            return None
        cleaned = [str(item).strip() for item in plan_steps if str(item).strip()]
        return cleaned or None

    def generate_patch(
        self,
        goal: str,
        snapshot: RepoSnapshot,
        relevant_files: list[RelevantFile],
    ) -> PatchInstruction | None:
        payload = self._request_json(
            system_prompt=(
                "You generate one minimal patch instruction for RepoCoder-Agent. "
                "Return strict JSON only."
            ),
            user_prompt=(
                "Generate one safe patch instruction for this coding task.\n"
                f"Goal: {goal}\n"
                f"Repository summary: indexed_files={snapshot.summary.indexed_count}\n"
                f"Relevant files: {self._serialize_relevant_files(relevant_files)}\n\n"
                f"File context:\n{self._serialize_file_context(snapshot, relevant_files)}\n\n"
                "Return JSON with this schema only:\n"
                '{"patch": {"file_path": "path.py", "operation": "replace", '
                '"find_text": "old", "replace_text": "new"}}\n'
                'If there is no safe minimal patch, return {"patch": null}.\n'
                "Rules:\n"
                "- file_path must be relative to the repository root.\n"
                "- Prefer one replace patch when possible.\n"
                "- find_text must exactly match provided file content.\n"
                "- Do not use markdown."
            ),
        )
        return self._parse_patch_from_payload(payload)

    def reflect_and_suggest_fix(
        self,
        goal: str,
        command_result: CommandResult,
        snapshot: RepoSnapshot,
        relevant_files: list[RelevantFile],
        applied_patch_summaries: list[str],
    ) -> LLMRetrySuggestion | None:
        payload = self._request_json(
            system_prompt=(
                "You are the retry and reflection module for RepoCoder-Agent. "
                "Return strict JSON only."
            ),
            user_prompt=(
                "The validation command failed. Reflect on the failure and suggest one "
                "minimal retry patch if possible.\n"
                f"Goal: {goal}\n"
                f"Failed command: {command_result.command}\n"
                f"Exit code: {command_result.exit_code}\n"
                f"Timed out: {command_result.timed_out}\n"
                f"Stdout:\n{command_result.stdout}\n\n"
                f"Stderr:\n{command_result.stderr}\n\n"
                f"Applied patch summaries: {applied_patch_summaries}\n"
                f"Relevant files: {self._serialize_relevant_files(relevant_files)}\n\n"
                f"Current file context:\n{self._serialize_file_context(snapshot, relevant_files)}\n\n"
                "Return JSON with this schema only:\n"
                '{"reflection": "short analysis", "retry_prompt": "next attempt guidance", '
                '"patch": {"file_path": "path.py", "operation": "replace", '
                '"find_text": "old", "replace_text": "new"}}\n'
                'If you cannot produce a safe patch, return "patch": null.\n'
                "Do not use markdown."
            ),
        )
        if not isinstance(payload, dict):
            return None

        patch = self._parse_patch_from_payload(payload)
        reflection = str(payload.get("reflection", "")).strip()
        retry_prompt = str(payload.get("retry_prompt", "")).strip()
        if not reflection and not retry_prompt and patch is None:
            return None
        return LLMRetrySuggestion(
            reflection=reflection,
            retry_prompt=retry_prompt,
            patch=patch,
        )

    def _request_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            return None

        try:
            message = response.choices[0].message
        except (AttributeError, IndexError):
            return None

        content = self._normalize_content(getattr(message, "content", None))
        payload = self._extract_json_payload(content)
        return payload if isinstance(payload, dict) else None

    def _parse_patch_from_payload(self, payload: dict[str, Any] | None) -> PatchInstruction | None:
        if not isinstance(payload, dict):
            return None

        patch_data = payload.get("patch")
        if patch_data is None or not isinstance(patch_data, dict):
            return None

        try:
            return PatchInstruction.model_validate(patch_data)
        except Exception:
            return None

    def _serialize_relevant_files(self, relevant_files: list[RelevantFile]) -> list[dict[str, Any]]:
        return [
            {
                "file_path": item.file_path,
                "score": item.score,
                "reason": item.reason,
            }
            for item in relevant_files
        ]

    def _serialize_patch_instructions(
        self, patches: list[PatchInstruction]
    ) -> list[dict[str, Any]]:
        return [item.model_dump(exclude_none=True) for item in patches]

    def _serialize_file_context(
        self,
        snapshot: RepoSnapshot,
        relevant_files: list[RelevantFile],
    ) -> str:
        files_by_path = {item.rel_path: item for item in snapshot.files}
        sections: list[str] = []
        total_chars = 0

        for relevant in relevant_files:
            repo_file = files_by_path.get(relevant.file_path)
            if repo_file is None:
                continue

            snippet = repo_file.content[:MAX_FILE_CONTEXT_CHARS]
            block = self._format_file_block(repo_file=repo_file, snippet=snippet)
            if total_chars + len(block) > MAX_CONTEXT_CHARS:
                break
            sections.append(block)
            total_chars += len(block)

        if not sections:
            return "No relevant file content available."
        return "\n\n".join(sections)

    def _format_file_block(self, repo_file: RepoFile, snippet: str) -> str:
        truncated = ""
        if len(repo_file.content) > len(snippet):
            truncated = "\n[truncated]"
        return f"FILE: {repo_file.rel_path}\n{snippet}{truncated}"

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        chunks: list[str] = []
        for item in content:
            text = None
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        return "\n".join(chunks)

    def _extract_json_payload(self, content: str) -> dict[str, Any] | None:
        content = content.strip()
        if not content:
            return None

        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        if fenced_match:
            content = fenced_match.group(1).strip()

        try:
            payload = json.loads(content)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None

        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
