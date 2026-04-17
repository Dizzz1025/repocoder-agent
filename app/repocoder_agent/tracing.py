from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import get_settings
from .models import AgentRunResponse, AgentTaskRequest, RelevantFile, RepositorySummary


class RunTraceWriter:
    def __init__(self, repo_path: str):
        self.repo_root = Path(repo_path).resolve()
        self.settings = get_settings(start_dir=self.repo_root)

    def write_run(
        self,
        request: AgentTaskRequest,
        summary: RepositorySummary,
        relevant_files: list[RelevantFile],
        response: AgentRunResponse,
        selection_trace: dict | None = None,
        sandbox_trace: dict | None = None,
        retrieval_trace: dict | None = None,
        hooks_trace: dict | None = None,
    ) -> Path | None:
        if not self.settings.trace_enabled:
            return None

        trace_dir = (self.repo_root / self.settings.trace_dirname).resolve()
        if self.repo_root not in trace_dir.parents and trace_dir != self.repo_root:
            return None

        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            trace_path = trace_dir / f"{timestamp}-{uuid4().hex[:8]}.json"
            payload = {
                "request": request.model_dump(mode="json"),
                "summary": summary.model_dump(mode="json"),
                "relevant_files": [item.model_dump(mode="json") for item in relevant_files],
                "retrieval_trace": retrieval_trace or {},
                "selection_trace": selection_trace or {},
                "sandbox_trace": sandbox_trace or {},
                "hooks_trace": hooks_trace or {},
                "response": response.model_dump(mode="json"),
            }
            trace_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return trace_path
        except OSError:
            return None
