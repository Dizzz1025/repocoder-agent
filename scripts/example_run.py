from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from repocoder_agent.agent import RepoCoderAgent  # noqa: E402
from repocoder_agent.models import AgentTaskRequest, PatchInstruction  # noqa: E402


def main() -> None:
    repo_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT)
    task = AgentTaskRequest(
        repository_path=repo_path,
        goal="在 README.md 将`TODO`替换为`RepoCoder-Agent MVP`",
        commands=["python -m pytest -q"],
        patches=[
            PatchInstruction(
                file_path="README.md",
                operation="replace",
                find_text="TODO",
                replace_text="RepoCoder-Agent MVP",
            )
        ],
        auto_fix=True,
        max_iterations=2,
    )
    result = RepoCoderAgent(task).run()
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
