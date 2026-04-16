from __future__ import annotations

import json
import sys
from pathlib import Path
import debugpy
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from repocoder_agent.agent import RepoCoderAgent  # noqa: E402
from repocoder_agent.models import AgentTaskRequest, PatchInstruction  # noqa: E402


def main() -> None:
    repo_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT)
    task = AgentTaskRequest(
        repository_path=repo_path,
        goal="在 calculator.py 将`return a - b`替换为`return a + b`",
        commands=["python -m pytest -q"],
        patches=[
            PatchInstruction(
                file_path="calculator.py",
                operation="replace",
                find_text="return a - b",
                replace_text="return a + b",
            )
        ],
        auto_fix=True,
        max_iterations=2,
    )
    result = RepoCoderAgent(task).run()
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    debugpy.listen(("0.0.0.0", 5678))   # 监听 5678 端口
    print("Waiting for debugger attach on port 5678...")
    debugpy.wait_for_client()           # 阻塞，直到调试器连上
    print("Debugger attached.")

    main()
