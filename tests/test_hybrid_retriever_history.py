from __future__ import annotations

from repocoder_agent.memory.graph_builder import RepositoryGraphBuilder
from repocoder_agent.repository import RepositoryScanner
from repocoder_agent.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_retriever_uses_history_bonuses(tmp_path) -> None:
    (tmp_path / "best.py").write_text("def handle_bug():\n    return True\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("def handle_bug():\n    return False\n", encoding="utf-8")

    snapshot = RepositoryScanner(str(tmp_path)).scan()
    graph = RepositoryGraphBuilder().build_from_snapshot(snapshot)
    retriever = HybridRetriever()
    relevant = retriever.retrieve(
        snapshot=snapshot,
        goal="fix handle bug",
        graph=graph,
        top_k=2,
        patch_success_counts={"best.py": 3},
        patch_failure_counts={"other.py": 1},
        command_failure_counts={"python -m pytest -q": 2},
        patch_history_events=[
            {
                'file_path': 'best.py',
                'operation': 'replace',
                'success': True,
                'message': 'fixed handle bug parser issue',
            },
            {
                'file_path': 'other.py',
                'operation': 'replace',
                'success': False,
                'message': 'failed to fix unrelated config problem',
            },
        ],
        command_failure_events=[
            {
                'command': 'python -m pytest -q tests/test_parser.py',
                'stderr': 'parser bug still failing',
                'stdout': '',
            }
        ],
    )

    assert relevant
    assert relevant[0].file_path == 'best.py'
    assert 'history patch-success match' in relevant[0].reason


def test_hybrid_retriever_requires_overlap_for_failure_bonus(tmp_path) -> None:
    (tmp_path / 'best.py').write_text('def parse_token():\n    return 1\n', encoding='utf-8')
    (tmp_path / 'other.py').write_text('def auth_check():\n    return 0\n', encoding='utf-8')

    snapshot = RepositoryScanner(str(tmp_path)).scan()
    graph = RepositoryGraphBuilder().build_from_snapshot(snapshot)
    retriever = HybridRetriever()
    relevant = retriever.retrieve(
        snapshot=snapshot,
        goal='fix parse token bug',
        graph=graph,
        top_k=2,
        patch_history_events=[
            {
                'file_path': 'other.py',
                'operation': 'replace',
                'success': False,
                'message': 'auth security failure',
            }
        ],
    )

    assert relevant
    assert relevant[0].file_path == 'best.py'
