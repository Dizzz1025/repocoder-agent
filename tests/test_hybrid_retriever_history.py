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
        file_memory={
            'best.py': {
                'patch_success_count': 3,
                'patch_failure_count': 0,
                'command_failure_count': 1,
                'hotspot_score': 1.3,
                'last_failure_message': '',
                'last_updated_at': '',
            }
        },
    )

    assert relevant
    assert relevant[0].file_path == 'best.py'
    assert 'history patch-success match' in relevant[0].reason


def test_hybrid_retriever_uses_file_memory_hotspot_bonus(tmp_path) -> None:
    (tmp_path / 'best.py').write_text('def parse_token():\n    return 1\n', encoding='utf-8')
    (tmp_path / 'other.py').write_text('def parse_token():\n    return 2\n', encoding='utf-8')

    snapshot = RepositoryScanner(str(tmp_path)).scan()
    graph = RepositoryGraphBuilder().build_from_snapshot(snapshot)
    retriever = HybridRetriever()
    retrieval = retriever.retrieve_with_details(
        snapshot=snapshot,
        goal='fix parse token bug',
        graph=graph,
        top_k=2,
        file_memory={
            'best.py': {
                'patch_success_count': 2,
                'patch_failure_count': 1,
                'command_failure_count': 1,
                'hotspot_score': 2.5,
                'last_failure_message': 'parser failed',
                'last_updated_at': '',
            }
        },
    )

    assert retrieval.relevant_files
    best = next(item for item in retrieval.evaluations if item.file_path == 'best.py')
    assert best.score_breakdown['file_memory_bonus'] > 0
