from __future__ import annotations

from repocoder_agent.memory.graph_builder import RepositoryGraphBuilder
from repocoder_agent.repository import RepositoryScanner
from repocoder_agent.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_retriever_prefers_symbol_match(tmp_path) -> None:
    (tmp_path / "best.py").write_text(
        "# parser helper\ndef parse_token(value):\n    return value\n",
        encoding="utf-8",
    )
    (tmp_path / "other.py").write_text(
        "# parser helper\nvalue = 1\n",
        encoding="utf-8",
    )

    scanner = RepositoryScanner(str(tmp_path))
    snapshot = scanner.scan()
    graph = RepositoryGraphBuilder().build_from_snapshot(snapshot)
    retrieval = HybridRetriever().retrieve_with_details(snapshot, "fix parse token parser", graph, top_k=2)

    assert retrieval.relevant_files
    assert retrieval.relevant_files[0].file_path == "best.py"
    assert "graph symbol match" in retrieval.relevant_files[0].reason
    assert retrieval.evaluations[0].score_breakdown["graph_symbol_bonus"] > 0
