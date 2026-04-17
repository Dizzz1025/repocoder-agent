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
    relevant = HybridRetriever().retrieve(snapshot, "fix parse token parser", graph, top_k=2)

    assert relevant
    assert relevant[0].file_path == "best.py"
    assert "graph symbol match" in relevant[0].reason
