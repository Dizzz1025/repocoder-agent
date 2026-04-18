from __future__ import annotations

from dataclasses import dataclass

from ..memory.graph_builder import RepositoryGraph
from ..memory.graph_store import RepositoryGraphStore
from ..memory.history_store import RepositoryHistoryStore
from ..repository import RepoSnapshot, RepositoryScanner
from ..retrieval.hybrid_retriever import HybridRetriever, RetrievalResult


@dataclass(frozen=True)
class RepoExplorerResult:
    snapshot: RepoSnapshot
    graph: RepositoryGraph
    graph_diff: dict
    retrieval_result: RetrievalResult


class RepoExplorerAgent:
    def __init__(
        self,
        scanner: RepositoryScanner,
        graph_store: RepositoryGraphStore,
        graph_builder,
        history_store: RepositoryHistoryStore,
        retriever: HybridRetriever,
    ):
        self.scanner = scanner
        self.graph_store = graph_store
        self.graph_builder = graph_builder
        self.history_store = history_store
        self.retriever = retriever

    def explore(self, goal: str, top_k_files: int) -> RepoExplorerResult:
        snapshot = self.scanner.scan()
        graph, graph_diff = self.graph_store.refresh(snapshot, self.graph_builder)
        retrieval_result = self.retriever.retrieve_with_details(
            snapshot=snapshot,
            goal=goal,
            graph=graph,
            top_k=top_k_files,
            patch_success_counts=self.history_store.patch_success_counts(),
            patch_failure_counts=self.history_store.patch_failure_counts(),
            command_failure_counts=self.history_store.command_failure_counts(),
            patch_history_events=self.history_store.patch_history_events(),
            command_failure_events=self.history_store.command_failure_events(),
            file_memory=self.history_store.file_memory(),
        )
        return RepoExplorerResult(
            snapshot=snapshot,
            graph=graph,
            graph_diff=graph_diff,
            retrieval_result=retrieval_result,
        )
