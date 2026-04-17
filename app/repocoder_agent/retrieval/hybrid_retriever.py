from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..memory.graph_builder import RepositoryGraph
from ..models import RelevantFile
from ..repository import RepoFile, RepoSnapshot


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', text.lower()):
        if len(token) >= 2:
            tokens.add(token)
        for part in token.split('_'):
            if len(part) >= 2:
                tokens.add(part)
    return tokens


@dataclass(frozen=True)
class RetrievalCandidateScore:
    file_path: str
    total_score: float
    reason: str
    score_breakdown: dict[str, float]


@dataclass(frozen=True)
class RetrievalResult:
    relevant_files: tuple[RelevantFile, ...]
    evaluations: tuple[RetrievalCandidateScore, ...]


class HybridRetriever:
    def retrieve(
        self,
        snapshot: RepoSnapshot,
        goal: str,
        graph: RepositoryGraph,
        top_k: int = 5,
        patch_success_counts: dict[str, int] | None = None,
        patch_failure_counts: dict[str, int] | None = None,
        command_failure_counts: dict[str, int] | None = None,
        patch_history_events: list[dict[str, Any]] | None = None,
        command_failure_events: list[dict[str, str]] | None = None,
        file_memory: dict[str, dict[str, Any]] | None = None,
    ) -> list[RelevantFile]:
        return list(
            self.retrieve_with_details(
                snapshot=snapshot,
                goal=goal,
                graph=graph,
                top_k=top_k,
                patch_success_counts=patch_success_counts,
                patch_failure_counts=patch_failure_counts,
                command_failure_counts=command_failure_counts,
                patch_history_events=patch_history_events,
                command_failure_events=command_failure_events,
                file_memory=file_memory,
            ).relevant_files
        )

    def retrieve_with_details(
        self,
        snapshot: RepoSnapshot,
        goal: str,
        graph: RepositoryGraph,
        top_k: int = 5,
        patch_success_counts: dict[str, int] | None = None,
        patch_failure_counts: dict[str, int] | None = None,
        command_failure_counts: dict[str, int] | None = None,
        patch_history_events: list[dict[str, Any]] | None = None,
        command_failure_events: list[dict[str, str]] | None = None,
        file_memory: dict[str, dict[str, Any]] | None = None,
    ) -> RetrievalResult:
        goal_tokens = _tokenize(goal)
        if not goal_tokens:
            return RetrievalResult(relevant_files=(), evaluations=())

        patch_success_counts = patch_success_counts or {}
        patch_failure_counts = patch_failure_counts or {}
        command_failure_counts = command_failure_counts or {}
        patch_history_events = patch_history_events or []
        command_failure_events = command_failure_events or []
        file_memory = file_memory or {}

        symbol_map = graph.symbol_names_by_file()
        import_map = graph.imported_modules_by_file()
        call_map = graph.call_names_by_file()
        scored: list[tuple[float, RepoFile, str, dict[str, float]]] = []

        for repo_file in snapshot.files:
            path_tokens = _tokenize(repo_file.rel_path)
            content_tokens = _tokenize(repo_file.content[:4000])
            symbol_tokens = _tokenize(' '.join(sorted(symbol_map.get(repo_file.rel_path, set()))))
            import_tokens = _tokenize(' '.join(sorted(import_map.get(repo_file.rel_path, set()))))
            call_tokens = _tokenize(' '.join(sorted(call_map.get(repo_file.rel_path, set()))))

            path_overlap = len(goal_tokens & path_tokens)
            content_overlap = len(goal_tokens & content_tokens)
            symbol_overlap = len(goal_tokens & symbol_tokens)
            import_overlap = len(goal_tokens & import_tokens)
            call_overlap = len(goal_tokens & call_tokens)

            breakdown: dict[str, float] = {}
            token_score = float(content_overlap) + float(path_overlap * 2)
            if token_score:
                breakdown['token_overlap'] = token_score
            python_bonus = 0.2 if repo_file.rel_path.endswith('.py') else 0.0
            if python_bonus:
                breakdown['python_bonus'] = python_bonus
            graph_symbol_bonus = float(symbol_overlap * 2.0)
            if graph_symbol_bonus:
                breakdown['graph_symbol_bonus'] = graph_symbol_bonus
            graph_import_bonus = float(import_overlap * 0.75)
            if graph_import_bonus:
                breakdown['graph_import_bonus'] = graph_import_bonus
            graph_call_bonus = float(call_overlap * 1.25)
            if graph_call_bonus:
                breakdown['graph_call_bonus'] = graph_call_bonus

            success_history_bonus = self._history_event_bonus(
                file_path=repo_file.rel_path,
                goal_tokens=goal_tokens,
                events=patch_history_events,
                success=True,
            )
            failure_history_bonus = self._history_event_bonus(
                file_path=repo_file.rel_path,
                goal_tokens=goal_tokens,
                events=patch_history_events,
                success=False,
            )
            if success_history_bonus == 0.0:
                success_history_bonus = float(patch_success_counts.get(repo_file.rel_path, 0)) * 0.05
            if failure_history_bonus == 0.0:
                failure_history_bonus = float(patch_failure_counts.get(repo_file.rel_path, 0)) * 0.10
            if success_history_bonus:
                breakdown['history_success_bonus'] = success_history_bonus
            if failure_history_bonus:
                breakdown['history_failure_bonus'] = failure_history_bonus

            command_failure_bonus = self._command_failure_bonus(
                goal_tokens=goal_tokens,
                events=command_failure_events,
            )
            if command_failure_bonus == 0.0 and command_failure_counts:
                for command, count in command_failure_counts.items():
                    command_tokens = _tokenize(command)
                    overlap = len(goal_tokens & command_tokens)
                    if overlap > 0:
                        command_failure_bonus += float(count) * 0.05
                        break
            if command_failure_bonus:
                breakdown['command_failure_bonus'] = command_failure_bonus

            file_memory_bonus = self._file_memory_bonus(
                file_path=repo_file.rel_path,
                file_memory=file_memory,
                has_base_signal=bool(token_score or graph_symbol_bonus or graph_import_bonus or graph_call_bonus),
            )
            if file_memory_bonus:
                breakdown['file_memory_bonus'] = file_memory_bonus

            score = sum(breakdown.values())
            if score <= 0:
                continue

            reasons: list[str] = []
            if token_score:
                reasons.append('token overlap with goal and path/content')
            if graph_symbol_bonus:
                reasons.append('graph symbol match')
            if graph_import_bonus:
                reasons.append('graph import match')
            if graph_call_bonus:
                reasons.append('graph call match')
            if success_history_bonus:
                reasons.append('history patch-success match')
            if failure_history_bonus:
                reasons.append('history patch-failure overlap')
            if command_failure_bonus:
                reasons.append('history command-failure context')
            if file_memory_bonus:
                reasons.append('file-memory hotspot bonus')
            scored.append((score, repo_file, '; '.join(reasons) if reasons else 'hybrid retrieval score', breakdown))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:top_k]
        return RetrievalResult(
            relevant_files=tuple(
                RelevantFile(
                    file_path=repo_file.rel_path,
                    score=score,
                    reason=reason,
                )
                for score, repo_file, reason, _breakdown in top
            ),
            evaluations=tuple(
                RetrievalCandidateScore(
                    file_path=repo_file.rel_path,
                    total_score=score,
                    reason=reason,
                    score_breakdown=breakdown,
                )
                for score, repo_file, reason, breakdown in top
            ),
        )

    def _history_event_bonus(
        self,
        file_path: str,
        goal_tokens: set[str],
        events: list[dict[str, Any]],
        success: bool,
    ) -> float:
        bonus = 0.0
        weight = 0.12 if success else 0.20
        for event in events:
            if event.get('success') is not success:
                continue
            if event.get('file_path') != file_path:
                continue
            event_text = ' '.join(
                [
                    str(event.get('file_path', '')),
                    str(event.get('operation', '')),
                    str(event.get('message', '')),
                ]
            )
            overlap = len(goal_tokens & _tokenize(event_text))
            if overlap > 0:
                bonus += float(overlap) * weight
        return bonus

    def _command_failure_bonus(
        self,
        goal_tokens: set[str],
        events: list[dict[str, str]],
    ) -> float:
        bonus = 0.0
        for event in events:
            event_text = ' '.join(
                [
                    event.get('command', ''),
                    event.get('stderr', ''),
                    event.get('stdout', ''),
                ]
            )
            overlap = len(goal_tokens & _tokenize(event_text))
            if overlap > 0:
                bonus += float(overlap) * 0.04
        return bonus

    def _file_memory_bonus(
        self,
        file_path: str,
        file_memory: dict[str, dict[str, Any]],
        has_base_signal: bool,
    ) -> float:
        memory = file_memory.get(file_path)
        if memory is None:
            return 0.0
        hotspot_score = float(memory.get('hotspot_score', 0.0))
        patch_success_count = int(memory.get('patch_success_count', 0))
        patch_failure_count = int(memory.get('patch_failure_count', 0))
        command_failure_count = int(memory.get('command_failure_count', 0))

        if has_base_signal:
            return (
                min(hotspot_score, 10.0) * 0.03
                + float(patch_success_count) * 0.04
                + float(patch_failure_count) * 0.06
                + float(command_failure_count) * 0.05
            )
        return min(hotspot_score, 5.0) * 0.01
