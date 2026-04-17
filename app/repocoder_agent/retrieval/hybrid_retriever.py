from __future__ import annotations

import re
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
    ) -> list[RelevantFile]:
        goal_tokens = _tokenize(goal)
        if not goal_tokens:
            return []

        patch_success_counts = patch_success_counts or {}
        patch_failure_counts = patch_failure_counts or {}
        command_failure_counts = command_failure_counts or {}
        patch_history_events = patch_history_events or []
        command_failure_events = command_failure_events or []

        symbol_map = graph.symbol_names_by_file()
        import_map = graph.imported_modules_by_file()
        scored: list[tuple[float, RepoFile, str]] = []

        for repo_file in snapshot.files:
            path_tokens = _tokenize(repo_file.rel_path)
            content_tokens = _tokenize(repo_file.content[:4000])
            symbol_tokens = _tokenize(' '.join(sorted(symbol_map.get(repo_file.rel_path, set()))))
            import_tokens = _tokenize(' '.join(sorted(import_map.get(repo_file.rel_path, set()))))

            path_overlap = len(goal_tokens & path_tokens)
            content_overlap = len(goal_tokens & content_tokens)
            symbol_overlap = len(goal_tokens & symbol_tokens)
            import_overlap = len(goal_tokens & import_tokens)
            python_bonus = 0.2 if repo_file.rel_path.endswith('.py') else 0.0
            graph_bonus = float(symbol_overlap * 2.0) + float(import_overlap * 0.75)

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

            score = (
                float(content_overlap)
                + float(path_overlap * 2)
                + python_bonus
                + graph_bonus
                + success_history_bonus
                + failure_history_bonus
                + command_failure_bonus
            )
            if score <= 0:
                continue

            reasons: list[str] = []
            if content_overlap or path_overlap:
                reasons.append('token overlap with goal and path/content')
            if symbol_overlap:
                reasons.append('graph symbol match')
            if import_overlap:
                reasons.append('graph import match')
            if success_history_bonus:
                reasons.append('history patch-success match')
            if failure_history_bonus:
                reasons.append('history patch-failure overlap')
            if command_failure_bonus:
                reasons.append('history command-failure context')
            scored.append((score, repo_file, '; '.join(reasons) if reasons else 'hybrid retrieval score'))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RelevantFile(
                file_path=repo_file.rel_path,
                score=score,
                reason=reason,
            )
            for score, repo_file, reason in scored[:top_k]
        ]

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
