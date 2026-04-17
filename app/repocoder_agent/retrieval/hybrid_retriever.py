from __future__ import annotations

import re

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
    ) -> list[RelevantFile]:
        goal_tokens = _tokenize(goal)
        if not goal_tokens:
            return []

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
            score = float(content_overlap) + float(path_overlap * 2) + python_bonus + graph_bonus
            if score <= 0:
                continue

            reasons: list[str] = []
            if content_overlap or path_overlap:
                reasons.append('token overlap with goal and path/content')
            if symbol_overlap:
                reasons.append('graph symbol match')
            if import_overlap:
                reasons.append('graph import match')
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
