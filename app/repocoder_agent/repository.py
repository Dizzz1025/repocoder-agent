from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .models import RelevantFile, RepositorySummary

IGNORED_DIRS = {
    '.git',
    '.hg',
    '.svn',
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    '.venv',
    'venv',
    'env',
    'node_modules',
    '.repocoder',
}
SUPPORTED_SUFFIXES = {'.py', '.md', '.toml', '.yaml', '.yml', '.json', '.ini', '.txt'}
MAX_FILE_BYTES = 512_000


@dataclass
class RepoFile:
    rel_path: str
    content: str


@dataclass
class RepoSnapshot:
    summary: RepositorySummary
    files: list[RepoFile]


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', text.lower()) if len(t) >= 2}


class RepositoryScanner:
    def __init__(self, repo_path: str):
        self.repo_root = Path(repo_path).resolve()

    def scan(self) -> RepoSnapshot:
        if not self.repo_root.exists() or not self.repo_root.is_dir():
            raise FileNotFoundError(f'Repository path not found: {self.repo_root}')

        indexed_files: list[RepoFile] = []
        total_count = 0
        skipped = 0

        for path in self.repo_root.rglob('*'):
            if path.is_dir():
                continue
            total_count += 1

            if any(part in IGNORED_DIRS for part in path.parts):
                skipped += 1
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                skipped += 1
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                skipped += 1
                continue

            try:
                content = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                skipped += 1
                continue
            except OSError:
                skipped += 1
                continue

            rel_path = str(path.relative_to(self.repo_root)).replace('\\', '/')
            indexed_files.append(RepoFile(rel_path=rel_path, content=content))

        summary = RepositorySummary(
            repo_path=str(self.repo_root),
            file_count=total_count,
            indexed_count=len(indexed_files),
            skipped_count=skipped,
        )
        return RepoSnapshot(summary=summary, files=indexed_files)

    def retrieve_relevant_files(
        self, snapshot: RepoSnapshot, goal: str, top_k: int = 5
    ) -> list[RelevantFile]:
        goal_tokens = _tokenize(goal)
        if not goal_tokens:
            return []

        scored: list[tuple[float, RepoFile]] = []
        for repo_file in snapshot.files:
            path_tokens = _tokenize(repo_file.rel_path)
            content_tokens = _tokenize(repo_file.content[:4000])
            path_overlap = len(goal_tokens & path_tokens)
            content_overlap = len(goal_tokens & content_tokens)
            if path_overlap == 0 and content_overlap == 0:
                continue
            python_bonus = 0.2 if repo_file.rel_path.endswith('.py') else 0.0
            score = float(content_overlap) + float(path_overlap * 2) + python_bonus
            scored.append((score, repo_file))

        scored.sort(key=lambda item: item[0], reverse=True)
        result: list[RelevantFile] = []
        for score, repo_file in scored[:top_k]:
            result.append(
                RelevantFile(
                    file_path=repo_file.rel_path,
                    score=score,
                    reason='token overlap with goal and path/content',
                )
            )
        return result
