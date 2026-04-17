from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..repository import RepoSnapshot
from .graph_builder import GraphEdge, GraphNode, RepositoryGraph, RepositoryGraphBuilder


class RepositoryGraphStore:
    def __init__(self, repo_path: str):
        self.repo_root = Path(repo_path).resolve()
        settings = get_settings(start_dir=self.repo_root)
        self.db_path = (self.repo_root / settings.graph_db_path).resolve()
        if self.repo_root not in self.db_path.parents and self.db_path != self.repo_root:
            self.db_path = self.repo_root / '.repocoder' / 'graph_memory.db'

    def save(self, graph: RepositoryGraph) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            self._ensure_schema(connection)
            connection.execute('DELETE FROM nodes')
            connection.execute('DELETE FROM edges')
            connection.execute('DELETE FROM file_hashes')
            connection.executemany(
                'INSERT INTO nodes (node_id, node_type, name, file_path, lineno) VALUES (?, ?, ?, ?, ?)',
                [
                    (node.node_id, node.node_type, node.name, node.file_path, node.lineno)
                    for node in graph.nodes
                ],
            )
            connection.executemany(
                'INSERT INTO edges (source_id, target_id, edge_type) VALUES (?, ?, ?)',
                [
                    (edge.source_id, edge.target_id, edge.edge_type)
                    for edge in graph.edges
                ],
            )
            connection.executemany(
                'INSERT INTO file_hashes (file_path, content_hash) VALUES (?, ?)',
                list(graph.file_hashes),
            )
            connection.commit()

    def load(self) -> RepositoryGraph | None:
        if not self.db_path.exists():
            return None
        with sqlite3.connect(self.db_path) as connection:
            self._ensure_schema(connection)
            node_rows = connection.execute(
                'SELECT node_id, node_type, name, file_path, lineno FROM nodes ORDER BY node_id'
            ).fetchall()
            edge_rows = connection.execute(
                'SELECT source_id, target_id, edge_type FROM edges ORDER BY rowid'
            ).fetchall()
            file_hash_rows = connection.execute(
                'SELECT file_path, content_hash FROM file_hashes ORDER BY file_path'
            ).fetchall()
        if not node_rows:
            return None
        return RepositoryGraph(
            repo_path=str(self.repo_root),
            nodes=tuple(
                GraphNode(
                    node_id=row[0],
                    node_type=row[1],
                    name=row[2],
                    file_path=row[3],
                    lineno=row[4],
                )
                for row in node_rows
            ),
            edges=tuple(
                GraphEdge(
                    source_id=row[0],
                    target_id=row[1],
                    edge_type=row[2],
                )
                for row in edge_rows
            ),
            file_hashes=tuple((row[0], row[1]) for row in file_hash_rows),
        )

    def diff(self, snapshot: RepoSnapshot) -> dict[str, Any]:
        existing = self.load()
        current_hashes = {item.rel_path: item.content_hash for item in snapshot.files}
        if existing is None: # 说明这是第一次建图, 所以不能复用旧图, 需要重新建图
            return {
                'reused': False,
                'changed_files': sorted(current_hashes.keys()),
                'removed_files': [],
                'unchanged_files': 0,
            }
        stored_hashes = existing.file_hash_map()
        changed_files = [
            path for path, content_hash in current_hashes.items()
            if stored_hashes.get(path) != content_hash
        ]
        removed_files = [path for path in stored_hashes if path not in current_hashes]
        unchanged_files = len(current_hashes) - len(changed_files)
        return {
            'reused': not changed_files and not removed_files,
            'changed_files': sorted(changed_files),
            'removed_files': sorted(removed_files),
            'unchanged_files': unchanged_files,
        }

    def refresh(
        self,
        snapshot: RepoSnapshot,
        builder: RepositoryGraphBuilder,
    ) -> tuple[RepositoryGraph, dict[str, Any]]:
        diff = self.diff(snapshot)
        loaded = self.load()
        if diff['reused'] and loaded is not None:
            return loaded, diff

        if loaded is None:
            graph = builder.build_from_snapshot(snapshot)
        else:
            graph = builder.update_graph(
                base_graph=loaded,
                snapshot=snapshot,
                changed_files=diff['changed_files'],
                removed_files=diff['removed_files'],
            )
        self.save(graph)
        return graph, diff

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                name TEXT NOT NULL,
                file_path TEXT,
                lineno INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL
            )
            """
        )
