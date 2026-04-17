from __future__ import annotations

import sqlite3
from pathlib import Path

from ..config import get_settings
from .graph_builder import GraphEdge, GraphNode, RepositoryGraph


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
        )

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
