from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ..config import get_settings


class RepositoryHistoryStore:
    def __init__(self, repo_path: str):
        self.repo_root = Path(repo_path).resolve()
        settings = get_settings(start_dir=self.repo_root)
        self.db_path = (self.repo_root / settings.graph_db_path).resolve()
        if self.repo_root not in self.db_path.parents and self.db_path != self.repo_root:
            self.db_path = self.repo_root / '.repocoder' / 'graph_memory.db'

    def record_patch_event(
        self,
        file_path: str,
        operation: str,
        success: bool,
        message: str,
    ) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            self._ensure_schema(connection)
            connection.execute(
                'INSERT INTO patch_history (file_path, operation, success, message) VALUES (?, ?, ?, ?)',
                (file_path, operation, int(success), message),
            )
            connection.commit()

    def record_command_failure(self, command: str, stderr: str, stdout: str) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            self._ensure_schema(connection)
            connection.execute(
                'INSERT INTO command_failures (command, stderr, stdout) VALUES (?, ?, ?)',
                (command, stderr, stdout),
            )
            connection.commit()

    def patch_success_counts(self) -> dict[str, int]:
        return self._counts_from_query(
            'SELECT file_path, COUNT(*) FROM patch_history WHERE success = 1 GROUP BY file_path'
        )

    def patch_failure_counts(self) -> dict[str, int]:
        return self._counts_from_query(
            'SELECT file_path, COUNT(*) FROM patch_history WHERE success = 0 GROUP BY file_path'
        )

    def command_failure_counts(self) -> dict[str, int]:
        if not self.db_path.exists():
            return {}
        with sqlite3.connect(self.db_path) as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                'SELECT command, COUNT(*) FROM command_failures GROUP BY command'
            ).fetchall()
        return {row[0]: int(row[1]) for row in rows}

    def patch_history_events(self, success: bool | None = None) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        query = 'SELECT file_path, operation, success, message FROM patch_history'
        params: tuple[Any, ...] = ()
        if success is not None:
            query += ' WHERE success = ?'
            params = (int(success),)
        query += ' ORDER BY id DESC'
        with sqlite3.connect(self.db_path) as connection:
            self._ensure_schema(connection)
            rows = connection.execute(query, params).fetchall()
        return [
            {
                'file_path': row[0],
                'operation': row[1],
                'success': bool(row[2]),
                'message': row[3],
            }
            for row in rows
        ]

    def command_failure_events(self) -> list[dict[str, str]]:
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as connection:
            self._ensure_schema(connection)
            rows = connection.execute(
                'SELECT command, stderr, stdout FROM command_failures ORDER BY id DESC'
            ).fetchall()
        return [
            {
                'command': row[0],
                'stderr': row[1],
                'stdout': row[2],
            }
            for row in rows
        ]

    def _counts_from_query(self, query: str) -> dict[str, int]:
        if not self.db_path.exists():
            return {}
        with sqlite3.connect(self.db_path) as connection:
            self._ensure_schema(connection)
            rows = connection.execute(query).fetchall()
        return {row[0]: int(row[1]) for row in rows}

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS patch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                operation TEXT NOT NULL,
                success INTEGER NOT NULL,
                message TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS command_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT NOT NULL,
                stderr TEXT NOT NULL,
                stdout TEXT NOT NULL
            )
            """
        )
