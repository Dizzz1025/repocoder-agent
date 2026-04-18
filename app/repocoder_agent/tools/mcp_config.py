from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MCPTransport = Literal['stdio', 'http', 'sse']
_ENV_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}')


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    transport: MCPTransport
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None


class MCPConfigLoader:
    def __init__(self, repo_path: str):
        self.repo_root = Path(repo_path).resolve()
        self.config_path = self.repo_root / '.mcp.json'

    def list_servers(self) -> list[MCPServerSpec]:
        payload = self._load_payload()
        servers_raw = payload.get('mcpServers', {}) if isinstance(payload, dict) else {}
        if not isinstance(servers_raw, dict):
            return []
        specs: list[MCPServerSpec] = []
        for name, value in servers_raw.items():
            if not isinstance(value, dict):
                continue
            spec = self._parse_server(name, value)
            if spec is not None:
                specs.append(spec)
        return sorted(specs, key=lambda item: item.name)

    def get_server(self, name: str) -> MCPServerSpec | None:
        for server in self.list_servers():
            if server.name == name:
                return server
        return None

    def _load_payload(self) -> dict[str, Any]:
        if not self.config_path.exists() or not self.config_path.is_file():
            return {}
        try:
            payload = json.loads(self.config_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _parse_server(self, name: str, value: dict[str, Any]) -> MCPServerSpec | None:
        transport_value = str(value.get('transport') or value.get('type') or '').strip().lower()
        if not transport_value:
            if 'command' in value:
                transport_value = 'stdio'
            elif 'url' in value:
                transport_value = 'http'
            else:
                return None
        if transport_value not in {'stdio', 'http', 'sse'}:
            return None

        expanded = self._expand_value(value)
        command = expanded.get('command') if isinstance(expanded.get('command'), str) else None
        args = tuple(str(item) for item in expanded.get('args', []) if isinstance(item, (str, int, float)))
        env = expanded.get('env') if isinstance(expanded.get('env'), dict) else None
        headers = expanded.get('headers') if isinstance(expanded.get('headers'), dict) else None
        url = expanded.get('url') if isinstance(expanded.get('url'), str) else None

        return MCPServerSpec(
            name=name,
            transport=transport_value,  # type: ignore[arg-type]
            command=command,
            args=args,
            env={str(k): str(v) for k, v in env.items()} if env else None,
            url=url,
            headers={str(k): str(v) for k, v in headers.items()} if headers else None,
        )

    def _expand_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return _ENV_PATTERN.sub(self._replace_match, value)
        if isinstance(value, list):
            return [self._expand_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._expand_value(v) for k, v in value.items()}
        return value

    def _replace_match(self, match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(3) or ''
        return os.getenv(name, default)
