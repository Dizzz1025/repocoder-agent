from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..config import get_settings

HookEvent = Literal["pre_patch", "post_patch", "pre_command", "post_command", "run_stop"]
HookAction = Literal["allow", "block", "log"]


@dataclass(frozen=True)
class HookResult:
    event: HookEvent
    action: HookAction
    message: str
    matched_rule: str | None = None
    blocked: bool = False


class HookManager:
    def __init__(self, repo_path: str):
        self.repo_root = Path(repo_path).resolve()
        settings = get_settings(start_dir=self.repo_root)
        self.hooks_path = (self.repo_root / settings.hooks_path).resolve()
        if self.repo_root not in self.hooks_path.parents and self.hooks_path != self.repo_root:
            self.hooks_path = self.repo_root / '.repocoder' / 'hooks.json'
        self.rules = self._load_rules()

    def handle(self, event: HookEvent, context: dict[str, Any]) -> list[HookResult]:
        results: list[HookResult] = []
        for rule in self.rules.get(event, []):
            if not self._matches(rule, context):
                continue
            action = str(rule.get('action', 'log'))
            message = str(rule.get('message', f'{event} hook matched'))
            blocked = action == 'block'
            results.append(
                HookResult(
                    event=event,
                    action=action if action in {'allow', 'block', 'log'} else 'log',
                    message=message,
                    matched_rule=str(rule.get('name')) if rule.get('name') else None,
                    blocked=blocked,
                )
            )
        return results # 注意这里的results返回的是一个list

    def _load_rules(self) -> dict[str, list[dict[str, Any]]]:
        if not self.hooks_path.exists():
            return {}
        try:
            payload = json.loads(self.hooks_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        normalized: dict[str, list[dict[str, Any]]] = {}
        for key, value in payload.items():
            if key not in {'pre_patch', 'post_patch', 'pre_command', 'post_command', 'run_stop'}:
                continue
            if isinstance(value, list):
                normalized[key] = [item for item in value if isinstance(item, dict)]
        return normalized

    def _matches(self, rule: dict[str, Any], context: dict[str, Any]) -> bool:
        target_file_contains = rule.get('target_file_contains')
        if target_file_contains is not None:
            file_path = str(context.get('file_path', ''))
            if str(target_file_contains) not in file_path:
                return False

        command_contains = rule.get('command_contains')
        if command_contains is not None:
            command = str(context.get('command', ''))
            if str(command_contains) not in command:
                return False

        message_contains = rule.get('message_contains')
        if message_contains is not None:
            message = str(context.get('message', ''))
            if str(message_contains) not in message:
                return False

        operation_is = rule.get('operation_is')
        if operation_is is not None:
            operation = str(context.get('operation', ''))
            if str(operation_is) != operation:
                return False

        return True
