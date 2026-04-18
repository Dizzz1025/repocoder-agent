from __future__ import annotations

import json
from pathlib import Path

from repocoder_agent import cli
from repocoder_agent.tools.adapters.local_tools import register_local_tools
from repocoder_agent.tools.registry import ToolRegistry


def test_tool_registry_lists_registered_tools() -> None:
    registry = ToolRegistry()
    register_local_tools(registry)

    tools = registry.list_tools()
    names = {tool.name for tool in tools}

    assert {'scan_repository', 'list_skills', 'show_skill'}.issubset(names)


def test_tool_registry_invokes_scan_repository(tmp_path: Path) -> None:
    (tmp_path / 'a.py').write_text("print('ok')\n", encoding='utf-8')
    registry = ToolRegistry()
    register_local_tools(registry)

    payload = registry.invoke('scan_repository', repository_path=str(tmp_path))

    assert payload['summary']['indexed_count'] >= 1


def test_cli_tools_list_and_show(capsys) -> None:
    exit_code = cli.main(['tools', 'list'])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert any(item['name'] == 'scan_repository' for item in payload['tools'])

    exit_code = cli.main(['tools', 'show', 'scan_repository'])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload['tool']['name'] == 'scan_repository'
    assert payload['tool']['schema']['type'] == 'object'
