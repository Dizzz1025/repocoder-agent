from __future__ import annotations

import json

from repocoder_agent import cli
from repocoder_agent.tools.mcp_client import MCPToolCallResult, MCPToolInfo


class FakeRuntime:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def list_tools(self, server_name: str):
        return [
            MCPToolInfo(
                server_name=server_name,
                name='search_docs',
                description='Search docs',
                input_schema={'type': 'object'},
            )
        ]

    def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        return MCPToolCallResult(
            server_name=server_name,
            tool_name=tool_name,
            content=[{'type': 'text', 'text': f'{tool_name}:{arguments}'}],
            structured_content={'ok': True},
        )


def test_cli_mcp_list_and_show(tmp_path, capsys) -> None:
    (tmp_path / '.mcp.json').write_text(
        json.dumps(
            {
                'mcpServers': {
                    'filesystem': {
                        'command': 'python',
                        'args': ['server.py'],
                    }
                }
            }
        ),
        encoding='utf-8',
    )

    exit_code = cli.main(['mcp', str(tmp_path), 'list'])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload['servers'][0]['name'] == 'filesystem'
    assert payload['servers'][0]['transport'] == 'stdio'

    exit_code = cli.main(['mcp', str(tmp_path), 'show', 'filesystem'])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload['server']['name'] == 'filesystem'
    assert payload['server']['command'] == 'python'


def test_cli_mcp_tools_and_call(tmp_path, capsys, monkeypatch) -> None:
    (tmp_path / '.mcp.json').write_text(
        json.dumps(
            {
                'mcpServers': {
                    'docs': {
                        'command': 'python',
                        'args': ['server.py'],
                    }
                }
            }
        ),
        encoding='utf-8',
    )
    monkeypatch.setattr(cli, 'MCPClientRuntime', FakeRuntime)

    exit_code = cli.main(['mcp', str(tmp_path), 'tools', 'docs'])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload['tools'][0]['name'] == 'search_docs'

    exit_code = cli.main(['mcp', str(tmp_path), 'call', 'docs', 'search_docs', '--arguments', '{"query": "mcp"}'])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload['result']['tool'] == 'search_docs'
    assert payload['result']['structured_content'] == {'ok': True}
