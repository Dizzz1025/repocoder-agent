from __future__ import annotations

import json

from repocoder_agent.tools.mcp_config import MCPConfigLoader


def test_mcp_loader_reads_stdio_server(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('TEST_TOKEN', 'abc123')
    (tmp_path / '.mcp.json').write_text(
        json.dumps(
            {
                'mcpServers': {
                    'filesystem': {
                        'command': 'npx',
                        'args': ['-y', '@modelcontextprotocol/server-filesystem', '${TEST_TOKEN}'],
                        'env': {'TOKEN': '${TEST_TOKEN}'},
                    }
                }
            }
        ),
        encoding='utf-8',
    )

    loader = MCPConfigLoader(str(tmp_path))
    servers = loader.list_servers()

    assert servers
    assert servers[0].name == 'filesystem'
    assert servers[0].transport == 'stdio'
    assert servers[0].args[-1] == 'abc123'
    assert servers[0].env == {'TOKEN': 'abc123'}


def test_mcp_loader_reads_http_server(tmp_path) -> None:
    (tmp_path / '.mcp.json').write_text(
        json.dumps(
            {
                'mcpServers': {
                    'docs': {
                        'type': 'http',
                        'url': 'https://example.invalid/mcp',
                        'headers': {'Authorization': 'Bearer token'},
                    }
                }
            }
        ),
        encoding='utf-8',
    )

    loader = MCPConfigLoader(str(tmp_path))
    server = loader.get_server('docs')

    assert server is not None
    assert server.transport == 'http'
    assert server.url == 'https://example.invalid/mcp'
    assert server.headers == {'Authorization': 'Bearer token'}
