from __future__ import annotations

from types import SimpleNamespace

from repocoder_agent.tools.mcp_client import MCPClientRuntime, MCPToolCallResult


class FakeSession:
    async def initialize(self):
        return None

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name='search_docs',
                    description='Search docs',
                    inputSchema={'type': 'object'},
                )
            ]
        )

    async def call_tool(self, name, arguments):
        return SimpleNamespace(
            content=[SimpleNamespace(type='text', text=f'{name}:{arguments}')],
            structuredContent={'ok': True},
        )


class FakeAsyncContext:
    async def __aenter__(self):
        return FakeSession()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeRuntime(MCPClientRuntime):
    def _session(self, spec):
        return FakeAsyncContext()


def test_mcp_runtime_lists_tools(tmp_path) -> None:
    (tmp_path / '.mcp.json').write_text('{"mcpServers": {"docs": {"command": "python", "args": ["server.py"]}}}', encoding='utf-8')
    runtime = FakeRuntime(str(tmp_path))

    tools = runtime.list_tools('docs')

    assert tools
    assert tools[0].name == 'search_docs'
    assert tools[0].server_name == 'docs'


def test_mcp_runtime_calls_tool(tmp_path) -> None:
    (tmp_path / '.mcp.json').write_text('{"mcpServers": {"docs": {"command": "python", "args": ["server.py"]}}}', encoding='utf-8')
    runtime = FakeRuntime(str(tmp_path))

    result = runtime.call_tool('docs', 'search_docs', {'query': 'mcp'})

    assert result.server_name == 'docs'
    assert result.tool_name == 'search_docs'
    assert result.structured_content == {'ok': True}
    assert result.content[0]['text'] == "search_docs:{'query': 'mcp'}"
