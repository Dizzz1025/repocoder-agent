from __future__ import annotations

from ..mcp_client import MCPClientRuntime
from ..registry import ToolRegistry, ToolSpec


def register_mcp_server_tools(registry: ToolRegistry, runtime: MCPClientRuntime, server_name: str) -> None:
    for tool in runtime.list_tools(server_name):
        registry.register(
            ToolSpec(
                name=f'mcp.{server_name}.{tool.name}',
                description=tool.description or f'MCP tool {tool.name} from server {server_name}',
                schema=tool.input_schema or {'type': 'object'},
                handler=lambda **kwargs: runtime.call_tool(server_name, tool.name, kwargs),
            )
        )
