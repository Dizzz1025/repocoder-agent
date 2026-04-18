from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .mcp_config import MCPConfigLoader, MCPServerSpec


@dataclass(frozen=True)
class MCPToolInfo:
    server_name: str
    name: str
    description: str | None
    input_schema: dict[str, Any] | None


@dataclass(frozen=True)
class MCPToolCallResult:
    server_name: str
    tool_name: str
    content: list[dict[str, Any]]
    structured_content: Any


class MCPClientRuntime:
    def __init__(self, repo_path: str):
        self.config_loader = MCPConfigLoader(repo_path)

    def list_tools(self, server_name: str) -> list[MCPToolInfo]:
        spec = self._require_server(server_name)
        return asyncio.run(self._list_tools_async(spec))

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        spec = self._require_server(server_name)
        return asyncio.run(self._call_tool_async(spec, tool_name, arguments))

    def _require_server(self, server_name: str) -> MCPServerSpec:
        server = self.config_loader.get_server(server_name)
        if server is None:
            raise KeyError(f"Unknown MCP server: {server_name}")
        return server

    async def _list_tools_async(self, spec: MCPServerSpec) -> list[MCPToolInfo]:
        async with self._session(spec) as session:
            result = await session.list_tools()
            tools = getattr(result, 'tools', result)
            return [
                MCPToolInfo(
                    server_name=spec.name,
                    name=getattr(tool, 'name', ''),
                    description=getattr(tool, 'description', None),
                    input_schema=getattr(tool, 'inputSchema', None),
                )
                for tool in tools
            ]

    async def _call_tool_async(
        self,
        spec: MCPServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MCPToolCallResult:
        async with self._session(spec) as session:
            result = await session.call_tool(tool_name, arguments=arguments)
            content_blocks = []
            for item in getattr(result, 'content', []):
                block = {
                    'type': getattr(item, 'type', None),
                }
                if hasattr(item, 'text'):
                    block['text'] = item.text
                if hasattr(item, 'data'):
                    block['data'] = item.data
                content_blocks.append(block)
            return MCPToolCallResult(
                server_name=spec.name,
                tool_name=tool_name,
                content=content_blocks,
                structured_content=getattr(result, 'structuredContent', None),
            )

    def _sdk(self):
        try:
            from mcp import ClientSession, StdioServerParameters  # type: ignore
            from mcp.client.stdio import stdio_client  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The official Python MCP SDK is not installed. Install the 'mcp' package to use MCP runtime features."
            ) from exc

        streamable_http_client = None
        sse_client = None
        try:
            from mcp.client.streamable_http import streamable_http_client  # type: ignore
        except ImportError:
            pass
        try:
            from mcp.client.sse import sse_client  # type: ignore
        except ImportError:
            pass

        return {
            'ClientSession': ClientSession,
            'StdioServerParameters': StdioServerParameters,
            'stdio_client': stdio_client,
            'streamable_http_client': streamable_http_client,
            'sse_client': sse_client,
        }

    def _session(self, spec: MCPServerSpec):
        runtime = self

        class _SessionContext:
            async def __aenter__(self_inner):
                sdk = runtime._sdk()
                ClientSession = sdk['ClientSession']

                if spec.transport == 'stdio':
                    StdioServerParameters = sdk['StdioServerParameters']
                    runtime._transport_context = sdk['stdio_client'](
                        StdioServerParameters(
                            command=spec.command or '',
                            args=list(spec.args),
                            env=spec.env,
                        )
                    )
                    streams = await runtime._transport_context.__aenter__()
                    read_stream, write_stream = streams[0], streams[1]
                elif spec.transport == 'http':
                    client_factory = sdk['streamable_http_client']
                    if client_factory is None:
                        raise RuntimeError('The installed MCP SDK does not support streamable HTTP transport.')
                    runtime._transport_context = client_factory(spec.url or '', headers=spec.headers)
                    streams = await runtime._transport_context.__aenter__()
                    read_stream, write_stream = streams[0], streams[1]
                elif spec.transport == 'sse':
                    client_factory = sdk['sse_client']
                    if client_factory is None:
                        raise RuntimeError('The installed MCP SDK does not support SSE transport.')
                    runtime._transport_context = client_factory(spec.url or '', headers=spec.headers)
                    streams = await runtime._transport_context.__aenter__()
                    read_stream, write_stream = streams[0], streams[1]
                else:
                    raise RuntimeError(f'Unsupported MCP transport: {spec.transport}')

                runtime._session_context = ClientSession(read_stream, write_stream)
                session = await runtime._session_context.__aenter__()
                await session.initialize()
                return session

            async def __aexit__(self_inner, exc_type, exc, tb):
                if getattr(runtime, '_session_context', None) is not None:
                    await runtime._session_context.__aexit__(exc_type, exc, tb)
                    runtime._session_context = None
                if getattr(runtime, '_transport_context', None) is not None:
                    await runtime._transport_context.__aexit__(exc_type, exc, tb)
                    runtime._transport_context = None
                return False

        return _SessionContext()
