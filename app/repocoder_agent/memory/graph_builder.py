from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Literal

from ..repository import RepoFile, RepoSnapshot

NodeType = Literal["file", "function", "class", "module", "call"]
EdgeType = Literal["defines", "imports", "calls"]


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: NodeType
    name: str
    file_path: str | None = None
    lineno: int | None = None


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType


@dataclass(frozen=True)
class RepositoryGraph:
    repo_path: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    file_hashes: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def file_hash_map(self) -> dict[str, str]:
        return dict(self.file_hashes)

    def symbol_names_by_file(self) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for node in self.nodes:
            if node.node_type not in {"function", "class"} or node.file_path is None:
                continue
            result.setdefault(node.file_path, set()).add(node.name)
        return result

    def imported_modules_by_file(self) -> dict[str, set[str]]:
        nodes_by_id = {node.node_id: node for node in self.nodes}
        result: dict[str, set[str]] = {}
        for edge in self.edges:
            if edge.edge_type != "imports":
                continue
            source = nodes_by_id.get(edge.source_id)
            target = nodes_by_id.get(edge.target_id)
            if source is None or target is None or source.file_path is None:
                continue
            result.setdefault(source.file_path, set()).add(target.name)
        return result

    def call_names_by_file(self) -> dict[str, set[str]]:
        nodes_by_id = {node.node_id: node for node in self.nodes}
        result: dict[str, set[str]] = {}
        for edge in self.edges:
            if edge.edge_type != "calls":
                continue
            source = nodes_by_id.get(edge.source_id)
            target = nodes_by_id.get(edge.target_id)
            if source is None or target is None or source.file_path is None:
                continue
            result.setdefault(source.file_path, set()).add(target.name)
        return result

    def summary(self) -> dict[str, int]:
        counts = {
            "files": 0,
            "functions": 0,
            "classes": 0,
            "modules": 0,
            "calls": 0,
            "edges": len(self.edges),
        }
        for node in self.nodes:
            if node.node_type == "file":
                counts["files"] += 1
            elif node.node_type == "function":
                counts["functions"] += 1
            elif node.node_type == "class":
                counts["classes"] += 1
            elif node.node_type == "module":
                counts["modules"] += 1
            elif node.node_type == "call":
                counts["calls"] += 1
        return counts


class RepositoryGraphBuilder:
    def build_from_snapshot(self, snapshot: RepoSnapshot) -> RepositoryGraph:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        file_hashes: list[tuple[str, str]] = []

        for repo_file in snapshot.files:
            file_nodes, file_edges, file_hash = self.build_file_subgraph(repo_file)
            nodes.extend(file_nodes)
            edges.extend(file_edges)
            file_hashes.append(file_hash)

        return self._finalize_graph(
            repo_path=snapshot.summary.repo_path,
            nodes=nodes,
            edges=edges,
            file_hashes=file_hashes,
        )

    def update_graph(
        self,
        base_graph: RepositoryGraph,
        snapshot: RepoSnapshot,
        changed_files: list[str],
        removed_files: list[str],
    ) -> RepositoryGraph:
        changed_or_removed = set(changed_files) | set(removed_files)
        snapshot_files = {item.rel_path: item for item in snapshot.files}
        existing_nodes = {node.node_id: node for node in base_graph.nodes}

        kept_nodes = [
            node
            for node in base_graph.nodes
            if node.file_path not in changed_or_removed
        ]
        kept_node_ids = {node.node_id for node in kept_nodes}
        kept_edges: list[GraphEdge] = []
        for edge in base_graph.edges:
            source = existing_nodes.get(edge.source_id)
            target = existing_nodes.get(edge.target_id)
            if (source is not None and source.file_path in changed_or_removed) or (
                target is not None and target.file_path in changed_or_removed
            ):
                continue
            if edge.source_id in kept_node_ids and (edge.target_id in kept_node_ids or edge.target_id.startswith("module:")): #module节点的file_path通常是None，所以特殊处理一下
                kept_edges.append(edge)

        updated_hashes = base_graph.file_hash_map()
        for path in removed_files:
            updated_hashes.pop(path, None)
        for path in changed_files:
            repo_file = snapshot_files.get(path)
            if repo_file is None:
                continue
            file_nodes, file_edges, file_hash = self.build_file_subgraph(repo_file)
            kept_nodes.extend(file_nodes)
            kept_edges.extend(file_edges)
            updated_hashes[file_hash[0]] = file_hash[1] # file_hash[0]表示文件路径, file_hash[1]表示这个文件当前的内容

        return self._finalize_graph(
            repo_path=base_graph.repo_path,
            nodes=kept_nodes,
            edges=kept_edges,
            file_hashes=sorted(updated_hashes.items()),
        )

    def build_file_subgraph(
        self,
        repo_file: RepoFile,
    ) -> tuple[list[GraphNode], list[GraphEdge], tuple[str, str]]:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        file_node = GraphNode(
            node_id=f"file:{repo_file.rel_path}",
            node_type="file",
            name=repo_file.rel_path,
            file_path=repo_file.rel_path,
        )
        nodes.append(file_node)

        if repo_file.rel_path.endswith('.py'):
            try:
                tree = ast.parse(repo_file.content)
            except SyntaxError:
                tree = None

            if tree is not None:
                for symbol_node in self._extract_symbol_nodes(repo_file.rel_path, tree):
                    nodes.append(symbol_node)
                    edges.append(
                        GraphEdge(
                            source_id=file_node.node_id,
                            target_id=symbol_node.node_id,
                            edge_type="defines",
                        )
                    )

                for call_node in self._extract_call_nodes(repo_file.rel_path, tree):
                    nodes.append(call_node)
                    edges.append(
                        GraphEdge(
                            source_id=file_node.node_id,
                            target_id=call_node.node_id,
                            edge_type="calls",
                        )
                    )

                for module_name in self._extract_imports(tree):
                    module_node = GraphNode(
                        node_id=f"module:{module_name}",
                        node_type="module",
                        name=module_name,
                    )
                    nodes.append(module_node)
                    edges.append(
                        GraphEdge(
                            source_id=file_node.node_id,
                            target_id=module_node.node_id,
                            edge_type="imports",
                        )
                    )

        return nodes, edges, (repo_file.rel_path, repo_file.content_hash)

    def _finalize_graph(
        self,
        repo_path: str,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        file_hashes: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    ) -> RepositoryGraph:
        node_map: dict[str, GraphNode] = {}
        for node in nodes:
            node_map[node.node_id] = node

        edge_map: dict[tuple[str, str, str], GraphEdge] = {}
        for edge in edges:
            edge_map[(edge.source_id, edge.target_id, edge.edge_type)] = edge

        referenced_module_ids = {
            edge.target_id
            for edge in edge_map.values()
            if edge.edge_type == "imports"
        }
        final_nodes = [
            node
            for node in node_map.values()
            if node.node_type != "module" or node.node_id in referenced_module_ids
        ]
        valid_node_ids = {node.node_id for node in final_nodes}
        final_edges = [
            edge
            for edge in edge_map.values()
            if edge.source_id in valid_node_ids and edge.target_id in valid_node_ids
        ]

        return RepositoryGraph(
            repo_path=repo_path,
            nodes=tuple(sorted(final_nodes, key=lambda item: item.node_id)),
            edges=tuple(sorted(final_edges, key=lambda item: (item.source_id, item.target_id, item.edge_type))),
            file_hashes=tuple(sorted(file_hashes)),
        )

    def _extract_symbol_nodes(self, rel_path: str, tree: ast.AST) -> list[GraphNode]:
        nodes: list[GraphNode] = []
        for item in ast.walk(tree):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nodes.append(
                    GraphNode(
                        node_id=f"function:{rel_path}:{item.name}:{getattr(item, 'lineno', 0)}",
                        node_type="function",
                        name=item.name,
                        file_path=rel_path,
                        lineno=getattr(item, 'lineno', None),
                    )
                )
            elif isinstance(item, ast.ClassDef):
                nodes.append(
                    GraphNode(
                        node_id=f"class:{rel_path}:{item.name}:{getattr(item, 'lineno', 0)}",
                        node_type="class",
                        name=item.name,
                        file_path=rel_path,
                        lineno=getattr(item, 'lineno', None),
                    )
                )
        return nodes

    def _extract_call_nodes(self, rel_path: str, tree: ast.AST) -> list[GraphNode]:
        nodes: list[GraphNode] = []
        for item in ast.walk(tree):
            if not isinstance(item, ast.Call):
                continue
            call_name = self._call_name(item)
            if call_name is None:
                continue
            nodes.append(
                GraphNode(
                    node_id=f"call:{rel_path}:{call_name}:{getattr(item, 'lineno', 0)}",
                    node_type="call",
                    name=call_name,
                    file_path=rel_path,
                    lineno=getattr(item, 'lineno', None),
                )
            )
        return nodes

    def _call_name(self, call: ast.Call) -> str | None:
        func = call.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    def _extract_imports(self, tree: ast.AST) -> list[str]:
        imports: list[str] = []
        for item in ast.walk(tree):
            if isinstance(item, ast.Import):
                for alias in item.names:
                    imports.append(alias.name)
            elif isinstance(item, ast.ImportFrom):
                if item.module:
                    imports.append(item.module)
        return imports
