from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Literal

from ..repository import RepoSnapshot

NodeType = Literal["file", "function", "class", "module"]
EdgeType = Literal["defines", "imports"]


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: NodeType
    name: str
    file_path: str | None = None # 节点属于哪个文件
    lineno: int | None = None # 它在文件里的行号


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

    def symbol_names_by_file(self) -> dict[str, set[str]]:
        # return: {
        #     "dry_run.py": {"DryRunSandbox", "build_sandbox", "validate_patch"},
        #     "patcher.py": {"PatchApplier", "apply"},
        # }
        result: dict[str, set[str]] = {}
        for node in self.nodes:
            if node.node_type not in {"function", "class"} or node.file_path is None:
                continue
            result.setdefault(node.file_path, set()).add(node.name)
        return result

    def imported_modules_by_file(self) -> dict[str, set[str]]:
        # return: {
        #     "graph_builder.py": {"ast", "dataclasses", "typing"},
        # }
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

    def summary(self) -> dict[str, int]:
        counts = {"files": 0, "functions": 0, "classes": 0, "modules": 0, "edges": len(self.edges)}
        for node in self.nodes:
            if node.node_type == "file":
                counts["files"] += 1
            elif node.node_type == "function":
                counts["functions"] += 1
            elif node.node_type == "class":
                counts["classes"] += 1
            elif node.node_type == "module":
                counts["modules"] += 1
        return counts


class RepositoryGraphBuilder:
    def build_from_snapshot(self, snapshot: RepoSnapshot) -> RepositoryGraph:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        module_nodes: dict[str, GraphNode] = {} # 存所有模块节点，用dict避免重复创建同名模块节点

        for repo_file in snapshot.files:
            file_node = GraphNode(
                node_id=f"file:{repo_file.rel_path}",
                node_type="file",
                name=repo_file.rel_path,
                file_path=repo_file.rel_path,
            )
            nodes.append(file_node)

            if not repo_file.rel_path.endswith('.py'):
                continue

            try:
                tree = ast.parse(repo_file.content)
            except SyntaxError:
                continue

            for symbol_node in self._extract_symbol_nodes(repo_file.rel_path, tree):
                nodes.append(symbol_node)
                edges.append(
                    GraphEdge(
                        source_id=file_node.node_id,
                        target_id=symbol_node.node_id,
                        edge_type="defines",
                    )
                )

            for module_name in self._extract_imports(tree):
                if module_name not in module_nodes:
                    module_nodes[module_name] = GraphNode(
                        node_id=f"module:{module_name}",
                        node_type="module",
                        name=module_name,
                    )
                edges.append(
                    GraphEdge(
                        source_id=file_node.node_id,
                        target_id=module_nodes[module_name].node_id,
                        edge_type="imports",
                    )
                )

        nodes.extend(module_nodes.values())
        return RepositoryGraph(
            repo_path=snapshot.summary.repo_path,
            nodes=tuple(nodes),
            edges=tuple(edges),
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
