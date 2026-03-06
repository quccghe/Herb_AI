# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional, List

from tools.mcp_client import MCPClient


class RelationAgent:
    """
    中药关系网络 Agent
    职责：
    1. 调 MCP 获取图谱子图
    2. 调 MCP 获取图谱摘要
    3. 返回给前端可直接渲染的 graph JSON
    """

    def __init__(self, mcp: MCPClient):
        self.mcp = mcp

    def run(
        self,
        herb_name: str,
        include_types: Optional[List[str]] = None,
        depth: int = 1,
        max_nodes_per_type: int = 20,
    ) -> Dict[str, Any]:
        herb_name = (herb_name or "").strip()
        if not herb_name:
            return {
                "type": "relation_graph",
                "ok": False,
                "error": "empty_herb_name",
                "graph": {"nodes": [], "edges": []},
                "summary": "",
            }

        graph = self.mcp.kg_subgraph(
            name=herb_name,
            depth=depth,
            include_types=include_types,
            max_nodes_per_type=max_nodes_per_type,
        )

        summary_resp = self.mcp.kg_graph_summary(name=herb_name)

        return {
            "type": "relation_graph",
            "ok": bool(graph.get("ok")),
            "name": herb_name,
            "graph": {
                "center": graph.get("center", herb_name),
                "nodes": graph.get("nodes", []),
                "edges": graph.get("edges", []),
                "meta": graph.get("meta", {}),
            },
            "summary": summary_resp.get("summary", ""),
            "stats": summary_resp.get("stats", {}),
            "error": graph.get("error") if not graph.get("ok") else "",
        }