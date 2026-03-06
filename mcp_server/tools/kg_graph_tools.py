# -*- coding: utf-8 -*-
"""
KG Graph Tools
用于中药关系网络可视化后端

功能：
1. kg_subgraph：获取某味中药的一阶关系网络
2. kg_relation_paths：获取两味中药之间的最短路径
3. kg_graph_summary：生成图谱摘要（规则版）

依赖：
- 复用 mcp_server/tools/kg_tools.py 中的 get_driver()
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict

from mcp_server.tools.kg_tools import get_driver


DEFAULT_INCLUDE_TYPES = [
    "Category",
    "Meridian",
    "TasteProperty",
    "Efficacy",
    "Disease",
]


def _node_id(label: str, name: str) -> str:
    return f"{label}::{name}"


def _extract_main_label(labels: List[str]) -> str:
    if not labels:
        return "Unknown"
    # 尽量保持和你的图谱标签一致
    priority = ["Medicine", "Category", "Meridian", "TasteProperty", "Efficacy", "Disease"]
    for p in priority:
        if p in labels:
            return p
    return labels[0]


def _serialize_node(node, labels: List[str]) -> Dict[str, Any]:
    props = dict(node)
    label = _extract_main_label(labels)
    name = props.get("name", "")
    return {
        "id": _node_id(label, name),
        "label": name,
        "type": label,
        "properties": props,
    }


def _serialize_edge(source_label: str, source_name: str, target_label: str, target_name: str, rel_type: str, rel_props: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": _node_id(source_label, source_name),
        "target": _node_id(target_label, target_name),
        "type": rel_type,
        "properties": rel_props or {},
    }


def tool_kg_subgraph(
    name: str,
    depth: int = 1,
    include_types: Optional[List[str]] = None,
    max_nodes_per_type: int = 20,
) -> Dict[str, Any]:
    """
    获取某个中药的一阶关系网络
    第一版建议 depth=1，避免图过大
    """
    name = (name or "").strip()
    if not name:
        return {"ok": False, "error": "empty_name", "center": "", "nodes": [], "edges": []}

    include_types = include_types or DEFAULT_INCLUDE_TYPES
    max_nodes_per_type = max(1, min(int(max_nodes_per_type), 100))

    try:
        driver = get_driver()
    except Exception as e:
        return {"ok": False, "error": str(e), "center": name, "nodes": [], "edges": []}

    try:
        with driver.session() as s:
            # 先找中心节点
            center_rec = s.run(
                """
                MATCH (m:Medicine)
                WHERE m.name = $name OR m.alias = $name
                RETURN m, labels(m) AS labels
                LIMIT 1
                """,
                name=name,
            ).single()

            if not center_rec:
                return {"ok": False, "error": "medicine_not_found", "center": name, "nodes": [], "edges": []}

            center_node = center_rec["m"]
            center_labels = center_rec["labels"]
            center_props = dict(center_node)
            center_name = center_props.get("name", name)

            # 一阶网络
            recs = s.run(
                """
                MATCH (m:Medicine {name:$name})-[r]->(n)
                WHERE any(label IN labels(n) WHERE label IN $include_types)
                RETURN m, labels(m) AS m_labels, r, type(r) AS rel_type, n, labels(n) AS n_labels
                LIMIT 500
                """,
                name=center_name,
                include_types=include_types,
            )

            nodes: List[Dict[str, Any]] = []
            edges: List[Dict[str, Any]] = []

            node_seen: Set[str] = set()
            edge_seen: Set[Tuple[str, str, str]] = set()
            type_counter = defaultdict(int)

            # 先放中心节点
            center_obj = _serialize_node(center_node, center_labels)
            nodes.append(center_obj)
            node_seen.add(center_obj["id"])

            for rec in recs:
                m = rec["m"]
                m_labels = rec["m_labels"]
                r = rec["r"]
                rel_type = rec["rel_type"]
                n = rec["n"]
                n_labels = rec["n_labels"]

                m_obj = _serialize_node(m, m_labels)
                n_obj = _serialize_node(n, n_labels)

                # 对邻居类型做数量限制
                n_type = n_obj["type"]
                if type_counter[n_type] >= max_nodes_per_type:
                    continue

                if n_obj["id"] not in node_seen:
                    nodes.append(n_obj)
                    node_seen.add(n_obj["id"])
                    type_counter[n_type] += 1

                edge_key = (m_obj["id"], n_obj["id"], rel_type)
                if edge_key not in edge_seen:
                    rel_props = dict(r)
                    edges.append(
                        _serialize_edge(
                            source_label=m_obj["type"],
                            source_name=m_obj["label"],
                            target_label=n_obj["type"],
                            target_name=n_obj["label"],
                            rel_type=rel_type,
                            rel_props=rel_props,
                        )
                    )
                    edge_seen.add(edge_key)

            return {
                "ok": True,
                "center": center_name,
                "nodes": nodes,
                "edges": edges,
                "meta": {
                    "depth": depth,
                    "include_types": include_types,
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                },
            }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "center": name,
            "nodes": [],
            "edges": [],
        }


def tool_kg_relation_paths(
    source: str,
    target: str,
    max_hops: int = 3,
) -> Dict[str, Any]:
    """
    获取两味中药之间的最短路径
    """
    source = (source or "").strip()
    target = (target or "").strip()
    max_hops = max(1, min(int(max_hops), 6))

    if not source or not target:
        return {"ok": False, "error": "empty_source_or_target", "paths": []}

    try:
        driver = get_driver()
    except Exception as e:
        return {"ok": False, "error": str(e), "paths": []}

    try:
        with driver.session() as s:
            recs = s.run(
                f"""
                MATCH p = shortestPath((a:Medicine {{name:$source}})-[*..{max_hops}]-(b:Medicine {{name:$target}}))
                RETURN p
                LIMIT 5
                """,
                source=source,
                target=target,
            )

            paths = []
            for rec in recs:
                p = rec["p"]
                if not p:
                    continue

                path_nodes = []
                path_edges = []

                for node in p.nodes:
                    labels = list(node.labels)
                    path_nodes.append(_serialize_node(node, labels))

                for rel in p.relationships:
                    start_node = rel.start_node
                    end_node = rel.end_node
                    start_label = _extract_main_label(list(start_node.labels))
                    end_label = _extract_main_label(list(end_node.labels))
                    path_edges.append(
                        _serialize_edge(
                            source_label=start_label,
                            source_name=start_node.get("name", ""),
                            target_label=end_label,
                            target_name=end_node.get("name", ""),
                            rel_type=rel.type,
                            rel_props=dict(rel),
                        )
                    )

                paths.append({
                    "nodes": path_nodes,
                    "edges": path_edges,
                })

            return {
                "ok": True,
                "source": source,
                "target": target,
                "paths": paths,
            }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "source": source,
            "target": target,
            "paths": [],
        }


def tool_kg_graph_summary(name: str) -> Dict[str, Any]:
    """
    根据一阶子图生成规则摘要
    不用 LLM，先保证稳定性
    """
    graph = tool_kg_subgraph(name=name, depth=1)

    if not graph.get("ok"):
        return {
            "ok": False,
            "name": name,
            "summary": "",
            "error": graph.get("error", "subgraph_failed"),
        }

    nodes = graph.get("nodes", [])
    center = graph.get("center", name)

    counter = defaultdict(int)
    for node in nodes:
        if node["type"] != "Medicine":
            counter[node["type"]] += 1

    efficacy_n = counter.get("Efficacy", 0)
    meridian_n = counter.get("Meridian", 0)
    taste_n = counter.get("TasteProperty", 0)
    disease_n = counter.get("Disease", 0)
    category_n = counter.get("Category", 0)

    parts = []

    if efficacy_n:
        parts.append(f"{efficacy_n} 个功效节点")
    if meridian_n:
        parts.append(f"{meridian_n} 个归经节点")
    if taste_n:
        parts.append(f"{taste_n} 个性味节点")
    if disease_n:
        parts.append(f"{disease_n} 个适应症节点")
    if category_n:
        parts.append(f"{category_n} 个类别节点")

    if parts:
        summary = (
            f"{center} 在图谱中主要连接了"
            + "、".join(parts)
            + "，呈现出较完整的中药属性关系网络。"
        )
    else:
        summary = f"{center} 在图谱中目前仅包含有限的结构信息。"

    return {
        "ok": True,
        "name": center,
        "summary": summary,
        "stats": dict(counter),
    }