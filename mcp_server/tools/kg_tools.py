# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase

_driver = None


def init_driver(uri: str, user: str, password: str) -> None:
    global _driver
    _driver = GraphDatabase.driver(uri, auth=(user, password))


def tool_health() -> Dict[str, Any]:
    try:
        with _driver.session() as s:
            r = s.run("RETURN 1 AS ok").single()
            return {"ok": bool(r and r["ok"] == 1)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# mcp_server/tools/kg_tools.py
from typing import Dict, Any
import os
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

_driver = None

def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver

def tool_kg_get_node(name: str) -> Dict[str, Any]:
    """
    KG 可选：Neo4j 不在线时也不能抛 500
    """
    name = (name or "").strip()
    if not name:
        return {"found": False, "name": "", "error": "empty_name"}

    try:
        driver = get_driver()
        cypher = """
        MATCH (n)
        WHERE n.name = $name OR n.alias = $name
        RETURN n LIMIT 1
        """
        with driver.session() as s:
            rec = s.run(cypher, name=name).single()

        if not rec:
            return {"found": False, "name": name}

        node = rec["n"]
        props = dict(node)
        # 尽量把 name 标准化
        canon = props.get("name") or name
        return {"found": True, "name": canon, "props": props}

    except Exception as e:
        # 关键：不要抛异常
        return {"found": False, "name": name, "error": str(e)}


def tool_kg_neighbors(
    name: str,
    limit: int = 30,
    rel_types: Optional[List[str]] = None,
    neighbor_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """返回一阶邻居（用于关系网络）"""
    if _driver is None:
        return {"found": False, "name": name, "error": "Neo4j driver not initialized"}

    name = (name or "").strip()
    limit = max(1, min(int(limit), 200))
    params: Dict[str, Any] = {"name": name, "limit": limit}

    rel_filter = ""
    if rel_types:
        rel_filter = "AND type(r) IN $rel_types"
        params["rel_types"] = rel_types

    label_filter = ""
    if neighbor_labels:
        label_filter = "AND any(l in labels(n) WHERE l IN $neighbor_labels)"
        params["neighbor_labels"] = neighbor_labels

    cypher = f"""
    MATCH (m:Medicine {{name:$name}})-[r]->(n)
    WHERE 1=1 {rel_filter} {label_filter}
    RETURN type(r) AS rel, labels(n) AS labels, n AS node
    LIMIT $limit
    """

    with _driver.session() as s:
        rs = s.run(cypher, **params)
        neighbors = []
        for rec in rs:
            neighbors.append({
                "rel": rec["rel"],
                "labels": rec["labels"],
                "node": dict(rec["node"]),
            })
        return {"found": True, "name": name, "neighbors": neighbors}


def tool_kg_paths(a: str, b: str, k: int = 3, max_hops: int = 3) -> Dict[str, Any]:
    """返回 a 到 b 的若干条最短路径（用于配伍解释器）"""
    if _driver is None:
        return {"found": False, "a": a, "b": b, "error": "Neo4j driver not initialized"}

    a = (a or "").strip()
    b = (b or "").strip()
    k = max(1, min(int(k), 10))
    max_hops = max(1, min(int(max_hops), 6))

    cypher = f"""
    MATCH (a:Medicine {{name:$a}}), (b:Medicine {{name:$b}})
    MATCH p = shortestPath((a)-[*..{max_hops}]-(b))
    RETURN p
    LIMIT $k
    """

    with _driver.session() as s:
        rs = s.run(cypher, a=a, b=b, k=k)
        paths = []
        for rec in rs:
            p = rec["p"]
            nodes = [dict(n) for n in p.nodes]
            rels = [{
                "type": r.type,
                "start": r.start_node.get("name"),
                "end": r.end_node.get("name"),
            } for r in p.relationships]
            paths.append({"nodes": nodes, "rels": rels})
        return {"found": True, "a": a, "b": b, "paths": paths}