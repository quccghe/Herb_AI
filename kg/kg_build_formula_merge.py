# kg_build_formula_merge.py
# 将方剂知识图谱合并导入现有中药知识图谱

import json
from neo4j import GraphDatabase

# ===== Neo4j配置 =====
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# ===== 文件路径 =====
NODES_FILE = "nodes_fangji.txt"
REL_FILE = "relations_fangji.json"


# ===== 标签映射 =====
LABEL_MAP = {
    "方剂": "Formula",
    "方名": "Formula",
    "处方": "Prescription",
    "中药名": "Medicine",
    "功能主治": "Indication",
    "来源": "Source",
    "别名": "Alias",
    "剂量": "Dose"
}

# ===== 关系映射 =====
REL_MAP = {
    "composition": "CONTAINS_HERB",
    "functions": "HAS_INDICATION",
    "include": "HAS_NAME",
    "prescription type": "HAS_PRESCRIPTION",
    "dose": "HAS_DOSE",
    "from": "FROM_SOURCE",
    "another name": "HAS_ALIAS"
}


def clean_name(name: str):
    """简单清洗"""
    if not name:
        return None

    name = name.strip()

    # 去掉明显错误字符
    if "?" in name:
        return None

    # 单字无意义过滤
    if len(name) == 1:
        return None

    return name


def load_nodes():
    nodes = []

    with open(NODES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")

            if len(parts) != 2:
                continue

            label, name = parts

            name = clean_name(name)

            if not name:
                continue

            if label not in LABEL_MAP:
                continue

            nodes.append((LABEL_MAP[label], name))

    return nodes


def load_relations():
    with open(REL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rels = []

    for r in data:

        n1 = clean_name(r["node_1"])
        n2 = clean_name(r["node_2"])
        rel = r["relation"]

        if not n1 or not n2:
            continue

        if rel not in REL_MAP:
            continue

        rels.append((n1, REL_MAP[rel], n2))

    return rels


def create_node(tx, label, name):

    query = f"""
    MERGE (n:{label} {{name:$name}})
    """

    tx.run(query, name=name)


def create_relation(tx, n1, rel, n2):

    query = f"""
    MATCH (a {{name:$n1}})
    MATCH (b {{name:$n2}})
    MERGE (a)-[:{rel}]->(b)
    """

    tx.run(query, n1=n1, n2=n2)


def import_nodes(nodes):

    with driver.session() as session:

        for label, name in nodes:
            session.execute_write(create_node, label, name)


def import_relations(rels):

    with driver.session() as session:

        for n1, rel, n2 in rels:
            session.execute_write(create_relation, n1, rel, n2)


def main():

    print("加载节点...")

    nodes = load_nodes()
    print("节点数量:", len(nodes))

    print("加载关系...")

    rels = load_relations()
    print("关系数量:", len(rels))

    print("导入节点...")

    import_nodes(nodes)

    print("导入关系...")

    import_relations(rels)

    print("完成")


if __name__ == "__main__":
    main()