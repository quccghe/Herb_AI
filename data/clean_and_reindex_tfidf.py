# clean_and_reindex_tfidf.py
# -*- coding: utf-8 -*-

"""
从已有 chunks.jsonl 二次清洗（过滤练习题/目录等噪声）并重建 TF-IDF 检索索引。

输入（你已生成）:
- chunks.jsonl  (JSONL，每行: {"id","text","metadata"})

输出:
- rag_out_clean/chunks_clean.jsonl
- rag_out_clean/tfidf_clean.joblib
- rag_out_clean/nn_index_clean.joblib
- rag_out_clean/clean_stats.json

并包含一个可直接调用的 query() 演示。
"""

from __future__ import annotations
import os
import re
import json
from typing import List, Dict, Any, Tuple

from tqdm import tqdm
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ======= 你需要改的路径（按你自己的目录）=======
# 你上传的 chunks.jsonl（原始知识库）
INPUT_CHUNKS_JSONL = "rag_out/chunks.jsonl"  # <-- 如果你的文件不在这里，改成你的实际路径

# 输出目录
OUT_DIR = "rag_out_clean"
CHUNKS_CLEAN_PATH = os.path.join(OUT_DIR, "chunks_clean.jsonl")
VECTORIZER_PATH = os.path.join(OUT_DIR, "tfidf_clean.joblib")
NN_INDEX_PATH = os.path.join(OUT_DIR, "nn_index_clean.joblib")
STATS_PATH = os.path.join(OUT_DIR, "clean_stats.json")

# ======= 参数 =======
# 如果你库里很多“题目/练习”，建议先用“过滤”模式（更猛更有效）
MODE = "filter"  # "filter" or "downweight"
TOP_K = 5

# TF-IDF 设置（char-gram: 对中文无需分词）
NGRAM_RANGE = (1, 2)
MAX_FEATURES = 200000
MIN_DF = 2

# 过滤阈值：噪声分太高就删
NOISE_SCORE_FILTER_THRESHOLD = 3.0   # MODE="filter" 时使用
NOISE_SCORE_DOWNWEIGHT_THRESHOLD = 2.5  # MODE="downweight" 时使用
DOWNWEIGHT_REPEAT = 3  # 降权：把“噪声 chunk”复制几次降低其相对权重？（这里不采用复制法，而采用标记后过滤展示）


# ======= 噪声识别规则 =======
# 这些规则专门针对“教材 PDF 的目录页/练习题页/思考题页/复习题页”
NOISE_KEYWORDS = [
    "思考题", "练习题", "复习题", "选择题", "填空题", "判断题", "问答题", "简答题",
    "名词解释", "论述题", "病例分析",
    "参考答案", "答案", "解析",
    "目录", "目 录", "索引",
    "第1题", "第2题", "第3题",
    "【思考】", "【练习】",
    "本章小结", "本章习题",
    "自测题",
    "查阅", "比较", "试述", "试举", "试分析",
]

QUESTION_PHRASES = [
    "如何理解", "为什么", "何以", "如何", "怎样", "何谓", "什么是",
    "试述", "试举", "试比较", "试分析", "简述", "说明",
]

def noise_score(text: str) -> Tuple[float, List[str]]:
    """
    返回 (score, matched_reasons)
    score 越高越像噪声（练习题/目录等）
    """
    t = text.strip()
    t_nospace = re.sub(r"\s+", "", t)

    score = 0.0
    reasons: List[str] = []

    # 1) 多问号：题目页特征
    q_count = t.count("？") + t.count("?")
    if q_count >= 2:
        score += 2.0
        reasons.append(f"多问号({q_count})")

    # 2) 关键词命中
    for kw in NOISE_KEYWORDS:
        if kw in t or kw in t_nospace:
            score += 1.0
            reasons.append(f"kw:{kw}")

    # 3) 典型问句开头/句式
    for ph in QUESTION_PHRASES:
        if ph in t:
            score += 0.6
            reasons.append(f"phrase:{ph}")

    # 4) 连续编号题（1. 2. 3. / 1、2、3、）
    # 命中多个编号行，噪声可能性很高
    num_lines = re.findall(r"(^|\n)\s*\d+\s*[\.、．]\s*", t)
    if len(num_lines) >= 3:
        score += 2.2
        reasons.append(f"多编号行({len(num_lines)})")
    elif len(num_lines) == 2:
        score += 1.2
        reasons.append("编号行(2)")

    # 5) “题目页通常短句多、信息密度低”
    # 如果很短又含问句结构，进一步加分
    if len(t) < 450 and (q_count >= 1 or any(ph in t for ph in QUESTION_PHRASES)):
        score += 1.0
        reasons.append("短+问句")

    return score, reasons


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_index(texts: List[str]) -> Tuple[TfidfVectorizer, NearestNeighbors]:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        min_df=MIN_DF
    )
    X = vectorizer.fit_transform(texts)

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(X)
    return vectorizer, nn


def query(
    question: str,
    chunks_path: str = CHUNKS_CLEAN_PATH,
    vectorizer_path: str = VECTORIZER_PATH,
    nn_index_path: str = NN_INDEX_PATH,
    top_k: int = TOP_K,
    skip_noise: bool = True,
) -> List[Dict[str, Any]]:
    vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)
    nn: NearestNeighbors = joblib.load(nn_index_path)
    chunks = load_jsonl(chunks_path)

    # 简单 query expansion：对术语类问题有帮助
    q = expand_query(question)
    qv = vectorizer.transform([q])

    distances, indices = nn.kneighbors(qv, n_neighbors=min(top_k * 8, len(chunks)))
    distances = distances[0].tolist()
    indices = indices[0].tolist()

    results = []
    for dist, idx in zip(distances, indices):
        item = chunks[idx]
        if skip_noise and item.get("metadata", {}).get("is_noise", False):
            continue
        score = 1.0 - float(dist)
        results.append({
            "score": score,
            "id": item.get("id"),
            "text": item.get("text"),
            "metadata": item.get("metadata", {}),
        })
        if len(results) >= top_k:
            break
    return results


def expand_query(q: str) -> str:
    """
    用少量规则增强查询（TF-IDF 下非常实用）
    你可以按自己需求继续扩展
    """
    q2 = q
    if "四气五味" in q:
        q2 += " 四气 五味 性味 寒热温凉 辛甘酸苦咸"
    if "辨证论治" in q:
        q2 += " 概念 定义 基本原则"
    if "桂枝汤" in q:
        q2 += " 桂枝汤 组成 功用 主治 方解"
    if "八纲辨证" in q or ("八纲" in q and "辨证" in q):
        q2 += " 表里 寒热 虚实 阴阳"
    return q2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[1/3] 读取原始 chunks: {INPUT_CHUNKS_JSONL}")
    raw = load_jsonl(INPUT_CHUNKS_JSONL)
    print(f"  - raw chunks: {len(raw)}")

    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    print(f"[2/3] 二次清洗（MODE={MODE}） ...")
    for item in tqdm(raw):
        text = item.get("text", "") or ""
        score, reasons = noise_score(text)

        meta = item.get("metadata", {}) or {}
        meta["noise_score"] = score
        meta["noise_reasons"] = reasons
        meta["is_noise"] = score >= NOISE_SCORE_DOWNWEIGHT_THRESHOLD  # 记录标记（无论 filter 还是 downweight）

        item["metadata"] = meta

        if MODE == "filter":
            if score >= NOISE_SCORE_FILTER_THRESHOLD:
                removed.append(item)
            else:
                kept.append(item)
        else:
            # downweight 模式：不删，只标记
            kept.append(item)

    stats = {
        "mode": MODE,
        "raw_chunks": len(raw),
        "kept_chunks": len(kept),
        "removed_chunks": len(removed),
        "filter_threshold": NOISE_SCORE_FILTER_THRESHOLD,
        "downweight_threshold": NOISE_SCORE_DOWNWEIGHT_THRESHOLD,
        "ngram_range": NGRAM_RANGE,
        "min_df": MIN_DF,
    }

    save_jsonl(CHUNKS_CLEAN_PATH, kept)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"  - kept: {len(kept)}  removed: {len(removed)}")
    print(f"  - saved clean chunks: {CHUNKS_CLEAN_PATH}")
    print(f"  - stats: {STATS_PATH}")

    print("[3/3] 重建 TF-IDF + 最近邻索引 ...")
    texts = [x["text"] for x in kept]
    vectorizer, nn = build_index(texts)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(nn, NN_INDEX_PATH)

    print(f"  - saved vectorizer: {VECTORIZER_PATH}")
    print(f"  - saved nn index : {NN_INDEX_PATH}")

    # ===== Demo =====
    demo_qs = [
        "八纲辨证包括哪些内容？",
        "中药的四气五味是什么意思？",
        "什么是辨证论治？",
        "桂枝汤的功用主治是什么？",
    ]
    print("\n=== Demo Query ===")
    for q in demo_qs:
        print("\nQ:", q)
        hits = query(q, top_k=5, skip_noise=True)
        for i, h in enumerate(hits, 1):
            meta = h["metadata"]
            loc = f'{meta.get("source")} p{meta.get("page")}' if meta.get("page") else meta.get("source")
            print(f"  [{i}] score={h['score']:.3f}  {loc}  noise_score={meta.get('noise_score')}")
            preview = (h["text"] or "").replace("\n", " ")
            print("      ", preview[:160], "...")


if __name__ == "__main__":
    main()
