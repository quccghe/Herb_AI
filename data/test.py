# build_rag.py
# -*- coding: utf-8 -*-
"""
从 TXT/PDF 书籍构建一个“可用的”初步 RAG：
- 抽取文本（PDF 用 PyMuPDF）
- 清洗与去噪
- 结构化切块（chunking + overlap）
- 保存 chunks.jsonl
- 使用 TF-IDF 建立向量检索索引（NearestNeighbors）
- 提供一个简单的 query 检索函数
"""

from __future__ import annotations
import os
import re
import json
import math
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple, Optional

from tqdm import tqdm

# PDF 解析
import fitz  # PyMuPDF

# 向量检索（初版用 TF-IDF，足够先跑通）
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib


# ============ 你可以在这里配置输入文件 ============
INPUT_FILES = [
    "G:/计算机设计大赛/data/中医基础理论.txt",
    "G:/计算机设计大赛/data/20.中医诊断学.pdf",
    "G:/计算机设计大赛/data/1.方剂学.pdf",
    "G:/计算机设计大赛/data/2.中药学.pdf",
    "G:/计算机设计大赛/data/34.中医食疗学（十三五）.pdf",
    "G:/计算机设计大赛/data/2020版中国药典（一部）.pdf",
    "G:/计算机设计大赛/data/2020版中国药典（二部）.pdf",
    "G:/计算机设计大赛/data/2020版中国药典（三部）.pdf",
    "G:/计算机设计大赛/data/2020版中国药典（四部）.pdf",
]

OUT_DIR = "rag_out"
CHUNKS_PATH = os.path.join(OUT_DIR, "chunks.jsonl")
VECTORIZER_PATH = os.path.join(OUT_DIR, "tfidf.joblib")
NN_INDEX_PATH = os.path.join(OUT_DIR, "nn_index.joblib")
STATS_PATH = os.path.join(OUT_DIR, "doc_stats.json")

# 切块参数：中文书籍一般 400~900 字一块比较合适
CHUNK_SIZE = 800          # 每块目标字符数
CHUNK_OVERLAP = 150       # 重叠字符数
MIN_CHUNK_LEN = 200       # 太短的块丢弃
MAX_LINE_LEN = 2000       # 超长行裁剪（PDF 解析偶发）

# TF-IDF 参数
MAX_FEATURES = 200000     # 特征上限
NGRAM_RANGE = (1, 2)      # 1~2gram，中文简单有效
TOP_K = 5                 # 默认检索返回条数


# ============ 工具函数 ============
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def normalize_text(text: str) -> str:
    """
    面向教材/书籍的轻量清洗：
    - 去掉多余空白、重复换行
    - 去掉明显的页眉页脚噪声（非常保守）
    - 裁剪异常超长行
    """
    if not text:
        return ""

    # 统一换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 裁剪异常超长行（PDF 提取时偶发）
    lines = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln:
            lines.append("")
            continue
        if len(ln) > MAX_LINE_LEN:
            ln = ln[:MAX_LINE_LEN]
        lines.append(ln)
    text = "\n".join(lines)

    # 去掉明显的“免费下载/网址”之类广告段落（保守一点）
    # 你给的《中医基础理论》里前言有 A+医学百科链接等，属于噪声，可以去掉
    text = re.sub(r"免费下载.*?http[^\s]+", "", text)
    text = re.sub(r"网址[:：]\s*http[^\s]+", "", text)

    # 合并多余空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def split_to_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    以“段落”为主，段落过长再按长度切分，带 overlap。
    """
    text = normalize_text(text)
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []

    def push_chunk(buf: str):
        buf = buf.strip()
        if len(buf) >= MIN_CHUNK_LEN:
            chunks.append(buf)

    buf = ""
    for p in paras:
        # 如果段落本身太长，先拆
        if len(p) > chunk_size * 1.5:
            # 先把已有 buf 推入
            if buf:
                push_chunk(buf)
                buf = ""

            start = 0
            while start < len(p):
                end = min(len(p), start + chunk_size)
                piece = p[start:end]
                push_chunk(piece)
                if end == len(p):
                    break
                start = max(0, end - overlap)
            continue

        # 尝试加入 buf
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= chunk_size:
            buf = buf + "\n" + p
        else:
            push_chunk(buf)
            # overlap：取 buf 尾部 overlap 字符作为新 buf 的开头
            tail = buf[-overlap:] if overlap > 0 and len(buf) > overlap else ""
            buf = (tail + "\n" + p).strip()

    if buf:
        push_chunk(buf)

    return chunks

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf_by_page(path: str) -> List[Tuple[int, str]]:
    """
    返回 [(page_no_from_1, page_text), ...]
    """
    doc = fitz.open(path)
    out = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = page.get_text("text")  # 简洁模式
        txt = normalize_text(txt)
        if txt:
            out.append((i + 1, txt))
    doc.close()
    return out


# ============ 构建 chunks ============
@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]

def build_chunks(input_files: List[str]) -> Tuple[List[Chunk], Dict[str, Any]]:
    all_chunks: List[Chunk] = []
    stats: Dict[str, Any] = {"docs": []}

    for fp in input_files:
        if not os.path.exists(fp):
            print(f"[WARN] 文件不存在，跳过: {fp}")
            continue

        base = os.path.basename(fp)
        ext = os.path.splitext(fp)[1].lower()

        doc_info = {
            "source_file": fp,
            "name": base,
            "type": ext.replace(".", ""),
            "pages": None,
            "chunks": 0,
        }

        if ext == ".txt":
            raw = read_txt(fp)
            chunks = split_to_chunks(raw, CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, ch in enumerate(chunks):
                cid = sha1(f"{base}::txt::{idx}::{ch[:50]}")
                all_chunks.append(Chunk(
                    id=cid,
                    text=ch,
                    metadata={
                        "source": base,
                        "source_path": fp,
                        "type": "txt",
                        "page": None,
                        "chunk_index": idx,
                    }
                ))
            doc_info["chunks"] = len(chunks)

        elif ext == ".pdf":
            pages = read_pdf_by_page(fp)
            doc_info["pages"] = len(pages)

            # 对每页切块，metadata 带页码
            ccount = 0
            for (page_no, page_text) in pages:
                chunks = split_to_chunks(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
                for idx, ch in enumerate(chunks):
                    cid = sha1(f"{base}::p{page_no}::{idx}::{ch[:50]}")
                    all_chunks.append(Chunk(
                        id=cid,
                        text=ch,
                        metadata={
                            "source": base,
                            "source_path": fp,
                            "type": "pdf",
                            "page": page_no,
                            "chunk_index": idx,
                        }
                    ))
                ccount += len(chunks)
            doc_info["chunks"] = ccount

        else:
            print(f"[WARN] 不支持的文件类型，跳过: {fp}")
            continue

        stats["docs"].append(doc_info)

    stats["total_chunks"] = len(all_chunks)
    return all_chunks, stats

def save_chunks_jsonl(chunks: List[Chunk], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            obj = {"id": c.id, "text": c.text, "metadata": c.metadata}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def build_tfidf_index(chunks: List[Chunk]) -> Tuple[TfidfVectorizer, NearestNeighbors]:
    corpus = [c.text for c in chunks]

    # 说明：中文最严格的做法是分词（jieba/THULAC），
    # 但为了“先跑通”，这里用 char-gram 近似（1-2gram），效果通常也还不错
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        min_df=2
    )
    X = vectorizer.fit_transform(corpus)

    # 余弦相似度：用 sklearn 的 NearestNeighbors(metric="cosine")，距离越小越相似
    nn = NearestNeighbors(n_neighbors=min(TOP_K, len(chunks)), metric="cosine", algorithm="brute")
    nn.fit(X)
    return vectorizer, nn

def query_rag(
    question: str,
    chunks_path: str = CHUNKS_PATH,
    vectorizer_path: str = VECTORIZER_PATH,
    nn_index_path: str = NN_INDEX_PATH,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    # load
    vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)
    nn: NearestNeighbors = joblib.load(nn_index_path)

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    qv = vectorizer.transform([question])
    distances, indices = nn.kneighbors(qv, n_neighbors=min(top_k, len(chunks)))
    distances = distances[0].tolist()
    indices = indices[0].tolist()

    results = []
    for dist, idx in zip(distances, indices):
        item = chunks[idx]
        # cosine distance -> similarity
        sim = 1.0 - float(dist)
        results.append({
            "score": sim,
            "id": item["id"],
            "text": item["text"],
            "metadata": item["metadata"],
        })
    return results


def main():
    ensure_dir(OUT_DIR)

    print("[1/3] 抽取文本并切块 ...")
    chunks, stats = build_chunks(INPUT_FILES)
    if not chunks:
        raise RuntimeError("没有生成任何 chunks，请检查文件路径/文件内容。")

    save_chunks_jsonl(chunks, CHUNKS_PATH)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"  - total chunks: {len(chunks)}")
    print(f"  - saved: {CHUNKS_PATH}")
    print(f"  - stats: {STATS_PATH}")

    print("[2/3] 建 TF-IDF 向量索引 ...")
    vectorizer, nn = build_tfidf_index(chunks)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(nn, NN_INDEX_PATH)

    print(f"  - saved: {VECTORIZER_PATH}")
    print(f"  - saved: {NN_INDEX_PATH}")

    print("[3/3] 简单检索演示 ...")
    demo_qs = [
        "八纲辨证包括哪些内容？",
        "中药的四气五味是什么意思？",
        "什么是辨证论治？",
        "桂枝汤的功用主治是什么？",
    ]
    for q in demo_qs:
        print("\nQ:", q)
        res = query_rag(q, top_k=3)
        for i, r in enumerate(res, 1):
            meta = r["metadata"]
            loc = f'{meta["source"]} p{meta["page"]}' if meta["page"] else meta["source"]
            print(f"  [{i}] score={r['score']:.3f}  {loc}")
            preview = r["text"].replace("\n", " ")
            print("      ", preview[:160], "...")


if __name__ == "__main__":
    main()
