# -*- coding: utf-8 -*-
"""
Embedding RAG (real vector search) with SentenceTransformer + HNSWLIB

Input:
  rag_out_clean/chunks_clean.jsonl   (each line is a JSON: {"text": ..., "meta": {...}})

Output:
  rag_out_vec/
    - chunks_vec.jsonl        (copy of chunks with assigned integer vector_id)
    - embeddings.npy          (float32 matrix [N, dim])
    - hnsw_index.bin          (HNSW vector index)
    - index_meta.json         (model name/path, dim, metric, N, etc.)

Run:
  python embedding_rag_hnsw.py --mode build
  python embedding_rag_hnsw.py --mode query --q "八纲辨证包括哪些内容？" --topk 5

说明（离线模型）：
  你已经把模型下载到本地后，直接把 --model 设置为本地目录路径，例如：
    --model "G:\\models\\bge-small-zh-v1.5"
"""

import os
import json
import argparse
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

import hnswlib
from sentence_transformers import SentenceTransformer


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u3000", " ").strip()
    # 可按需做更激进清洗；这里保持轻量，避免误删内容
    return s


def resolve_model_path(model_arg: str) -> str:
    """
    允许 model_arg 既可以是 HuggingFace repo id，也可以是本地目录路径。
    - 若传入的是本地路径但不存在，直接报错，避免又去联网下载导致失败。
    """
    model_arg = model_arg.strip().strip('"').strip("'")
    # 判断是否像路径：含盘符/反斜杠/斜杠，或者实际存在
    looks_like_path = (":" in model_arg) or ("\\" in model_arg) or ("/" in model_arg)
    if looks_like_path:
        # 统一成规范路径（不要求存在时也可，但这里强制存在，避免误写）
        p = os.path.normpath(model_arg)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"你传入的是本地模型路径，但该路径不存在：{p}\n"
                f"请确认模型目录是否正确（目录里应有 modules.json / config.json 等文件）。"
            )
        return p
    # 否则当作 repo id
    return model_arg


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return emb.astype(np.float32)


def build_hnsw_index(
    embeddings: np.ndarray,
    out_index_path: str,
    space: str = "cosine",
    ef_construction: int = 200,
    M: int = 32,
) -> hnswlib.Index:
    assert embeddings.ndim == 2
    n, dim = embeddings.shape

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.add_items(embeddings, np.arange(n))
    index.set_ef(64)  # 查询时的 ef，可调大提升召回但更慢
    index.save_index(out_index_path)
    return index


def load_hnsw_index(index_path: str, dim: int, space: str = "cosine") -> hnswlib.Index:
    index = hnswlib.Index(space=space, dim=dim)
    index.load_index(index_path)
    index.set_ef(64)
    return index


def pretty_print_hits(chunks: List[Dict[str, Any]], labels: np.ndarray, dists: np.ndarray, topk: int) -> None:
    # hnswlib 在 cosine space 下返回的是距离（越小越相似），我们转成“相似度”方便看
    for rank in range(min(topk, labels.shape[0])):
        idx = int(labels[rank])
        dist = float(dists[rank])
        score = 1.0 - dist  # cosine distance = 1 - cosine similarity

        meta = chunks[idx].get("meta", {}) or chunks[idx].get("metadata", {}) or {}
        src = meta.get("source") or meta.get("source_file") or meta.get("file") or meta.get("doc") or "unknown"
        page = meta.get("page", meta.get("page_num", meta.get("pageno", None)))
        page_str = f" p{page}" if page is not None else ""

        text = (chunks[idx].get("text", "") or "").replace("\n", " ").strip()
        if len(text) > 220:
            text = text[:220] + " ..."

        print(f"  [{rank+1}] score={score:.3f}  {os.path.basename(str(src))}{page_str}")
        print(f"       {text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build", "query"], default="build")
    ap.add_argument("--in_chunks", default=r"rag_out_clean/chunks_clean.jsonl")
    ap.add_argument("--out_dir", default=r"rag_out_vec")

    # ✅ 这里改成你的本地模型目录（你已经下载好了模型）
    #    如果你放在别的目录，运行时用 --model "你的路径" 覆盖即可。
    ap.add_argument("--model", default=r"G:\models\bge-small-zh-v1.5")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--q", default="")
    ap.add_argument("--space", choices=["cosine", "l2", "ip"], default="cosine")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    out_chunks_path = os.path.join(args.out_dir, "chunks_vec.jsonl")
    out_emb_path = os.path.join(args.out_dir, "embeddings.npy")
    out_index_path = os.path.join(args.out_dir, "hnsw_index.bin")
    out_meta_path = os.path.join(args.out_dir, "index_meta.json")

    if args.mode == "build":
        if not os.path.exists(args.in_chunks):
            raise FileNotFoundError(f"找不到输入 chunks: {args.in_chunks}")

        chunks = read_jsonl(args.in_chunks)
        if not chunks:
            raise RuntimeError("chunks 为空，检查输入文件是否正确。")

        texts = [normalize_text(c.get("text", "")) for c in chunks]

        model_ref = resolve_model_path(args.model)
        print(f"[1/4] Load embedding model: {model_ref}")
        model = SentenceTransformer(model_ref)

        print(f"[2/4] Encode {len(texts)} chunks ...")
        embeddings = embed_texts(model, texts, batch_size=args.batch_size, normalize=(args.space == "cosine"))

        # 保存 embeddings
        np.save(out_emb_path, embeddings)

        # 给每个 chunk 写入 vector_id（= 行号）
        for i, c in enumerate(chunks):
            c["vector_id"] = i

        write_jsonl(out_chunks_path, chunks)

        print(f"[3/4] Build HNSW index (space={args.space}) ...")
        build_hnsw_index(
            embeddings=embeddings,
            out_index_path=out_index_path,
            space=args.space,
            ef_construction=200,
            M=32,
        )

        meta = {
            # ✅ 新增：推荐字段名（后续 herb_query.py 更好读）
            "embedding_model": model_ref,

            # ✅ 兼容：保留你旧字段名
            "model": model_ref,

            "space": args.space,
            "n": int(embeddings.shape[0]),
            "dim": int(embeddings.shape[1]),
            "chunks_in": args.in_chunks,
            "chunks_out": out_chunks_path,
            "embeddings": out_emb_path,
            "index": out_index_path,
        }
        with open(out_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("[4/4] Done.")
        print(f"  - chunks_vec : {out_chunks_path}")
        print(f"  - embeddings : {out_emb_path}")
        print(f"  - hnsw index : {out_index_path}")
        print(f"  - meta       : {out_meta_path}")

    else:  # query
        if not args.q.strip():
            raise ValueError("query 模式需要 --q '...问题...'")

        if not (os.path.exists(out_chunks_path) and os.path.exists(out_index_path) and os.path.exists(out_meta_path)):
            raise FileNotFoundError(
                "找不到向量索引输出文件。请先运行：python embedding_rag_hnsw.py --mode build"
            )

        with open(out_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        dim = int(meta["dim"])
        space = meta.get("space", args.space)

        # ✅ 优先读取 embedding_model；否则兼容读取 model
        model_name_or_path = meta.get("embedding_model") or meta.get("model") or args.model
        model_ref = resolve_model_path(str(model_name_or_path))

        print(f"[1/3] Load model: {model_ref}")
        model = SentenceTransformer(model_ref)

        print("[2/3] Load chunks + index ...")
        chunks = read_jsonl(out_chunks_path)
        index = load_hnsw_index(out_index_path, dim=dim, space=space)

        print("[3/3] Search ...")
        q_emb = embed_texts(model, [args.q], batch_size=1, normalize=(space == "cosine"))
        labels, dists = index.knn_query(q_emb, k=args.topk)
        labels = labels[0]
        dists = dists[0]

        print(f"\nQ: {args.q}")
        pretty_print_hits(chunks, labels, dists, args.topk)


if __name__ == "__main__":
    main()
