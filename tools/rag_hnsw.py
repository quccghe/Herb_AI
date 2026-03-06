import os
import json
import re
from typing import List, Dict, Any, Tuple

import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

class VectorStore:
    def __init__(self, vec_dir: str):
        meta_path = os.path.join(vec_dir, "index_meta.json")
        idx_path = os.path.join(vec_dir, "hnsw_index.bin")
        chunks_path = os.path.join(vec_dir, "chunks_vec.jsonl")

        for p in [meta_path, idx_path, chunks_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing vector store file: {p}")

        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        self.dim = int(meta["dim"])
        self.space = (meta.get("space") or "cosine").lower()
        self.model_path = meta.get("embedding_model") or meta.get("model")
        if not self.model_path:
            raise RuntimeError("index_meta.json missing embedding_model/model")

        self.embedder = SentenceTransformer(self.model_path)

        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(idx_path)
        self.index.set_ef(64)

        chunks = read_jsonl(chunks_path)
        self.id2chunk = {int(c.get("vector_id", i)): c for i, c in enumerate(chunks)}

    def search(self, query: str, topk: int = 6) -> List[Tuple[float, Dict[str, Any]]]:
        q = normalize(query)
        qv = self.embedder.encode([q], normalize_embeddings=(self.space == "cosine"))
        qv = np.asarray(qv, dtype=np.float32)
        labels, dists = self.index.knn_query(qv, k=topk)
        labels = labels[0].tolist()
        dists = dists[0].tolist()

        out: List[Tuple[float, Dict[str, Any]]] = []
        for vid, dist in zip(labels, dists):
            ch = self.id2chunk.get(int(vid))
            if not ch:
                continue
            score = (1.0 - float(dist)) if self.space == "cosine" else (-float(dist))
            out.append((score, ch))
        out.sort(key=lambda x: x[0], reverse=True)
        return out

def compose_evidence(rag_hits: List[Tuple[float, Dict[str, Any]]], max_items: int = 4, max_chars_each: int = 340) -> str:
    parts: List[str] = []
    for i, (score, ch) in enumerate(rag_hits[:max_items], 1):
        txt = normalize(ch.get("text", ""))[:max_chars_each]
        meta = ch.get("meta") or ch.get("metadata") or {}
        src = meta.get("source") or meta.get("file") or meta.get("doc") or "unknown"
        page = meta.get("page") or meta.get("pageno")
        loc = f"{src}" + (f" p{page}" if page is not None and str(page) != "" else "")
        parts.append(f"[证据{i}] {loc} score={score:.3f}\n{txt}")
    return "\n\n".join(parts).strip()