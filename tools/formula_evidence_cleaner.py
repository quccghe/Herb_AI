# -*- coding: utf-8 -*-
import re
from typing import List, Dict, Optional

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


LOCAL_RERANKER_PATH = r"G:\hf_models\bge-reranker-base"


class FormulaEvidenceCleaner:
    """
    方剂证据清洗器
    流程：
    1. 规则初筛
    2. 本地 reranker 精排（若可用）
    3. 返回最适合目标方剂本方的证据块
    """

    KEY_BONUS = ["【组成】", "【主治】", "【功用】", "【功效】", "【用法】", "【煎服法】", "【方解】"]
    POLLUTION_HINTS = ["加减", "加味", "附方", "类方", "又名", "衍方"]

    def __init__(
        self,
        reranker_model_path: str = LOCAL_RERANKER_PATH,
        enable_rerank: bool = True,
    ):
        self.enable_rerank = enable_rerank and (CrossEncoder is not None)
        self.reranker_model_path = reranker_model_path
        self.reranker: Optional[CrossEncoder] = None

        if self.enable_rerank:
            try:
                self.reranker = CrossEncoder(reranker_model_path)
                print(f"[FormulaEvidenceCleaner] reranker loaded: {reranker_model_path}")
            except Exception as e:
                print(f"[FormulaEvidenceCleaner] reranker load failed: {e}")
                self.reranker = None
                self.enable_rerank = False

    def split_evidence_blocks(self, evidence: str) -> List[str]:
        if not evidence:
            return []
        blocks = re.split(r"(?=\[证据\d+\])", evidence)
        return [b.strip() for b in blocks if b.strip()]

    def score_block_rule(self, query: str, block: str) -> float:
        score = 0.0

        if query in block:
            score += 5.0

        for k in self.KEY_BONUS:
            if k in block:
                score += 1.5

        # 惩罚“桂枝加XXX汤”这类串方
        if query.endswith("汤"):
            prefix = query[:-1]
            polluted_names = re.findall(rf"{re.escape(prefix)}加[\u4e00-\u9fffA-Za-z0-9]+汤", block)
            if polluted_names:
                score -= 4.0

        # 惩罚明显串行条目 / 多方并列
        formula_name_like = re.findall(r"方名[:：]|方剂序号[:：]|===========|全文", block)
        if len(formula_name_like) >= 2:
            score -= 2.0

        for hint in self.POLLUTION_HINTS:
            if hint in block:
                score -= 0.5

        return score

    def _build_rerank_query(self, query: str) -> str:
        return f"请找出只属于“{query}本方”的组成、功效、主治、用法，不包含加减方、类方、相关方。"

    def rerank_blocks(self, query: str, blocks: List[str]) -> List[str]:
        if not self.enable_rerank or not self.reranker or not blocks:
            return blocks

        rank_query = self._build_rerank_query(query)
        pairs = [[rank_query, b] for b in blocks]

        try:
            scores = self.reranker.predict(pairs)
            scored = list(zip(scores, blocks))
            scored.sort(key=lambda x: float(x[0]), reverse=True)
            return [b for _, b in scored]
        except Exception as e:
            print(f"[FormulaEvidenceCleaner] rerank failed, fallback to rule order: {e}")
            return blocks

    def clean(self, query: str, evidence: str, topk: int = 4) -> str:
        blocks = self.split_evidence_blocks(evidence)
        if not blocks:
            return evidence

        # 规则初筛
        scored = []
        for b in blocks:
            s = self.score_block_rule(query, b)
            scored.append((s, b))
        scored.sort(key=lambda x: x[0], reverse=True)

        preselected = [b for s, b in scored if s >= 1.5]
        if not preselected:
            preselected = [b for _, b in scored[:4]]

        # rerank 精排
        reranked = self.rerank_blocks(query, preselected)

        final_blocks = reranked[:topk]
        return "\n\n".join(final_blocks)

    def extract_sections(self, evidence: str) -> Dict[str, str]:
        def pick(keys):
            for k in keys:
                pattern = rf"(?:【{re.escape(k)}】|{re.escape(k)}[：:])\s*(.*?)(?=(?:【[\u4e00-\u9fff]+】)|(?:\n\[证据)|$)"
                m = re.search(pattern, evidence, flags=re.S)
                if m:
                    return m.group(1).strip()
            return ""

        return {
            "source": self._extract_source(evidence),
            "composition": pick(["组成"]),
            "dosage_usage": pick(["用法", "用法与用量", "服法", "煎服法"]),
            "efficacy": pick(["功用", "功效"]),
            "indications": pick(["主治", "应用"]),
            "analysis": pick(["方解"]),
            "cautions": pick(["注意", "注意事项", "禁忌"]),
        }

    def _extract_source(self, evidence: str) -> str:
        m = re.search(r"《([^》]{1,40})》", evidence or "")
        return f"《{m.group(1)}》" if m else ""