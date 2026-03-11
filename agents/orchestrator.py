from typing import Dict, Any, Optional

from tools.rag_hnsw import VectorStore, compose_evidence
from tools.mcp_client import MCPClient
from tools.qwen_client import QwenClient
from tools.text_utils import resolve_query
from tools.entity_extractor import LLMEntityExtractor
from tools.evidence_cleaner import clean_evidence_for_entity

from agents.herb_card_agent import HerbCardAgent
from agents.flavor_style_agent import FlavorStyleAgent
from agents.persona_agent import PersonaAgent
from agents.image_agent import ImageAgent
from agents.tts_agent import TTSAgent
from agents.formula_agent import FormulaAgent

class Orchestrator:
    """
    多Agent编排器
    """

    FORMULA_SUFFIX = ["汤","散","丸","饮","膏","丹","剂"]

    def __init__(self, *, vec_dir: str, mcp_url: str, min_rag_score: float = 0.28):

        self.vs = VectorStore(vec_dir)

        self.mcp = MCPClient(mcp_url)

        self.llm = QwenClient()

        self.entity_extractor = LLMEntityExtractor(self.llm)

        self.card_agent = HerbCardAgent()

        self.style_agent = FlavorStyleAgent(self.llm)

        self.persona_agent = PersonaAgent(self.llm)

        self.tts_agent = TTSAgent(self.mcp)

        self.image_agent = ImageAgent(self.mcp)

        self.min_rag_score = min_rag_score

        self.topic: Optional[str] = None

        self.formula_agent = FormulaAgent(self.vs, self.llm, min_rag_score=self.min_rag_score)


    def is_formula(self, text: str):

        for s in self.FORMULA_SUFFIX:

            if text.endswith(s):

                return True

        return False


    def run_once(self, user_input: str) -> Dict[str, Any]:

        if self.is_formula(user_input):

            return self.run_formula(user_input)

        return self.run_herb(user_input)

    def run_formula(self, query: str):
        data = self.formula_agent.run(query)
        if data.get("ok"):
            try:
                write_ret = self.mcp.formula_write_json(query, data)
                if write_ret.get("ok"):
                    data["card_json_url"] = write_ret.get("web_path", "")
                    data["card_json_saved"] = True
                else:
                    data["card_json_saved"] = False
                    data["card_json_error"] = write_ret.get("error") or write_ret.get("raw")
            except Exception as e:
                data["card_json_saved"] = False
                data["card_json_error"] = str(e)
        return data


    def run_herb(self, user_input: str) -> Dict[str, Any]:

        q0 = resolve_query(user_input, self.topic)

        pre_hits = self.vs.search(q0, topk=2)

        pre_hint = ""

        if pre_hits:

            pre_hint = (pre_hits[0][1].get("text") or "")[:240]


        entity = self.entity_extractor.extract(

            user_input=q0,

            rag_hint=pre_hint

        )

        query = entity or q0


        rag_hits = self.vs.search(query, topk=6)

        best = rag_hits[0][0] if rag_hits else -1.0

        evidence_raw = compose_evidence(rag_hits)


        if not rag_hits or best < self.min_rag_score:

            return {

                "ok": False,

                "query": query,

                "entity": entity,

                "topic": self.topic,

                "reason": f"资料不足或相关性偏低(best_score={best:.3f})",

            }


        evidence = clean_evidence_for_entity(

            evidence_raw,

            entity or query

        )


        kg = {"found": False, "name": query}

        try:

            kg = self.mcp.kg_get_node(query)

        except Exception as e:

            kg = {"found": False, "name": query, "error": str(e)}


        if kg.get("found"):

            self.topic = kg.get("name") or query

        else:

            if entity:

                self.topic = entity

            elif self.topic is None and len(query) <= 8:

                self.topic = query


        card = self.card_agent.run(

            query,

            kg if kg.get("found") else None,

            evidence

        )


        narration = self.style_agent.run(card)


        persona = self.persona_agent.run(

            card,

            narration

        )


        tts = {"type": "tts", "ok": False}

        try:

            tts = self.tts_agent.run(

                card.get("name", query),

                narration,

                persona

            )

        except Exception as e:

            tts = {"type": "tts", "ok": False, "error": str(e)}


        image = {"type": "image", "ok": False}

        try:

            image = self.image_agent.run(

                card.get("name", query),

                persona

            )

        except Exception as e:

            image = {"type": "image", "ok": False, "error": str(e)}


        return {

            "ok": True,

            "type": "herb",

            "query": query,

            "entity": entity,

            "topic": self.topic,

            "card": card,

            "narration": narration,

            "persona": persona,

            "tts": tts,

            "image": image
        }