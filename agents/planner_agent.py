class PlannerAgent:

    FORMULA_SUFFIX = [
        "汤","散","丸","饮","膏","丹","剂"
    ]

    def plan(self, query: str):

        q = query.strip()

        for s in self.FORMULA_SUFFIX:
            if q.endswith(s):

                return {
                    "intent": "formula"
                }

        return {
            "intent": "herb"
        }