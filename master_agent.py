# -*- coding: utf-8 -*-
import argparse
import json
import os

from agents.orchestrator import Orchestrator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec_dir", default="data/rag_out_vec", help="HNSW向量库目录")
    ap.add_argument("--mcp", default=os.environ.get("MCP_BASE_URL", "http://127.0.0.1:8001"), help="MCP Server base url")
    ap.add_argument("--min_rag_score", type=float, default=0.28)
    args = ap.parse_args()

    orch = Orchestrator(vec_dir=args.vec_dir, mcp_url=args.mcp, min_rag_score=args.min_rag_score)

    print("✅ HerbAI 多Agent 启动（exit退出）")
    while True:
        q = input("\n用户> ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break

        result = orch.run_once(q)
        print("\n===== RESULT(JSON) =====\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()