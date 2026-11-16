"""Chạy agent thật trên agent_cases.json, lưu trace từng câu.

    python src/run_agent.py --split dev --budget 1800 --output starter/results/agent-dev-selected
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "src"))
from agentic import run_agent            # noqa: E402
from controller import OpenAIController  # noqa: E402
from retrievers import DenseRetriever    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/agent_cases.json")
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--budget", type=int, default=1800)
    ap.add_argument("--max-searches", type=int, default=3)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    cases = [c for c in json.load(open(args.cases, encoding="utf-8")) if c["split"] == args.split]
    retriever, controller = DenseRetriever(), OpenAIController()
    config = {"model": controller.model, "retriever": "dense day6_A e5-small", "top_k": args.top_k,
              "budget_chars": args.budget, "max_searches": args.max_searches, "split": args.split}

    traces = []
    for case in cases:
        controller.reset_usage()
        start = time.perf_counter()
        try:
            result = run_agent(case["question"], controller, retriever, filters=case.get("filter"),
                               max_searches=args.max_searches, top_k=args.top_k, budget=args.budget)
            error = None
        except Exception as e:  # ghi lỗi thật, không thay bằng kết quả giả
            result = {"status": "error", "answer": None, "citations": [], "events": []}
            error = f"{type(e).__name__}: {e}"
        elapsed = time.perf_counter() - start
        n_search = sum(e["event"] == "search" for e in result["events"])
        traces.append({"query_id": case["id"], "split": case["split"], "question": case["question"],
                       "answerable": case["answerable"], **result, "error": error,
                       "n_searches": n_search, "elapsed_s": round(elapsed, 2), "usage": dict(controller.usage)})
        print(f"{case['id']:<5} {result['status']:<9} searches={n_search} {elapsed:5.1f}s  {(result['answer'] or error or '')[:80]}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tok = sum(t["usage"]["prompt_tokens"] + t["usage"]["completion_tokens"] for t in traces)
    print(f"\n{len(traces)} câu | answered={sum(t['status']=='answered' for t in traces)} "
          f"abstain={sum(t['status']=='abstain' for t in traces)} error={sum(t['status']=='error' for t in traces)} | tokens={tok}")


if __name__ == "__main__":
    main()