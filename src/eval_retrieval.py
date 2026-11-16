"""So sánh các retriever trên cùng bộ câu hỏi; dùng chung hit/pack_context với baseline.

Chạy:
    python src/eval_retrieval.py --split dev --output starter/results/dev-retrieval
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "src"))
from baseline import hit, pack_context  # noqa: E402
from retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever  # noqa: E402

def evaluate(name, retriever, questions, *, top_k=3, budget=800):
    traces = []
    for q in questions:
        start = time.perf_counter()
        results = retriever.search(q["question"], filters=q.get("filter"), top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        context, ids = pack_context(results, budget)
        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]
        top3 = marked[:3]  # luôn so sánh ở cutoff 3, kể cả khi top_k khác 3
        traces.append({
            "query_id": q["id"],
            "question": q["question"],
            "filter": q.get("filter"),
            "strategy": name,
            "top_k": top_k,
            "budget_chars": budget,
            "results": [{k: c[k] for k in ("id", "page", "score", "hit")} for c in marked],
            "context": context,
            "context_ids": ids,
            "context_chars": len(context),
            "elapsed_ms": round(elapsed_ms, 2),
            "hit1": bool(marked and marked[0]["hit"]),
            "hit3": any(c["hit"] for c in top3),
            "context_hit": any(c["hit"] and c["id"] in ids for c in marked),
        })
    summary = {
        "strategy": name,
        "n": len(traces),
        "top_k": top_k,
        "budget_chars": budget,
        "hit1": sum(t["hit1"] for t in traces),
        "hit3": sum(t["hit3"] for t in traces),
        "context_hit": sum(t["context_hit"] for t in traces),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
        "cost_usd": 0,
    }
    return summary, traces


def main():
    ap = argparse.ArgumentParser(description="So sánh dense / BM25 / hybrid trên một split")
    ap.add_argument("--questions", default="data/questions.json")
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--budget", type=int, default=800, help="context budget (ký tự)")
    ap.add_argument("--candidates", type=int, default=20, help="số ứng viên mỗi nhánh của hybrid")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        questions = [q for q in json.load(f) if q["split"] == args.split]
    if not questions:
        raise SystemExit(f"Không có câu hỏi nào thuộc split '{args.split}'")

    dense, bm25 = DenseRetriever(), BM25Retriever()
    hybrid = HybridRetriever(dense, bm25, candidates=args.candidates)
    retrievers = {
        "dense": dense,
        "bm25": bm25,
        "hybrid_rrf": hybrid,
        "hybrid_rerank": RerankRetriever(hybrid, candidates=20),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, r in retrievers.items():
        summary, traces = evaluate(name, r, questions, top_k=args.top_k, budget=args.budget)
        rows.append(summary)
        (args.output / f"traces_{name}.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'strategy':<14} {'Hit@1':>8} {'Hit@3':>8} {'Context':>8} {'ms':>8}")
    for s in rows:
        n = s["n"]
        print(f"{s['strategy']:<14} {s['hit1']:>5}/{n} {s['hit3']:>5}/{n} "
              f"{s['context_hit']:>5}/{n} {s['median_ms']:>8}")
    print(f"Đã lưu summary.json và traces_*.json vào {args.output}")


if __name__ == "__main__":
    main()