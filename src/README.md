# Code — Đào Thị Ngọc Anh · 2A202603790

| File | Công dụng |
|---|---|
| `export_chroma.py` | Xuất chunk từ Chroma Day 6 (`day6_A`/`day6_B`) sang định dạng chunk JSON của Day 7 |
| `retrievers.py` | `DenseRetriever` (e5-small + Chroma), `BM25Retriever`, `HybridRetriever` (RRF, k=60), `RerankRetriever` (cross-encoder) |
| `eval_retrieval.py` | So sánh 4 phương pháp trên cùng split, tính Hit@1, Evidence Hit@3, Context Evidence Rate, thời gian; lưu `summary.json` và `traces_*.json` |
| `controller.py` | Controller `gpt-4o-mini` (tool calling với tool `search`), ghi token usage |
| `run_agent.py` | Chạy agent thật (`starter/agentic.py::run_agent`) trên `data/agent_cases.json`, lưu trace và config từng câu |

Cần `OPENAI_API_KEY` trong `.env` (không commit) và `chroma_db` của Day 6 tại `starter/data/chroma_db`. Lệnh chạy đầy đủ có ở cuối [báo cáo](../report/individual/2A202603790_DaoThiNgocAnh.md).
