import chromadb
from sentence_transformers import SentenceTransformer
import json
import re
import unicodedata
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class DenseRetriever:
    def __init__(self, chroma_path="starter/data/chroma_db", collection="day6_A",
                 model_name="intfloat/multilingual-e5-small"):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.col = chromadb.PersistentClient(path=chroma_path).get_collection(collection)

    def search(self, query, *, filters=None, top_k=3):
        vec = self.model.encode(f"query: {query}", normalize_embeddings=True).tolist()
        r = self.col.query(query_embeddings=[vec], n_results=top_k, where=filters or None,
                           include=["documents", "metadatas", "distances"])
        out = []
        for cid, doc, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]):
            meta = dict(meta)
            out.append({
                "id": cid, "text": doc,
                "source_id": meta.pop("document_id"),
                "source": meta.pop("source"),
                "page": int(meta.pop("page")),
                "metadata": meta,
                "score": round(1 - dist, 4),
            })
        return out


def tokenize(text):
    # Giống hệt baseline.py để so sánh công bằng với TF-IDF
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


class BM25Retriever:
    def __init__(self, chunks_path="starter/data/chunks_A.json", k1=1.5, b=0.75):
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks], k1=k1, b=b)

    def search(self, query, *, filters=None, top_k=3):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        out = []
        for i in ranked:
            c = self.chunks[i]
            if scores[i] <= 0:
                break
            if filters and any(c["metadata"].get(k) != v for k, v in filters.items()):
                continue
            out.append({**c, "score": round(float(scores[i]), 4)})
            if len(out) == top_k:
                break
        return out

class HybridRetriever:
    def __init__(self, dense, bm25, k=60, candidates=20):
        self.dense, self.bm25, self.k, self.candidates = dense, bm25, k, candidates

    def search(self, query, *, filters=None, top_k=3):
        lists = [self.dense.search(query, filters=filters, top_k=self.candidates),
                 self.bm25.search(query, filters=filters, top_k=self.candidates)]
        fused, by_id = {}, {}
        for results in lists:
            for rank, c in enumerate(results, start=1):
                fused[c["id"]] = fused.get(c["id"], 0) + 1 / (self.k + rank)
                by_id[c["id"]] = c
                ranked_ids = sorted(fused, key=lambda cid: -fused[cid])
        out = []
        for cid in ranked_ids[:top_k]:
            c = dict(by_id[cid])
            c["score"] = round(fused[cid], 4)
            out.append(c)
        return out



class RerankRetriever:
    """Lấy nhiều ứng viên từ một retriever gốc rồi xếp lại bằng cross-encoder."""

    def __init__(self, base, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", candidates=20):
        self.base = base
        self.model = CrossEncoder(model_name, device="cpu")
        self.candidates = candidates

    def search(self, query, *, filters=None, top_k=3):
        cands = self.base.search(query, filters=filters, top_k=self.candidates)
        if not cands:
            return []
        scores = self.model.predict([(query, c["text"]) for c in cands])
        ranked = sorted(zip(cands, scores), key=lambda x: -x[1])[:top_k]
        return [{**c, "score": round(float(s), 4)} for c, s in ranked]