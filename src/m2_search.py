"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
import re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    # Lightweight fallback segmentation: normalize and split on non-word characters.
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # keep unicode letters and numbers
    tokens = re.findall(r"[\w]+", text, flags=re.UNICODE)
    return " ".join(tokens)


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks or []
        self.corpus_tokens = []
        for c in self.documents:
            seg = segment_vietnamese(c.get("text", ""))
            tokens = [t for t in seg.split() if t]
            self.corpus_tokens.append(tokens)

        # compute DF and IDF for simple BM25
        import math
        df = {}
        for tokens in self.corpus_tokens:
            seen = set()
            for t in tokens:
                if t in seen:
                    continue
                df[t] = df.get(t, 0) + 1
                seen.add(t)
        n = max(len(self.corpus_tokens), 1)
        self.idf = {t: math.log((n - v + 0.5) / (v + 0.5) + 1) for t, v in df.items()}
        self.avgdl = sum(len(toks) for toks in self.corpus_tokens) / max(len(self.corpus_tokens), 1)
        self.k1 = 1.5
        self.b = 0.75

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not getattr(self, "corpus_tokens", None):
            return []
        qtokens = [t for t in segment_vietnamese(query).split() if t]
        scores = []
        for idx, doc_tokens in enumerate(self.corpus_tokens):
            score = 0.0
            freqs = {}
            for t in doc_tokens:
                freqs[t] = freqs.get(t, 0) + 1
            dl = len(doc_tokens)
            for q in qtokens:
                if q not in freqs:
                    continue
                idf = self.idf.get(q, 0.0)
                freq = freqs[q]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                score += idf * (freq * (self.k1 + 1)) / denom
            scores.append((score, idx))
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, idx in scores[:top_k]:
            doc = self.documents[idx]
            results.append(SearchResult(text=doc.get("text", ""), score=float(s), metadata=doc.get("metadata", {}), method="bm25"))
        return results


class DenseSearch:
    def __init__(self):
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception:
            self.client = None
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception:
                self._encoder = None
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if self.client is None:
            # Qdrant client not available — skip dense indexing
            return
        encoder = self._get_encoder()
        if encoder is None:
            return
        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct
            self.client.recreate_collection(collection, VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
            texts = [c["text"] for c in chunks]
            vectors = encoder.encode(texts, show_progress_bar=False)
            points = [PointStruct(id=i, vector=v.tolist() if hasattr(v, 'tolist') else list(v), payload={**c["metadata"], "text": c["text"]}) for i, (v, c) in enumerate(zip(vectors, chunks))]
            self.client.upsert(collection=collection, points=points)
        except Exception:
            return

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if self.client is None:
            return []
        encoder = self._get_encoder()
        if encoder is None:
            return []
        try:
            qv = encoder.encode(query)
            qvec = qv.tolist() if hasattr(qv, 'tolist') else list(qv)
            hits = self.client.search(collection=collection, query_vector=qvec, limit=top_k)
            out = []
            for h in hits:
                payload = getattr(h, 'payload', {}) or {}
                text = payload.get("text") or payload.get("_text") or ""
                score = getattr(h, 'score', 0.0)
                out.append(SearchResult(text=text, score=float(score), metadata=payload, method="dense"))
            return out
        except Exception:
            return []


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf = {}
    for lst in results_list:
        for rank, res in enumerate(lst):
            key = res.text
            if key not in rrf:
                rrf[key] = {"score": 0.0, "best": res}
            rrf[key]["score"] += 1.0 / (k + rank + 1)

    merged = sorted(rrf.values(), key=lambda x: x["score"], reverse=True)
    out = []
    for item in merged[:top_k]:
        r = item["best"]
        out.append(SearchResult(text=r.text, score=float(item["score"]), metadata=r.metadata, method="hybrid"))
    return out


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
