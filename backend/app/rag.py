"""Small retrieval index. Uses ChromaDB when SEAM_VECTOR=chroma, otherwise TF-IDF (no downloads, works offline)."""
import os, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk(text: str, size: int = 700):
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if len(cur) + len(para) > size and cur:
            parts.append(cur.strip()); cur = ""
        cur += para + "\n\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


class Index:
    def __init__(self):
        self.reset()

    def reset(self):
        self.items, self._vec, self._mat, self._col = [], None, None, None

    def add(self, text: str, source: str):
        for c in chunk(text):
            self.items.append({"text": c, "source": source})
        self._vec = None
        self._col = None

    def _build(self):
        if os.getenv("SEAM_VECTOR") == "chroma":
            try:
                import chromadb
                client = chromadb.Client()
                try:
                    client.delete_collection("seam")
                except Exception:
                    pass
                self._col = client.create_collection("seam")
                self._col.add(ids=[str(i) for i in range(len(self.items))],
                              documents=[i["text"] for i in self.items],
                              metadatas=[{"source": i["source"]} for i in self.items])
                return
            except Exception:
                self._col = None
        self._vec = TfidfVectorizer(stop_words="english").fit([i["text"] for i in self.items])
        self._mat = self._vec.transform([i["text"] for i in self.items])

    def query(self, q: str, k: int = 3):
        if not self.items:
            return []
        if self._vec is None and self._col is None:
            self._build()
        if self._col is not None:
            r = self._col.query(query_texts=[q], n_results=min(k, len(self.items)))
            return [{"text": d, "source": m["source"]} for d, m in zip(r["documents"][0], r["metadatas"][0])]
        sims = cosine_similarity(self._vec.transform([q]), self._mat)[0]
        return [self.items[i] for i in sims.argsort()[::-1][:k] if sims[i] > 0]
