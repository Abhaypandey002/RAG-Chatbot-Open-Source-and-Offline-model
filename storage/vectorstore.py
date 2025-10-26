import os, json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

class FaissStore:
    def __init__(self, dir_path: str):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.faiss"
        self.meta_path = self.dir / "meta.jsonl"
        self.index = None
        self.dim = None

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.dim = dim
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
            else:
                self.index = faiss.IndexFlatIP(dim)  # cosine if vectors normalized

    def add(self, vectors: np.ndarray, metadatas: List[Dict]):
        self._ensure_index(vectors.shape[1])
        self.index.add(vectors)
        with self.meta_path.open("a", encoding="utf-8") as f:
            for m in metadatas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        faiss.write_index(self.index, str(self.index_path))

    def _read_meta(self) -> List[Dict]:
        if not self.meta_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.meta_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, List[Dict]]:
        self._ensure_index(query_vec.shape[1])
        D, I = self.index.search(query_vec, top_k)
        metas = self._read_meta()
        results = []
        for idx in I[0]:
            if idx == -1 or idx >= len(metas):
                results.append({})
            else:
                results.append(metas[idx])
        return D[0], results
