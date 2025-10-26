# import os
# import numpy as np
# from models.embedder import embed_texts
# from models.llm import build_prompt, generate_answer
# from storage.vectorstore import FaissStore

# TOP_K = int(os.getenv("TOP_K", 5))
# MIN_SIM = float(os.getenv("MIN_SIMILARITY", 0.26))
# FAISS_DIR = os.getenv("FAISS_DIR", "./storage/faiss_index")

# store = FaissStore(FAISS_DIR)

# def retrieve(query: str):
#     q_vec = np.array([embed_texts([query])[0]], dtype="float32")
#     sims, metas = store.search(q_vec, TOP_K)
#     return sims, metas

# def format_contexts(metas):
#     return [m.get("content", "") for m in metas]

# def build_sources(metas):
#     out = []
#     for m in metas:
#         md = m.get("metadata", {})
#         out.append(f"{md.get('source','?')} (p.{md.get('page','?')})")
#     return out

# def answer_query(query: str):
#     sims, metas = retrieve(query)
#     if len(sims) == 0 or float(max(sims)) < MIN_SIM:
#         return None, []
#     contexts = format_contexts(metas)
#     prompt = build_prompt(query, contexts)
#     text = generate_answer(prompt)
#     return text, build_sources(metas)



import os
import re
import numpy as np
from typing import List, Dict, Tuple
from models.embedder import embed_texts
from models.llm import build_prompt, generate_answer
from storage.vectorstore import FaissStore

TOP_K = int(os.getenv("TOP_K", 5))
MIN_SIM = float(os.getenv("MIN_SIMILARITY", 0.26))
FAISS_DIR = os.getenv("FAISS_DIR", "./storage/faiss_index")

store = FaissStore(FAISS_DIR)

PRONOUNS_RE = re.compile(r"\b(he|she|they|them|his|her|their|it|this|that|those|these|above|former|latter)\b", re.I)

def _make_chat_context(turns: List[Dict], max_pairs: int = 6, max_chars: int = 1200) -> str:
    """
    Format recent conversation as brief memory the model can use.
    Each turn is a dict: {"role": "user"|"assistant", "content": "..."}
    We'll include up to max_pairs*2 messages (user+assistant) and trim to max_chars.
    """
    if not turns:
        return ""
    # Take the last messages, but keep order
    clipped = turns[-(max_pairs * 2):]
    lines = []
    for t in clipped:
        role = "User" if t["role"] == "user" else "Assistant"
        text = t["content"].strip()
        lines.append(f"{role}: {text}")
    s = "\n".join(lines)
    if len(s) > max_chars:
        s = "…" + s[-max_chars:]
    return s

def _expand_query_with_memory(query: str, turns: List[Dict]) -> str:
    """
    Lightweight memory for retrieval:
    - If the current query has pronouns or is short, append the last user + assistant messages to help retrieval.
    - Otherwise return the query as is.
    """
    if not turns:
        return query
    needs_memory = bool(PRONOUNS_RE.search(query)) or len(query.strip()) < 8
    if not needs_memory:
        return query
    # grab last 2 turns (user + assistant)
    tail = [t["content"] for t in turns[-4:]]  # small window
    memory_hint = " | ".join(tail)[-400:]  # keep it compact
    return f"{query} | context: {memory_hint}"

def retrieve(query: str, history: List[Dict] | None = None) -> Tuple[np.ndarray, List[Dict]]:
    q = _expand_query_with_memory(query, history or [])
    qv = np.array([embed_texts([q])[0]], dtype="float32")
    sims, metas = store.search(qv, TOP_K)
    return sims, metas

def _format_contexts(metas: List[Dict]) -> List[str]:
    return [m.get("content", "") for m in metas]

def _build_sources(metas: List[Dict]) -> List[str]:
    srcs = []
    for m in metas:
        md = m.get("metadata", {})
        srcs.append(f"{md.get('source','?')} (p.{md.get('page','?')})")
    return srcs

def answer_query(query: str, history: List[Dict] | None = None):
    history = history or []
    sims, metas = retrieve(query, history)
    if len(sims) == 0 or float(max(sims)) < MIN_SIM:
        return None, []

    contexts = _format_contexts(metas)
    chat_ctx = _make_chat_context(history, max_pairs=6, max_chars=1200)
    prompt = build_prompt(query, contexts, chat_context=chat_ctx)
    text = generate_answer(prompt)
    return text, _build_sources(metas)
