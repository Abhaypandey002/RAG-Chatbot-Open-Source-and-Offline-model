from sentence_transformers import SentenceTransformer
import os

_model = None

def get_embedder():
    global _model
    if _model is None:
        name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _model = SentenceTransformer(name)
    return _model

def embed_texts(texts):
    model = get_embedder()
    # normalize_embeddings=True makes inner product equivalent to cosine similarity
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
