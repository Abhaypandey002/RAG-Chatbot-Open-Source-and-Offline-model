import os
import glob
import sqlite3
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from utils.pdf import extract_pdf
from utils.text import chunk_sections
from models.embedder import embed_texts
from storage.vectorstore import FaissStore
from storage.logger import JsonlLogger

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data/pdfs"))
FAISS_DIR = os.getenv("FAISS_DIR", "./storage/faiss_index")
META_DB = os.getenv("META_DB", "./storage/meta.sqlite")
LOGS_DIR = os.getenv("LOGS_DIR", "./storage/logs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

logger = JsonlLogger(LOGS_DIR)
store = FaissStore(FAISS_DIR)

def ingest_folder(folder: str) -> tuple[int, int]:
    folder_path = Path(folder)
    pdf_paths = sorted(glob.glob(str(folder_path / "*.pdf")))
    if not pdf_paths:
        return (0, 0)

    conn = sqlite3.connect(META_DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
      id TEXT PRIMARY KEY,
      source TEXT,
      page INTEGER,
      content TEXT
    )
    """)
    conn.commit()

    all_chunks = []
    for path in pdf_paths:
        logger.log("pdf_found", {"path": path})
        pages = extract_pdf(path)
        source_name = Path(path).name
        chunks = chunk_sections(pages, CHUNK_SIZE, CHUNK_OVERLAP, source_name)
        all_chunks.extend(chunks)

    metas = []
    for ch in all_chunks:
        md = ch["metadata"]
        c.execute(
            "INSERT OR REPLACE INTO chunks (id, source, page, content) VALUES (?, ?, ?, ?)",
            (md["chunk_id"], md["source"], md["page"], ch["content"])
        )
        metas.append({"content": ch["content"], "metadata": md})

    conn.commit(); conn.close()

    if not metas:
        return (len(pdf_paths), 0)

    texts = [m["content"] for m in metas]
    vecs = embed_texts(texts)
    store.add(np.array(vecs, dtype="float32"), metas)
    logger.log("ingest_complete", {"pdf_count": len(pdf_paths), "chunk_count": len(metas)})
    return (len(pdf_paths), len(metas))

if __name__ == "__main__":
    n_pdfs, n_chunks = ingest_folder(str(DATA_DIR))
    print(f"Ingested {n_pdfs} PDFs → {n_chunks} chunks.")
