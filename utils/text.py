import re
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

HEADER_FOOTER_PATTERN = re.compile(r"(^.*Page \d+.*$)|(^\s*\d+\s*$)", re.MULTILINE)

def clean_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t)
    t = HEADER_FOOTER_PATTERN.sub(" ", t)
    return t.strip()

def chunk_sections(pages: List[Dict], chunk_size: int, chunk_overlap: int, source_name: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    chunks = []
    for p in pages:
        cleaned = clean_text(p["text_raw"]) or ""
        parts = splitter.split_text(cleaned)
        for i, chunk in enumerate(parts):
            chunks.append({
                "content": chunk,
                "metadata": {
                    "source": source_name,
                    "page": p["page"],
                    "chunk_id": f"{source_name}_p{p['page']}_{i}",
                }
            })
    return chunks
