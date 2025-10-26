import fitz  # PyMuPDF
from typing import List, Dict

def extract_pdf(path: str) -> List[Dict]:
    """Return a list of {page, text_raw} dicts."""
    out = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            out.append({"page": i + 1, "text_raw": text})
    return out
