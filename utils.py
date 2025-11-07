# utils.py
from typing import List
from pypdf import PdfReader

def extract_text_from_pdf(file) -> str:
    """
    Reads a Streamlit uploaded file or a file path and returns concatenated text.
    """
    if hasattr(file, "read"):  # Streamlit UploadedFile
        reader = PdfReader(file)
    else:
        reader = PdfReader(str(file))

    pages: List[str] = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(txt)
    return "\n".join(pages).strip()
