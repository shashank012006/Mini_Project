# ai_agent.py
import os
import faiss
import numpy as np
from typing import List, Tuple, Dict
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import google.generativeai as genai

# ---------- Setup ----------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# Choose a fast, widely used embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitter tuned for PDFs / academic docs
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    add_start_index=True,
    separators=["\n\n", "\n", " ", ""]
)

# One-time global embedding model (small + quick)
_embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

def _embed(texts: List[str]) -> np.ndarray:
    embs = _embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embs, dtype="float32")

# ---------- Vector Store ----------
def create_vector_store(raw_text: str) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Splits raw_text into chunks, embeds, and builds a FAISS index (Inner Product for cosine on normalized vectors).
    Returns (index, chunks_meta) where chunks_meta[i]["text"] is the original chunk.
    """
    if not raw_text.strip():
        raise ValueError("Empty text. Upload a PDF with readable text.")
    docs = _text_splitter.create_documents([raw_text])
    chunks = [{"text": d.page_content, "start_index": d.metadata.get("start_index", 0)} for d in docs]

    vectors = _embed([c["text"] for c in chunks])
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine when vectors normalized
    index.add(vectors)
    return index, chunks

def _retrieve(query: str, index: faiss.IndexFlatIP, chunks: List[Dict], k: int = 4) -> List[Dict]:
    qvec = _embed([query])
    D, I = index.search(qvec, k)
    hits = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        hits.append({
            "rank": rank,
            "score": float(D[0][rank]),
            "text": chunks[idx]["text"]
        })
    return hits

# ---------- Gemini helpers ----------
def _gemini(model_name: str = "gemini-1.5-flash"):
    if not GEMINI_KEY:
        raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY in .env")
    return genai.GenerativeModel(model_name)

# ---------- Core tasks ----------
def get_summary_from_text(text: str, sentences: int = 8) -> str:
    model = _gemini()
    prompt = f"""You are an expert study assistant.
Summarize the document below in about {sentences} concise bullet points.
Focus on main concepts, definitions, and results. Avoid fluff.

DOCUMENT:
{text}
"""
    resp = model.generate_content(prompt)
    return resp.text.strip()

def get_qa_answer(question: str,
                  index: faiss.IndexFlatIP,
                  chunks: List[Dict],
                  k: int = 4) -> Dict:
    hits = _retrieve(question, index, chunks, k=k)
    context = "\n\n---\n\n".join(h["text"] for h in hits)

    model = _gemini()
    prompt = f"""You are a strict RAG assistant.
Answer the user's question using ONLY the context below.
If the answer is not contained in the context, say exactly:
"Answer not found in the uploaded document."

CONTEXT:
{context}

QUESTION:
{question}

Rules:
- Quote or paraphrase only from CONTEXT.
- If not present in CONTEXT, respond with the exact sentence above.
"""
    resp = model.generate_content(prompt)
    answer = resp.text.strip()
    return {"answer": answer, "sources": hits}

def get_quiz_from_text(text: str, num_q: int = 5) -> Dict:
    model = _gemini()
    prompt = f"""You are an expert quiz creator.
Generate {num_q} multiple-choice questions (MCQ) from the document below.
Return STRICT JSON with this schema:
{{
  "quiz": [
    {{
      "question": "...",
      "options": ["A", "B", "C", "D"],
      "correct_index": 0,
      "explanation": "why the answer is correct"
    }}
  ]
}}

Document:
{text}
"""
    resp = model.generate_content(prompt)
    # In case Gemini wraps JSON with backticks or prose, try to extract JSON portion:
    import json, re
    raw = resp.text
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        raw = m.group(0)
    try:
        data = json.loads(raw)
    except Exception:
        # Fallback: present text as-is
        data = {"quiz": [], "raw": resp.text}
    return data

def get_open_questions_from_text(text: str, count: int = 12) -> List[str]:
    model = _gemini()
    prompt = f"""Generate {count} open-ended study questions from the document below.
Mix recall ("define/describe"), understanding ("explain/compare"), and application ("apply/argue") prompts.
Return them as a simple numbered list without extra explanations.

Document:
{text}
"""
    resp = model.generate_content(prompt)
    lines = [ln.strip(" -") for ln in resp.text.splitlines() if ln.strip()]
    # Normalize to plain list (strip numbering if present)
    cleaned = []
    import re
    for ln in lines:
        cleaned.append(re.sub(r"^\d+[\).:\-]\s*", "", ln))
    return cleaned

def build_study_plan(subjects: List[str], days: int, hours_per_day: float) -> Dict:
    """
    Simple even allocation: spread subjects across days, with recap buffer.
    Returns dict: { "daily_plan": [ {day, slots:[{subject, hours}]} ], "total_hours": ... }
    """
    if days <= 0 or hours_per_day <= 0 or not subjects:
        return {"daily_plan": [], "total_hours": 0}

    total_hours = days * hours_per_day
    slots_per_day = max(1, min(len(subjects), 4))  # 1..4 slots/day to avoid fragmentation
    # Allocate hours: 90% content + 10% recap
    focus_hours = round(hours_per_day * 0.9, 2)
    recap_hours = round(hours_per_day - focus_hours, 2)

    plan = []
    si = 0
    for d in range(1, days + 1):
        today = {"day": d, "slots": []}
        per_slot = round(focus_hours / slots_per_day, 2)
        for _ in range(slots_per_day):
            subj = subjects[si % len(subjects)]
            today["slots"].append({"subject": subj, "hours": per_slot})
            si += 1
        if recap_hours > 0:
            today["slots"].append({"subject": "Review/Revision", "hours": recap_hours})
        plan.append(today)

    return {"daily_plan": plan, "total_hours": total_hours}
