# app.py
import os
from dotenv import load_dotenv
import streamlit as st

from utils import extract_text_from_pdf
from ai_agent import (
    create_vector_store,
    get_summary_from_text,
    get_qa_answer,
    get_quiz_from_text,
    get_open_questions_from_text,
    build_study_plan,
)

# ----------------- Setup -----------------
load_dotenv()

st.set_page_config(
    page_title="AI Smart Study Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("AI-Powered Smart Study Assistant 🎓")
st.caption("Summarize • Ask • Quiz • Open Questions • Plan")

with st.sidebar:
    st.header("1) Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.markdown("---")
    st.header("2) Settings")
    top_k = st.slider("RAG: Top-K chunks", min_value=2, max_value=8, value=4)
    summary_len = st.slider("Summary: bullet points", 4, 12, 8)
    num_mcq = st.slider("Quiz: number of MCQs", 3, 10, 5)
    num_open = st.slider("Open Questions: count", 6, 20, 12)

# Initialize session state
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "last_file" not in st.session_state:
    st.session_state.last_file = None

# Process file
if uploaded_file:
    if st.session_state.last_file != uploaded_file.name:
        with st.spinner("Reading PDF and building vector store..."):
            text = extract_text_from_pdf(uploaded_file)
            index, chunks = create_vector_store(text)
            st.session_state.doc_text = text
            st.session_state.faiss_index = index
            st.session_state.chunks = chunks
            st.session_state.last_file = uploaded_file.name
        st.success(f"Loaded: {uploaded_file.name}")

# Tabs for features
tab_sum, tab_qa, tab_quiz, tab_qgen, tab_plan = st.tabs(
    ["Summarization", "Document Q&A (RAG)", "Quiz Generator (MCQ)", "Question Generator (Open-Ended)", "Study Planner"]
)

# ------------- Summarization -------------
with tab_sum:
    st.subheader("Summarize the Document")
    disabled = not bool(st.session_state.doc_text)
    if st.button("Generate Summary", disabled=disabled):
        with st.spinner("Summarizing with Gemini..."):
            summary = get_summary_from_text(st.session_state.doc_text, sentences=summary_len)
        st.markdown(summary)

# ------------- Document Q&A (RAG) -------------
with tab_qa:
    st.subheader("Ask Questions From the PDF")
    q = st.text_input("Your question", placeholder="e.g., What is the main contribution described in section 3?")
    btn = st.button("Answer from the Document", disabled=not (st.session_state.faiss_index and q))
    if btn:
        with st.spinner("Retrieving context and answering..."):
            result = get_qa_answer(
                q,
                st.session_state.faiss_index,
                st.session_state.chunks,
                k=top_k
            )
        st.markdown("**Answer:**")
        st.write(result["answer"])
        with st.expander("Show sources (retrieved chunks)"):
            for h in result["sources"]:
                st.markdown(f"- **Rank {h['rank']+1} • Score {h['score']:.3f}**")
                st.code(h["text"][:1500])

# ------------- Quiz Generator -------------
with tab_quiz:
    st.subheader("Generate Multiple-Choice Quiz")
    btn_quiz = st.button("Create MCQ Quiz", disabled=not st.session_state.doc_text)
    if btn_quiz:
        with st.spinner("Creating quiz with Gemini..."):
            data = get_quiz_from_text(st.session_state.doc_text, num_q=num_mcq)
        quiz = data.get("quiz", [])
        if not quiz:
            st.warning("Could not parse a structured quiz. Showing raw output.")
            st.code(data.get("raw", ""))
        else:
            for i, q in enumerate(quiz, start=1):
                st.markdown(f"**Q{i}. {q['question']}**")
                for j, opt in enumerate(q["options"]):
                    st.write(f"{chr(65+j)}. {opt}")
                st.markdown(f"**Answer:** {chr(65 + q['correct_index'])}")
                st.caption(q.get("explanation", ""))

# ------------- Question Generator (Open-Ended) -------------
with tab_qgen:
    st.subheader("Generate Open-Ended Study Questions")
    btn_qgen = st.button("Create Open-Ended Questions", disabled=not st.session_state.doc_text)
    if btn_qgen:
        with st.spinner("Generating questions with Gemini..."):
            qs = get_open_questions_from_text(st.session_state.doc_text, count=num_open)
        for i, qq in enumerate(qs, start=1):
            st.write(f"{i}. {qq}")

# ------------- Study Planner -------------
with tab_plan:
    st.subheader("Plan Your Study")
    with st.form(key="study_form"):
        subs = st.text_input("Subjects (comma-separated)", placeholder="OS, DS, DDCO, Math, Java")
        days = st.number_input("Days available", min_value=1, value=7)
        hrs = st.number_input("Hours per day", min_value=0.5, value=3.0, step=0.5)
        submitted = st.form_submit_button("Generate Plan")
    if submitted:
        subjects = [s.strip() for s in subs.split(",") if s.strip()]
        plan = build_study_plan(subjects, int(days), float(hrs))
        if not plan["daily_plan"]:
            st.warning("Please enter valid subjects and time.")
        else:
            st.markdown(f"**Total Hours:** {plan['total_hours']}")
            for day in plan["daily_plan"]:
                with st.expander(f"Day {day['day']}"):
                    for slot in day["slots"]:
                        st.write(f"- {slot['subject']}: {slot['hours']} hrs")

st.markdown("---")
st.caption("Tip: If a PDF is scanned or image-only, extract text first (OCR) before uploading.")
