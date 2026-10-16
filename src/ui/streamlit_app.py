"""Task 7 — Streamlit Frontend Application.

Features:
- Question input box with quick-select example questions.
- Answer display in styled card.
- Dynamic Confidence Badge: 🟢 High / 🟡 Medium / 🔴 Low.
- Expandable sources with clickable DailyMed links.
- Visible notice when a query is sent for human review on abstain.
- Prominent fixed medical advice disclaimer.
- Demo mode detection when GEMINI_API_KEY is not configured.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path so app can run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import streamlit as st

from src.config import DISCLAIMER_TEXT, GEMINI_API_KEY

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")

st.set_page_config(
    page_title="Drug Label QA — FDA Reference Assistant",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-high {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-medium {
        background-color: #FEF9C3;
        color: #A16207;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .review-notice {
        background-color: #FFF7ED;
        border-left: 4px solid #EA580C;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 12px;
        margin-bottom: 12px;
        color: #9A3412;
        font-size: 0.95rem;
    }
    .disclaimer-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration & Info")
    st.markdown("**Architecture**: Hybrid RRF (ChromaDB + BM25)")
    st.markdown("**Index**: 61 FDA Drug Labels (894 Chunks)")
    st.markdown(f"**Backend Endpoint**: `{API_URL}`")

    if not GEMINI_API_KEY:
        st.info("ℹ️ **Demo Mode Active**: No `GEMINI_API_KEY` detected. Running with local label data & mock responses.")
    else:
        st.success("🟢 **Live Gemini API Connected**")

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    sample_queries = [
        "What is the maximum daily dose of metformin for adults?",
        "Can I take ibuprofen with warfarin?",
        "What are the contraindications for lisinopril?",
        "Can amlodipine cause peripheral edema?",
        "Does atorvastatin cure bacterial pneumonia?",
    ]
    for sq in sample_queries:
        if st.button(sq, key=f"btn_{sq}"):
            st.session_state["question_input"] = sq

# ── Main Content ───────────────────────────────────────────────────────

st.markdown('<div class="main-title">💊 FDA Drug Label Reference QA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Plain-language questions answered strictly from official FDA labeling data with verifiable citations.</div>',
    unsafe_allow_html=True,
)

default_q = st.session_state.get("question_input", "")
question = st.text_input(
    "Ask a question about a drug label:",
    value=default_q,
    placeholder="e.g. What is the maximum daily dose of metformin?",
    key="main_question_box",
)

col1, col2 = st.columns([1, 4])
with col1:
    submit_pressed = st.button("Submit Question", type="primary", use_container_width=True)

if submit_pressed and question.strip():
    with st.spinner("Searching FDA label index and generating answer..."):
        # Attempt to call FastAPI backend; fallback to direct AnswerEngine if API not running
        response_data = None
        try:
            resp = requests.post(API_URL, json={"question": question.strip()}, timeout=15)
            if resp.status_code == 200:
                response_data = resp.json()
            elif resp.status_code == 503:
                st.error("⚠️ 503 Service Unavailable: The LLM provider is currently overloaded. Please try again shortly.")
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception:
            # Fall back to in-process AnswerEngine if backend server is not running
            from src.answer.engine import AnswerEngine
            engine = AnswerEngine()
            direct_res = engine.answer_question(question.strip())
            response_data = direct_res.to_dict()

        if response_data:
            answer = response_data.get("answer", "")
            confidence = response_data.get("confidence", "low")
            abstained = response_data.get("abstained", False)
            sources = response_data.get("sources", [])
            review_queued = response_data.get("review_queued", False)

            st.markdown("### Answer")

            # Confidence Badge
            if confidence == "high":
                badge_html = '<span class="badge-high">🟢 High Confidence</span>'
            elif confidence == "medium":
                badge_html = '<span class="badge-medium">🟡 Medium Confidence</span>'
            else:
                badge_html = '<span class="badge-low">🔴 Low Confidence (Abstained)</span>'

            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown("")

            # Answer Card
            st.markdown(f"> {answer}")

            # Abstain / Human Review Notice
            if abstained or review_queued:
                st.markdown(
                    """
                    <div class="review-notice">
                        📋 <strong>Sent for Human Review</strong>: This query had low retrieval confidence or insufficient evidence in the official FDA label and has been logged to the review queue for regulatory spot-check.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Sources Section
            if sources:
                with st.expander(f"📚 Retrieved Sources & DailyMed Links ({len(sources)})", expanded=True):
                    for i, s in enumerate(sources, 1):
                        drug = s.get("drug") or s.get("drug_name") or "label"
                        sec = s.get("section") or "general"
                        url = s.get("url") or "#"
                        st.markdown(f"**{i}. [{drug.capitalize()} — {sec.capitalize()}]({url})**")
                        if url and url != "#":
                            st.caption(f"Source URL: {url}")
            else:
                st.info("No sources cited for this response.")

# ── Medical Advice Disclaimer ──────────────────────────────────────────

st.markdown(
    f"""
    <div class="disclaimer-card">
        ⚠️ <strong>Medical Disclaimer</strong>: {DISCLAIMER_TEXT} This tool is for informational and educational reference only and must never be used for clinical decision-making or self-treatment.
    </div>
    """,
    unsafe_allow_html=True,
)
