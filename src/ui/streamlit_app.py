"""Bangladesh Medical, First-Aid & Drug Assistant (সহকারী).

Features:
- Conversational Chat Interface with Context Memory & Follow-up Resolution.
- Multi-intent clinical routing: Identification, Dosage, Side Effects, Interactions, BD Brand Alternatives.
- Strict Grounding: Answers only from retrieved FDA label text, explicitly returns 'Not Found' otherwise.
- Multi-tier Safety Layer: Persistent Disclaimers & Emergency/Pregnancy/Pediatric Escalations.
- Verifiable Citations with direct DailyMed package insert and DGDA links under every answer.
- Optional Multimodal Image Upload: OCR packaging analysis for medicine blister packs & prescriptions.
- 100% Free & Open-Source: Runs 24/7 on Streamlit Cloud with zero GPU & zero API keys required.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import streamlit as st

from src.answer.chat_assistant import (
    ChatTurnResponse,
    ConversationalMedicalAssistant,
    SafetyAlert,
)
from src.config import DISCLAIMER_TEXT, GEMINI_API_KEY
from src.firstaid.triage import FIRST_AID_PROTOCOLS, find_first_aid_protocol
from src.knowledge.bangladesh_pharma import (
    BANGLADESH_BRANDS,
    BANGLADESH_EMERGENCY_HOTLINES,
    find_brand_alternatives,
    lookup_bangladesh_brand,
)

st.set_page_config(
    page_title="Bangladesh Medical Assistant (সহকারী)",
    page_icon="🇧🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Modern Clinical Look ───────────────────────────────

st.markdown(
    """
    <style>
    .emergency-banner {
        background: linear-gradient(90deg, #DC2626 0%, #991B1B 100%);
        color: white;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 6px -1px rgba(220, 38, 38, 0.2);
    }
    .hotline-chip {
        background-color: white;
        color: #DC2626;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        margin: 0 3px;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.0rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 8px;
    }
    .badge-high {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }
    .badge-medium {
        background-color: #FEF9C3;
        color: #A16207;
        border: 1px solid #FDE047;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }
    .badge-intent {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .safety-alert-card {
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-size: 0.92rem;
    }
    .alert-emergency {
        background-color: #FEF2F2;
        border-left: 5px solid #DC2626;
        color: #991B1B;
    }
    .alert-pregnancy {
        background-color: #FFFBEB;
        border-left: 5px solid #D97706;
        color: #92400E;
    }
    .alert-pediatric {
        background-color: #EFF6FF;
        border-left: 5px solid #2563EB;
        color: #1E40AF;
    }
    .context-focus-bar {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        color: #334155;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── In-Memory Resource Caching ─────────────────────────────────────────


@st.cache_resource
def get_cached_chat_assistant(api_key: str = ""):
    from src.answer.engine import AnswerEngine
    from src.llm.client import GeminiClient, LocalTransformersClient

    if api_key:
        engine = AnswerEngine(llm_client=GeminiClient(api_key=api_key))
    else:
        try:
            engine = AnswerEngine(llm_client=LocalTransformersClient())
        except Exception:
            engine = AnswerEngine()
    return ConversationalMedicalAssistant(answer_engine=engine)


# ── Session State Initialization ───────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 **Hello! I am your Bangladesh Medical & Drug Assistant (সহকারী).**\n\n"
                "I am designed to assist you like a medical consultation companion with conversation memory:\n\n"
                "- 🔍 **Identify medicines**: Ask about brand names (Napa, Seclo, Alatrol) or upload packaging photos.\n"
                "- 📋 **Dosage & Administration**: Official instructions from verified drug labels.\n"
                "- ⚠️ **Side effects & Precautions**: Known adverse reactions and safety warnings.\n"
                "- 🔄 **Interactions**: Ask follow-ups like *'Can I take this with paracetamol?'*\n"
                "- 💊 **Bangladeshi Alternatives**: Find all locally registered brands sharing the generic active molecule.\n"
                "- 🚑 **First-Aid Protocols**: Dengue fever, snakebites, burns, dehydration, and bone trauma.\n\n"
                "⚠️ *Strict Grounding Rule: All pharmaceutical information is verified against official FDA & DGDA package inserts. When evidence is not documented, I explicitly abstain rather than fabricate answers.*"
            ),
            "sources": [],
            "safety_alerts": [],
            "active_drug": None,
            "query_type": "welcome",
            "retrieved_chunks": [],
        }
    ]

if "active_drug" not in st.session_state:
    st.session_state["active_drug"] = None

if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None


# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🌐 Language / ভাষা")
    lang_choice = st.radio(
        "Select Language:",
        ("English", "বাংলা (Bengali)"),
        index=0,
        horizontal=True,
    )
    is_bengali = "বাংলা" in lang_choice
    lang_code = "bn" if is_bengali else "en"

    st.markdown("---")
    st.markdown("### 📞 Bangladesh Emergency (জরুরি নম্বর)")
    for num, hl in BANGLADESH_EMERGENCY_HOTLINES.items():
        st.markdown(f"**{hl['title']}**: `📞 {num}`\n\n*{hl['description']}*")

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    active_key = st.session_state.get("user_gemini_key", "").strip() or GEMINI_API_KEY
    if active_key:
        st.success("🟢 Cloud Gemini 2.0 Flash Active")
    else:
        st.success("🟢 Neural SLM Active (SmolLM-135M on CPU)")
        st.caption("🔒 100% Free, Zero GPU, Zero API Keys — 24/7 Cloud Inference!")

    with st.expander("🔑 Optional Gemini Key"):
        custom_key = st.text_input(
            "Gemini API Key (Optional):",
            value=st.session_state.get("user_gemini_key", ""),
            type="password",
            placeholder="AIzaSy...",
            help="Optional. Leave blank to run 100% free with the open-source CPU model.",
        )
        if custom_key != st.session_state.get("user_gemini_key", ""):
            st.session_state["user_gemini_key"] = custom_key
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Reset Chat / নতুন কথোপকথন", use_container_width=True):
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Conversation memory reset. How can I assist you with medicines or first-aid today?"
                if not is_bengali
                else "কথোপকথনের মেমোরি রিসেট করা হয়েছে। আজ আপনাকে কীভাবে সাহায্য করতে পারি?",
                "sources": [],
                "safety_alerts": [],
                "active_drug": None,
                "query_type": "welcome",
                "retrieved_chunks": [],
            }
        ]
        st.session_state["active_drug"] = None
        st.rerun()


# ── Top Emergency Banner ───────────────────────────────────────────────

st.markdown(
    """
    <div class="emergency-banner">
        <span>🚨 <strong>Bangladesh Emergency Hotlines:</strong> Ambulance/Police: <span class="hotline-chip">999</span> | Health Doctor: <span class="hotline-chip">16263</span> | Poison/Snakebite: <span class="hotline-chip">01700-700700</span></span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Title & Intro ──────────────────────────────────────────────────────

if is_bengali:
    st.markdown('<div class="main-title">🇧🇩 বাংলাদেশ মেডিকেল ও ড্রাগ অ্যাসিস্ট্যান্ট</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">ক্লিনিক্যাল স্মৃতি এবং ভেরিফাইড তথ্যের সাথে ইন্টারঅ্যাক্টিভ মেডিকেল চ্যাট।</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="main-title">🇧🇩 Bangladesh Medical & Drug Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Conversational clinical guidance with context memory, strict grounding, and safety guardrails.</div>', unsafe_allow_html=True)


# ── Navigation Tabs ────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "💬 Medical Assistant (Chat)" if not is_bengali else "💬 মেডিকেল সহকারী (চ্যাট)",
    "💊 Bangladesh Medicine Directory" if not is_bengali else "💊 স্থানীয় ওষুধের তালিকা (ব্র্যান্ড)",
    "🚑 Emergency Protocols & Hospitals" if not is_bengali else "🚑 জরুরি প্রোটোকল ও হাসপাতাল",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Conversational Medical Assistant
# ═══════════════════════════════════════════════════════════════════════

with tab1:
    # Quick Action Chips
    st.markdown("**Quick Topics / দ্রুত প্রশ্ন শুরু করুন:**")
    q_cols = st.columns(6)
    quick_topics = [
        ("💊 What is Napa?", "What is Napa and what is its generic?"),
        ("🔄 Alternatives to Napa", "What are Bangladeshi brand alternatives for Napa?"),
        ("📋 Metformin Dosage", "What is the recommended dosage for metformin?"),
        ("⚠️ Atorvastatin Risks", "What are the common side effects of atorvastatin?"),
        ("🤰 Pregnancy & Seclo", "Can a pregnant woman take Seclo (omeprazole)?"),
        ("🚑 Snakebite First Aid", "Immediate first aid steps for snakebite in Bangladesh"),
    ]

    for idx, (label, prompt_text) in enumerate(quick_topics):
        if q_cols[idx].button(label, use_container_width=True, key=f"chip_{idx}"):
            st.session_state["pending_query"] = prompt_text

    # Active Context Banner
    if st.session_state.get("active_drug"):
        c_col1, c_col2 = st.columns([0.85, 0.15])
        with c_col1:
            st.markdown(
                f"""
                <div class="context-focus-bar">
                    <span>📌 <strong>Active Medicine Focus:</strong> <code>{st.session_state['active_drug'].title()}</code> (Follow-ups like <em>"can I take this with..."</em> or <em>"side effects"</em> will resolve to this medicine)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_col2:
            if st.button("✕ Clear Focus", key="clear_drug_focus", use_container_width=True):
                st.session_state["active_drug"] = None
                st.rerun()

    # Optional Image Attachment Expander
    uploaded_image_file = None
    with st.expander("📷 Upload Medicine Packaging / Blister Pack / Prescription (Optional / ঐচ্ছিক)"):
        uploaded_image_file = st.file_uploader(
            "Upload image for OCR & visual identification:",
            type=["png", "jpg", "jpeg", "webp"],
            key="chat_img_uploader",
            help="Upload a photo of a medicine blister strip, bottle, or prescription.",
        )
        if uploaded_image_file:
            st.image(uploaded_image_file, caption="Attached Medicine Packaging Photo", width=320)

    # Render Conversation Messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            # Display Safety Alerts if present
            if msg.get("safety_alerts"):
                for alert in msg["safety_alerts"]:
                    a_class = "alert-emergency" if alert.alert_type == "EMERGENCY" else (
                        "alert-pregnancy" if alert.alert_type == "PREGNANCY" else "alert-pediatric"
                    )
                    hl_links = " | ".join([f"<strong>📞 {h}</strong>" for h in alert.hotlines]) if alert.hotlines else ""
                    st.markdown(
                        f"""
                        <div class="safety-alert-card {a_class}">
                            <strong>{alert.title}</strong><br/>
                            {alert.message}
                            {f'<br/><span style="margin-top: 6px; display: inline-block;">Hotlines: {hl_links}</span>' if hl_links else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Display Query Type & Confidence Badges for Assistant
            if msg["role"] == "assistant" and msg.get("query_type") not in ("welcome", None):
                q_type = msg.get("query_type", "general")
                conf = msg.get("confidence", "high")

                # Type badge
                type_labels = {
                    "identify": "💊 Medicine Identification",
                    "dosage": "📋 Dosage & Administration",
                    "side_effects": "⚠️ Adverse Effects & Warnings",
                    "interactions": "🔄 Drug Interactions",
                    "alternatives": "🔄 Bangladeshi Brand Alternatives",
                    "first_aid": "🚑 First-Aid Protocol",
                    "rag": "📚 FDA Label RAG",
                }
                type_badge = type_labels.get(q_type.split("_")[0], "📚 Clinical Information")
                st.markdown(f'<span class="badge-pill badge-intent">{type_badge}</span>', unsafe_allow_html=True)

                # Confidence badge
                if conf == "high":
                    st.markdown('<span class="badge-pill badge-high">🟢 Grounded in Labeling</span>', unsafe_allow_html=True)
                elif conf == "medium":
                    st.markdown('<span class="badge-pill badge-medium">🟡 Moderate Match</span>', unsafe_allow_html=True)
                elif conf == "abstained":
                    st.markdown('<span class="badge-pill badge-low">🔴 Abstained (Strict Grounding)</span>', unsafe_allow_html=True)

            # Main content text
            st.markdown(msg["content"])

            # Sources Drawer
            if msg.get("sources"):
                with st.expander(f"📚 Verified Sources & Package Inserts ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.markdown(f"- **{s.get('title', 'Official Label')}** — [View Source Link]({s.get('url', 'https://dailymed.nlm.nih.gov')})")

            # Evidence Chunks Drawer (Strict Grounding Inspection)
            if msg.get("retrieved_chunks"):
                with st.expander(f"🔬 Inspect RAG Evidence Passages ({len(msg['retrieved_chunks'])})"):
                    st.caption("These context passages were retrieved from ChromaDB + BM25 to answer your question strictly without fabrication:")
                    for i, ch in enumerate(msg["retrieved_chunks"], 1):
                        st.markdown(f"**Passage #{i}** — `{ch.get('drug', '').title()}` | Section: `{ch.get('section', '')}` | **Similarity:** `{ch.get('similarity', 0.0)}`")
                        st.text(ch.get("text", "")[:400] + "...")

    # Chat Input Bar & Execution
    chat_prompt = st.chat_input(
        "Ask a question, follow-up, or drug inquiry (e.g. 'Can I take this with paracetamol?')..."
        if not is_bengali
        else "ওষুধ বা প্রাথমিক চিকিৎসা বিষয়ক প্রশ্ন লিখুন (যেমনঃ 'নাপা খাওয়ার নিয়ম কী?')..."
    )

    # Handle pending query from quick topic chips
    if st.session_state.get("pending_query"):
        chat_prompt = st.session_state["pending_query"]
        st.session_state["pending_query"] = None

    if chat_prompt:
        # Load attached image if provided
        pil_img = None
        if uploaded_image_file is not None:
            try:
                pil_img = Image.open(uploaded_image_file)
            except Exception:
                pil_img = None

        # Add user message to state and display
        st.session_state["messages"].append({"role": "user", "content": chat_prompt})
        with st.chat_message("user"):
            st.markdown(chat_prompt)
            if pil_img is not None:
                st.image(pil_img, caption="Attached Packaging Image", width=240)

        # Process Turn with Conversational Medical Assistant
        with st.chat_message("assistant"):
            with st.spinner("Analyzing clinical labeling & local protocols..." if not is_bengali else "চিকিৎসা ও ওষুধের তথ্য বিশ্লেষণ করা হচ্ছে..."):
                assistant = get_cached_chat_assistant(active_key)
                
                # Prepare conversation history for antecedent resolution
                conv_history = []
                for m in st.session_state["messages"]:
                    if m.get("role") == "user":
                        conv_history.append({"user": m["content"]})
                    elif m.get("role") == "assistant":
                        if conv_history:
                            conv_history[-1]["active_drug"] = m.get("active_drug")
                            conv_history[-1]["reply"] = m["content"]

                response: ChatTurnResponse = assistant.respond(
                    user_message=chat_prompt,
                    image=pil_img,
                    conversation_history=conv_history,
                    language=lang_code,
                )

                # Update active drug context in session state
                if response.active_drug:
                    st.session_state["active_drug"] = response.active_drug

                # Store assistant response in history
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": response.reply_text,
                    "confidence": response.confidence,
                    "sources": response.sources,
                    "safety_alerts": response.safety_alerts,
                    "active_drug": response.active_drug,
                    "query_type": response.query_type,
                    "retrieved_chunks": response.retrieved_chunks,
                })

                st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: Bangladesh Brand & Medicine Directory
# ═══════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 🔍 Search Common Bangladeshi Medicines (জনপ্রিয় ওষুধ অনুসন্ধান)")
    search_term = st.text_input("Type a brand or generic name (e.g. Napa, Seclo, Sergel, Alatrol, Zimax, Monas):", "")

    filtered_brands = list(BANGLADESH_BRANDS.items())
    if search_term.strip():
        term = search_term.lower()
        filtered_brands = [
            (k, v) for k, v in filtered_brands
            if term in k or term in v.generic_name.lower() or term in v.manufacturer.lower()
        ]

    for brand_key, info in filtered_brands:
        with st.expander(f"💊 {info.brand_name} ({info.generic_name.title()}) — {info.manufacturer}"):
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown(f"**Generic (মূল উপাদান)**: {info.generic_name.title()} ({info.bengali_generic})")
                st.markdown(f"**Manufacturer (কোম্পানি)**: {info.manufacturer}")
                st.markdown(f"**Strength / Form**: {info.strength} ({info.dosage_form})")
                st.markdown(f"**OTC Status**: {'Over The Counter (প্রেসক্রিপশন ছাড়া পাওয়া যায়)' if info.otc else 'Prescription Only (ডাক্তারের প্রেসক্রিপশন আবশ্যক)'}")
            with col_b2:
                st.markdown(f"**Common Uses (ব্যবহার)**: {info.common_uses}")
                st.markdown(f"**বাংলায় ব্যবহার**: {info.bengali_uses}")
                if info.warning:
                    st.warning(f"⚠️ {info.warning}")

            # Show Alternatives Button
            alts = find_brand_alternatives(info.generic_name)
            if alts and len(alts) > 1:
                st.markdown(f"**Available Bangladeshi Alternatives ({len(alts)-1}):**")
                alt_names = [f"**{a.brand_name}** ({a.manufacturer})" for a in alts if a.brand_name.lower() != info.brand_name.lower()]
                st.markdown(" • ".join(alt_names))


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: Emergency Protocols & Specialized Hospitals
# ═══════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 🚑 Step-by-Step Bangladesh First Aid Protocols")
    for pid, proto in FIRST_AID_PROTOCOLS.items():
        with st.expander(f"🚨 {proto.title_en} ({proto.title_bn})"):
            st.markdown(f"**Summary**: {proto.quick_summary_en}")
            st.markdown(f"**সংক্ষেপ**: {proto.quick_summary_bn}")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("#### ✅ Immediate Actions (কী করবেন)")
                for do in proto.dos_en:
                    st.markdown(f"- {do}")
            with col_p2:
                st.markdown("#### ❌ Critical Don'ts (কী করবেন না)")
                for dont in proto.donts_en:
                    st.markdown(f"- {dont}")

            st.error(f"🚨 **Red Flag Symptoms**: {', '.join(proto.red_flags_en)}")
            st.info(f"📞 **Emergency Contacts**: {', '.join(proto.emergency_contacts)}")

    st.markdown("---")
    st.markdown("### 🏥 Specialized Government Hospitals in Dhaka & Bangladesh")
    hospitals = [
        ("Dhaka Medical College Hospital (DMCH)", "Central emergency, trauma, toxicology & snakebite treatment", "02-55165088 / 01700-700700"),
        ("Sheikh Hasina National Institute of Burn & Plastic Surgery", "Largest specialized burn treatment hospital", "02-41052601"),
        ("ICDDR,B (Cholera Hospital, Mohakhali)", "24/7 world-renowned diarrhea, cholera & dehydration hospital", "02-9827001"),
        ("National Institute of Traumatology (NITOR / পঙ্গু হাসপাতাল)", "Major bone fractures, accidents, limb trauma & orthopedics", "02-9144190"),
        ("National Institute of Cardiovascular Diseases (NICVD / হৃদরোগ ইন্সটিটিউট)", "Emergency acute heart attack & cardiac care", "02-9122560"),
    ]
    for name, desc, contact in hospitals:
        st.markdown(f"**{name}** — *{desc}* | `📞 {contact}`")


# ── Global Medical Advice Disclaimer ───────────────────────────────────

st.markdown("---")
st.caption(
    "⚠️ **Medical Disclaimer (দায়মুক্তি বিবৃতি)**: "
    "This tool provides pharmaceutical information and first-aid triage awareness strictly for educational purposes in Bangladesh. "
    "It is NOT a substitute for formal clinical diagnosis or treatment by a BMDC (Bangladesh Medical & Dental Council) registered physician. "
    "In acute life-threatening emergencies, immediately dial **999** for emergency ambulance services or visit the nearest government hospital."
)
