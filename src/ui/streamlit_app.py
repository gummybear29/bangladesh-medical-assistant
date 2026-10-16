"""Bangladesh Medical, First-Aid & Drug Assistant (সহকারী).

Features:
- Multimodal Image Upload: Analyze medicine strips, prescriptions, wounds, burns, and rashes.
- Bangladesh First-Aid Triage Protocols: Dengue fever, burns, snakebites, ORS, bleeding, choking.
- Bangladesh Pharma Directory: Search Napa, Ace, Seclo, Sergel, Pantonix, Alatrol, Zimax, etc.
- 24/7 Bangladesh Emergency Hotlines: 999, 16263 (Shastho Batayan), DMC Poison Center.
- Bilingual support: 🇬🇧 English and 🇧🇩 বাংলা.
- In-memory cached inference with sub-second response times.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Add project root to sys.path so app can run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import requests
import streamlit as st

from src.config import DISCLAIMER_TEXT, GEMINI_API_KEY
from src.firstaid.triage import FIRST_AID_PROTOCOLS, find_first_aid_protocol
from src.knowledge.bangladesh_pharma import (
    BANGLADESH_BRANDS,
    BANGLADESH_EMERGENCY_HOTLINES,
    lookup_bangladesh_brand,
)
from src.llm.multimodal import MultimodalMedicalAssistant

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")

st.set_page_config(
    page_title="Bangladesh Medical & First Aid Assistant",
    page_icon="🇧🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── In-Memory Caching for Sub-Second Responses ────────────────────────


@st.cache_resource
def get_cached_answer_engine():
    from src.answer.engine import AnswerEngine
    return AnswerEngine()


@st.cache_resource
def get_cached_multimodal_assistant():
    return MultimodalMedicalAssistant()


# ── Custom CSS ─────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .emergency-banner {
        background: linear-gradient(90deg, #DC2626 0%, #B91C1C 100%);
        color: white;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 6px -1px rgba(220, 38, 38, 0.2);
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .card-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .badge-high {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-medium {
        background-color: #FEF9C3;
        color: #A16207;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .review-notice {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #1E40AF;
        margin-top: 12px;
    }
    .hotline-chip {
        background-color: white;
        color: #DC2626;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        margin: 0 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    if GEMINI_API_KEY:
        st.success("🟢 Live Gemini 2.0 Flash Multimodal Active")
    else:
        st.info("ℹ️ Local Offline / Demo Engine Active (Sub-second inference)")


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
    st.markdown('<div class="main-title">🇧🇩 বাংলাদেশ প্রাথমিক চিকিৎসা ও ওষুধ নির্দেশিকা</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">ওষুধের ছবি, প্রেসক্রিপশন বা আঘাতের ছবি আপলোড করুন এবং তাৎক্ষণিক নিরাপদ প্রাথমিক চিকিৎসা পরামর্শ নিন।</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="main-title">🇧🇩 Bangladesh Medical & First Aid Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI First-Aid Guidance & Multimodal Vision for Medicine Packaging, Prescriptions, Burns, Dengue, and Injuries.</div>', unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "💬 First Aid & Multimodal Assistant" if not is_bengali else "💬 প্রাথমিক চিকিৎসা ও ছবি বিশ্লেষণ",
    "💊 Bangladesh Medicine Directory" if not is_bengali else "💊 স্থানীয় ওষুধের তালিকা (ব্র্যান্ড)",
    "🚑 Emergency Protocols & Hospitals" if not is_bengali else "🚑 জরুরি প্রোটোকল ও হাসপাতাল",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Multimodal Assistant (Text + Image)
# ═══════════════════════════════════════════════════════════════════════

with tab1:
    # Quick Chips
    st.markdown("**Quick Topics / দ্রুত বিষয় নির্বাচন করুন:**")
    q_cols = st.columns(6)
    quick_queries = [
        ("Dengue Care", "What is the first aid and medicine for Dengue fever?", "ডেঙ্গু জ্বর ও নাপা"),
        ("Burns First Aid", "What should I immediately do for hot water burn?", "আগুনে পোড়ার প্রাথমিক চিকিৎসা"),
        ("Snakebite Protocol", "First aid steps for snakebite in Bangladesh", "সাপের কামড় দিলে কী করবেন?"),
        ("ORS / Diarrhea", "How to prepare Orsaline-N properly for diarrhea?", "খাবার স্যালাইন বানানোর সঠিক নিয়ম"),
        ("Napa Dosage", "What is the maximum daily dose of Napa (paracetamol)?", "নাপা ৫০০ খাওয়ার সঠিক ডোজ কত?"),
        ("Cut & Bleeding", "How to stop heavy bleeding from a cut wound?", "রক্তপাত বন্ধ করার প্রাথমিক চিকিৎসা"),
    ]

    selected_query = ""
    for idx, (label, en_q, bn_q) in enumerate(quick_queries):
        btn_text = bn_q if is_bengali else label
        if q_cols[idx].button(btn_text, use_container_width=True, key=f"quick_btn_{idx}"):
            selected_query = bn_q if is_bengali else en_q

    # Input Fields
    col_input, col_media = st.columns([1.6, 1.0])

    with col_input:
        default_val = selected_query or st.session_state.get("active_query", "")
        user_query = st.text_area(
            "Enter your medical or first-aid question:" if not is_bengali else "আপনার প্রশ্ন বা ওষুধের নাম লিখুন:",
            value=default_val,
            placeholder="e.g. Can I take Napa Extra for dengue? Or: How to treat a burn wound?",
            height=120,
        )

    with col_media:
        st.markdown("**📷 Upload Image (Optional / ঐচ্ছিক):**")
        uploaded_file = st.file_uploader(
            "Medicine strip, prescription, or injury image:",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload a photo of a medicine strip, doctor prescription, burn, or wound.",
        )
        pil_image = None
        if uploaded_file is not None:
            try:
                pil_image = Image.open(uploaded_file)
                st.image(pil_image, caption="Uploaded Medical Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")

    submit_button = st.button(
        "🚀 Get Instant Guidance" if not is_bengali else "🚀 তাৎক্ষণিক পরামর্শ নিন",
        type="primary",
        use_container_width=True,
    )

    if submit_button:
        active_text = user_query.strip()
        if not active_text and pil_image is None:
            st.warning("Please enter a question or upload an image.")
        else:
            with st.spinner("Analyzing clinical data & first-aid guidelines..."):
                # If an image is uploaded or the query is First-Aid/Bangladesh specific:
                multimodal_assistant = get_cached_multimodal_assistant()
                has_image = pil_image is not None
                bd_match = lookup_bangladesh_brand(active_text)
                firstaid_match = find_first_aid_protocol(active_text)

                # Route 1: Multimodal or First Aid or BD Brand Query
                if has_image or bd_match or firstaid_match or is_bengali:
                    analysis = multimodal_assistant.analyze(
                        prompt=active_text,
                        image=pil_image,
                        language=lang_code,
                    )

                    st.markdown("### 📋 Analysis & Guidance")
                    if analysis.emergency_level == "EMERGENCY":
                        st.markdown(
                            """
                            <div class="review-notice" style="background-color: #FEE2E2; border-color: #DC2626; color: #991B1B;">
                                🚨 <strong>CRITICAL EMERGENCY ALERT</strong>: This condition requires immediate professional emergency care. Call <strong>999</strong> for an ambulance or rush to the nearest hospital.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown(analysis.text)

                # Route 2: Standard Drug Label RAG Query
                else:
                    # Query cached in-process AnswerEngine
                    engine = get_cached_answer_engine()
                    res = engine.answer_question(active_text)

                    st.markdown("### 📋 Clinical Answer")

                    # Confidence Badge
                    if res.confidence == "high":
                        badge_html = '<span class="badge-high">🟢 High Confidence</span>'
                    elif res.confidence == "medium":
                        badge_html = '<span class="badge-medium">🟡 Medium Confidence</span>'
                    else:
                        badge_html = '<span class="badge-low">🔴 Low Confidence (Abstained)</span>'

                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.markdown(f"> {res.answer}")

                    if res.sources:
                        with st.expander(f"📚 Verified Sources & Package Inserts ({len(res.sources)})"):
                            for s in res.sources:
                                st.markdown(f"- **{s.drug.title()}** | Section: `{s.section}` — [View DailyMed Link]({s.url})")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: Bangladesh Brand & Medicine Directory
# ═══════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 🔍 Search Common Bangladeshi Medicines (জনপ্রিয় ওষুধ অনুসন্ধান)")
    search_term = st.text_input("Type a brand name (e.g. Napa, Seclo, Sergel, Alatrol, Zimax, Monas):", "")

    filtered_brands = BANGLADESH_BRANDS.items()
    if search_term.strip():
        term = search_term.lower()
        filtered_brands = [
            (k, v) for k, v in BANGLADESH_BRANDS.items()
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
    "This tool provides first-aid reference and pharmaceutical information strictly for educational and triage awareness in Bangladesh. "
    "It is NOT a substitute for formal clinical diagnosis or treatment by a BMDC (Bangladesh Medical & Dental Council) registered physician. "
    "In acute life-threatening emergencies, immediately dial **999** for emergency ambulance services or visit the nearest government health complex."
)
