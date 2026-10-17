"""Hugging Face Spaces Entry Point — Gradio Interface for Bangladesh Medical Assistant.

Features:
- Multimodal Vision (Medicine strips, prescriptions, injuries, burns).
- First-Aid & Trauma Triage (Bone fractures, snakebites, dengue, burns).
- Bangladesh Pharma Directory (Napa, Seclo, Sergel, Alatrol, etc.).
- 100% Free, Zero API Keys, Runs 24/7 on Hugging Face Spaces.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Hugging Face ZeroGPU compatibility: spaces MUST be imported first before torch/transformers
try:
    import spaces
except Exception:
    class spaces:  # type: ignore
        @staticmethod
        def GPU(fn=None, *args, **kwargs):
            if fn is None:
                return lambda f: f
            if callable(fn):
                return fn
            return lambda f: f

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import gradio as gr

# Hotfix: gradio_client on HF ZeroGPU crashes when additionalProperties is a
# boolean (True) instead of a dict — "if 'const' in schema" raises TypeError.
try:
    import gradio_client.utils as _gc_utils

    _orig_json_schema = getattr(_gc_utils, "_json_schema_to_python_type", None)
    if _orig_json_schema:
        def _patched_json_schema(schema, defs=None):
            if not isinstance(schema, dict):
                return "Any"
            return _orig_json_schema(schema, defs)
        _gc_utils._json_schema_to_python_type = _patched_json_schema

    _orig_get_type = getattr(_gc_utils, "get_type", None)
    if _orig_get_type:
        def _patched_get_type(schema):
            if not isinstance(schema, dict):
                return "Any"
            return _orig_get_type(schema)
        _gc_utils.get_type = _patched_get_type
except Exception:
    pass

from src.answer.engine import AnswerEngine
from src.firstaid.triage import FIRST_AID_PROTOCOLS, find_first_aid_protocol
from src.knowledge.bangladesh_pharma import (
    BANGLADESH_BRANDS,
    BANGLADESH_EMERGENCY_HOTLINES,
    lookup_bangladesh_brand,
)
from src.llm.multimodal import MultimodalMedicalAssistant
from src.retrieval.hybrid import detect_drugs_in_query

# Cached assistants
multimodal_assistant = MultimodalMedicalAssistant()
answer_engine = None


def get_answer_engine():
    global answer_engine
    if answer_engine is None:
        answer_engine = AnswerEngine()
    return answer_engine


@spaces.GPU
def consult_assistant(
    question: str,
    image: Image.Image | None,
    language: str,
) -> str:
    """Process medical question or image and return structured clinical advice."""
    clean_q = (question or "").strip()
    is_bengali = "বাংলা" in language or language == "bn"
    lang_code = "bn" if is_bengali else "en"

    if not clean_q and image is None:
        return (
            "⚠️ Please enter a medical question or upload an image."
            if not is_bengali
            else "⚠️ দয়া করে একটি চিকিৎসা বা প্রাথমিক চিকিৎসার প্রশ্ন লিখুন অথবা ছবি আপলোড করুন।"
        )

    # Route 1: Multimodal image / BD Brand / First Aid / Bengali / General Health
    bd_match = lookup_bangladesh_brand(clean_q)
    firstaid_match = find_first_aid_protocol(clean_q)
    fda_drugs = detect_drugs_in_query(clean_q)

    if image is not None or bd_match or firstaid_match or is_bengali or not fda_drugs:
        analysis = multimodal_assistant.analyze(
            prompt=clean_q,
            image=image,
            language=lang_code,
        )
        output = []
        if analysis.emergency_level == "EMERGENCY":
            output.append(
                "### 🚨 CRITICAL EMERGENCY ALERT / জরুরি সতর্কতা\n"
                "> **Immediate Emergency Care Required.** Dial **999** for an ambulance or rush to the nearest hospital."
            )
        elif analysis.emergency_level == "WARNING":
            output.append(
                "### ⚠️ CLINICAL WARNING / সতর্কতা\n"
                "> Follow safe first-aid steps carefully. If symptoms persist or worsen, visit a registered BMDC doctor."
            )
        output.append(analysis.text)
        return "\n\n".join(output)

    # Route 2: FDA Drug Label Hybrid RAG
    engine = get_answer_engine()
    res = engine.answer_question(clean_q)
    output = [f"### 📋 Verified Clinical Answer\n\n> {res.answer}"]

    if hasattr(res, "retrieved_chunks") and res.retrieved_chunks:
        output.append("#### 🔬 Retrieved FDA Context Passages:")
        for idx, ch in enumerate(res.retrieved_chunks[:3], 1):
            output.append(
                f"**[{idx}] {ch['drug'].title()} ({ch['section']})** — *Similarity: {ch.get('similarity', 0.0)}*\n"
                f"> {ch['text'][:300]}..."
            )

    return "\n\n".join(output)


def search_medicine(brand_query: str) -> str:
    """Search Bangladesh medicine brand directory."""
    if not brand_query or not brand_query.strip():
        return "Type a brand name (e.g. Napa, Seclo, Sergel, Alatrol, Zimax) to see details."

    term = brand_query.strip().lower()
    matches = [
        (k, v) for k, v in BANGLADESH_BRANDS.items()
        if term in k.lower() or term in v.generic_name.lower() or term in v.bengali_generic
    ]
    if not matches:
        return f"No direct match found for '{brand_query}'. Check spelling or search by generic name."

    lines = [f"### 💊 Search Results ({len(matches)} found):\n"]
    for brand, info in matches[:10]:
        lines.append(
            f"- **{info.brand_name}** ({info.strength})\n"
            f"  - **Generic**: {info.generic_name.title()} ({info.bengali_generic})\n"
            f"  - **Manufacturer**: {info.manufacturer}\n"
            f"  - **Uses**: {info.common_uses}\n"
            f"  - **Warning**: {info.warning or 'Follow doctor prescription.'}\n"
        )
    return "\n".join(lines)


# ── Gradio UI Construction ─────────────────────────────────────────────

custom_css = """
.gradio-container { max-width: 950px !important; margin: auto; }
.emergency-strip {
    background: linear-gradient(90deg, #DC2626 0%, #B91C1C 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
}
"""

with gr.Blocks(title="Bangladesh Medical & First-Aid Assistant", css=custom_css) as demo:
    gr.HTML(
        """
        <div class="emergency-strip">
            🚨 <strong>Bangladesh Emergency Hotlines:</strong> Ambulance/Police: <strong>999</strong> | Health Doctor: <strong>16263</strong> | Poison: <strong>01700-700700</strong>
        </div>
        <h1 style="text-align: center; margin-bottom: 4px;">🇧🇩 Bangladesh Medical, First-Aid & Drug Assistant</h1>
        <p style="text-align: center; color: #64748B; margin-bottom: 20px;">
            AI First-Aid Guidance & Multimodal Vision for Medicine Packaging, Prescriptions, Burns, Dengue, and Bone Fractures.
        </p>
        """
    )

    with gr.Tabs():
        # TAB 1: Multimodal Consultation
        with gr.Tab("💬 First Aid & Multimodal Assistant"):
            with gr.Row():
                with gr.Column(scale=3):
                    q_input = gr.Textbox(
                        label="Enter your medical or first-aid question:",
                        placeholder="e.g. How to fix a broken bone? Can I take Napa for dengue? How to treat hot water burn?",
                        lines=4,
                    )
                    lang_radio = gr.Radio(
                        choices=["English", "বাংলা (Bengali)"],
                        value="English",
                        label="Response Language / ভাষা:",
                        interactive=True,
                    )

                    # Quick topic buttons
                    gr.Markdown("**Quick Topics / দ্রুত বিষয়:**")
                    with gr.Row():
                        b_fracture = gr.Button("Broken Bone / হাড় ভাঙা", size="sm")
                        b_dengue = gr.Button("Dengue Care / ডেঙ্গু", size="sm")
                        b_burn = gr.Button("Burns First Aid / পোড়া", size="sm")
                        b_snake = gr.Button("Snakebite / সাপের কামড়", size="sm")

                with gr.Column(scale=2):
                    img_input = gr.Image(
                        label="📷 Upload Medical Image (Optional):",
                        type="pil",
                    )
                    submit_btn = gr.Button("🚀 Get Instant Guidance", variant="primary", size="lg")

            ans_output = gr.Markdown(label="Clinical Guidance & Analysis")

            # Button Actions
            submit_btn.click(
                fn=consult_assistant,
                inputs=[q_input, img_input, lang_radio],
                outputs=ans_output,
            )
            q_input.submit(
                fn=consult_assistant,
                inputs=[q_input, img_input, lang_radio],
                outputs=ans_output,
            )
            b_fracture.click(lambda: "How to fix a broken bone or fracture? Immediate first aid", outputs=q_input)
            b_dengue.click(lambda: "What is the first aid and medicine for Dengue fever?", outputs=q_input)
            b_burn.click(lambda: "What should I immediately do for hot water burn?", outputs=q_input)
            b_snake.click(lambda: "First aid steps for snakebite in Bangladesh", outputs=q_input)

        # TAB 2: Medicine Directory
        with gr.Tab("💊 Bangladesh Medicine Directory"):
            med_query = gr.Textbox(
                label="Search common medicines in Bangladesh:",
                placeholder="Type brand name (e.g. Napa, Seclo, Sergel, Alatrol, Zimax, Monas)...",
            )
            search_btn = gr.Button("🔍 Search Brand", variant="secondary")
            med_output = gr.Markdown()

            search_btn.click(fn=search_medicine, inputs=med_query, outputs=med_output)
            med_query.submit(fn=search_medicine, inputs=med_query, outputs=med_output)

        # TAB 3: Emergency & Hospitals
        with gr.Tab("🚑 Emergency Hotlines & Hospitals"):
            gr.Markdown(
                """
                ### 🚨 National Emergency Contacts (বাংলাদেশ জরুরি নম্বর):
                - **999**: National Emergency (Ambulance, Police, Fire Service)
                - **16263**: Shastho Batayan (স্বাস্থ্য বাতায়ন - 24/7 Free Doctor Consultation)
                - **10655**: IEDCR Disease Control & Epidemic Helpline
                - **01700-700700**: Dhaka Medical College Hospital Poison Information Center

                ### 🏥 Specialized Emergency Hospitals in Dhaka:
                - **NITOR (পঙ্গu হাসপাতাল)**: Sher-e-Bangla Nagar, Dhaka | Phone: `02-9144190` *(National Orthopedic & Bone Trauma)*
                - **Sheikh Hasina National Institute of Burn & Plastic Surgery**: Chankharpul, Dhaka | Phone: `02-223381666`
                - **ICDDR,B Diarrheal Hospital**: Mohakhali, Dhaka | Phone: `02-9827001` *(24/7 Cholera & Dehydration)*
                - **Dhaka Medical College Hospital (DMCH)**: Central Emergency | Phone: `02-55165088`
                """
            )

    gr.Markdown(
        """
        ---
        <small>⚠️ **Medical Disclaimer**: This application is strictly an educational reference and triage guide for Bangladesh. It does NOT replace a BMDC-registered physician. For severe emergencies, call 999 immediately.</small>
        """
    )

if __name__ == "__main__":
    demo.launch(share=True, ssr_mode=False)
