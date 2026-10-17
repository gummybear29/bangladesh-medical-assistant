"""Multimodal Vision & Clinical Intelligence Module.

Analyzes medical images (medicine strips/bottles, doctor prescriptions,
wounds, burns, and rashes) combined with text queries. Works live via Google
Gemini 2.0 Flash or in offline demo mode with mock visual feature extraction.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.knowledge.bangladesh_pharma import (
    BANGLADESH_EMERGENCY_HOTLINES,
    lookup_bangladesh_brand,
)

logger = logging.getLogger(__name__)

MULTIMODAL_SYSTEM_PROMPT = """You are a specialized medical assistant and first-aid expert serving patients and caregivers in Bangladesh.
Your task is to analyze medical queries and/or uploaded images (medicine strips, prescriptions, injuries, burns, or rashes) to provide accurate, safe, evidence-based guidance.

RULES:
1. For MEDICINE STRIPS / PRESCRIPTIONS:
   - Identify the brand name, active generic ingredient, strength (e.g. 500 mg), and manufacturer (e.g. Square, Beximco, Incepta) if visible.
   - Explain what the medication is commonly used for in plain language.
   - State standard precautions, maximum safe dose, and contraindications.
   - For prescriptions, explain the doctor's frequency/dosage instructions clearly.
2. For INJURIES / BURNS / WOUNDS:
   - Provide immediate, structured FIRST-AID steps (Do's and Don'ts).
   - Explicitly warn against dangerous traditional home remedies (no toothpaste/butter on burns, no tight tourniquets or blade-cutting on snakebites).
   - Identify RED-FLAG symptoms requiring an immediate hospital/emergency visit.
3. LANGUAGE:
   - If the user asks in Bengali (বাংলা) or the query is in Bangladesh context, provide your answer with clear Bengali translations and terms.
4. SAFETY:
   - Emphasize that AI image analysis is informational and does NOT replace a registered BMDC (Bangladesh Medical & Dental Council) physician.
   - In life-threatening emergencies, urge calling 999 or 16263 immediately.
"""


@dataclass
class MultimodalAnalysisResult:
    text: str
    identified_items: list[str] = field(default_factory=list)
    first_aid_steps: list[str] = field(default_factory=list)
    warning: str = ""
    emergency_level: str = "STANDARD"  # "EMERGENCY" | "WARNING" | "STANDARD"
    is_live_gemini: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultimodalMedicalAssistant:
    """Orchestrates image + text clinical analysis with Gemini 2.0 Flash."""

    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model_name: str = GEMINI_MODEL,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._model = None

    def _init_model(self) -> None:
        if self._model is not None or not self.api_key:
            return
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=MULTIMODAL_SYSTEM_PROMPT,
        )

    def analyze(
        self,
        prompt: str,
        image: Image.Image | bytes | None = None,
        language: str = "en",
    ) -> MultimodalAnalysisResult:
        """Process text prompt and optional image, returning structured clinical guidance."""
        clean_prompt = prompt.strip() if prompt else "Analyze this medical image and provide first-aid / medicine identification."

        # Convert bytes to PIL Image if necessary
        pil_img: Image.Image | None = None
        if isinstance(image, bytes):
            try:
                pil_img = Image.open(io.BytesIO(image))
            except Exception as exc:
                logger.warning("Could not parse image bytes: %s", exc)
        elif isinstance(image, Image.Image):
            pil_img = image

        # 1. If real Gemini API key is configured, call Gemini Multimodal
        if self.api_key:
            try:
                self._init_model()
                content_payload: list[Any] = [clean_prompt]
                if pil_img is not None:
                    content_payload.append(pil_img)

                resp = self._model.generate_content(content_payload)
                resp_text = resp.text.strip() if resp and resp.text else "Unable to generate analysis."
                
                # Check emergency keywords
                emergency_level = "STANDARD"
                if any(w in resp_text.lower() for w in ["999", "emergency", "immediate hospital", "life-threatening", "জরুরি"]):
                    emergency_level = "EMERGENCY"

                return MultimodalAnalysisResult(
                    text=resp_text,
                    emergency_level=emergency_level,
                    is_live_gemini=True,
                )
            except Exception as exc:
                logger.error("Live Gemini Vision call failed: %s. Falling back to offline engine.", exc)

        # 2. Offline / Demo Mode Clinical Engine
        return self._generate_offline_analysis(clean_prompt, pil_img, language=language)

    def _generate_offline_analysis(
        self,
        prompt: str,
        image: Image.Image | None,
        language: str = "en",
    ) -> MultimodalAnalysisResult:
        """Heuristic-grounded clinical analysis for offline / demo mode."""
        prompt_lower = prompt.lower()
        identified: list[str] = []
        steps: list[str] = []
        warning = ""
        emergency_level = "STANDARD"

        # 1. If an image is provided, run lightweight OCR to extract printed medicine packaging text
        extracted_text = ""
        if image is not None:
            w, h = image.size
            img_desc = f"Image dimensions: {w}x{h} px"
            identified.append("Uploaded Medical Image")
            try:
                import pytesseract
                raw_ocr = pytesseract.image_to_string(image).strip()
                if raw_ocr:
                    extracted_text = raw_ocr
                    identified.append(f"Extracted Text: {extracted_text[:60]}...")
            except Exception as ocr_err:
                logger.info("OCR engine not available or no text found: %s", ocr_err)
        else:
            img_desc = "Text-only query"

        # Combine user prompt with extracted image text for clinical matching
        search_corpus = f"{prompt} {extracted_text}".strip()

        # Check for Bangladesh brands in prompt or extracted image text
        bd_info = lookup_bangladesh_brand(search_corpus)

        if bd_info:
            identified.append(f"{bd_info.brand_name} ({bd_info.generic_name}) - {bd_info.manufacturer}")
            warning = bd_info.warning or "Follow package instructions."
            ocr_banner = f"> 📷 **Image OCR Detected**: *\"{extracted_text[:120]}\"*\n\n" if extracted_text else ""
            if language == "bn":
                text = (
                    f"{ocr_banner}"
                    f"### 💊 ওষুধ পরিচিতি: {bd_info.brand_name}\n\n"

                    f"- **মূল উপাদান (Generic)**: {bd_info.bengali_generic} ({bd_info.generic_name.title()})\n"
                    f"- **প্রস্তুতকারক**: {bd_info.manufacturer}\n"
                    f"- **মাত্রা (Strength)**: {bd_info.strength}\n"
                    f"- **সাধারণ ব্যবহার**: {bd_info.bengali_uses}\n"
                    f"- **সতর্কতা**: {bd_info.warning or 'চিকিৎসকের পরামর্শ অনুযায়ী সেবন করুন।'}\n\n"
                    f"⚠️ *জরুরি স্বাস্থ্য প্রয়োজনে কল করুন: ১৬২৬৩ (স্বাস্থ্য বাতায়ন)*"
                )
            else:
                text = (
                    f"### 💊 Medicine Identification: {bd_info.brand_name}\n\n"
                    f"- **Active Ingredient**: {bd_info.generic_name.title()}\n"
                    f"- **Manufacturer**: {bd_info.manufacturer} (Bangladesh)\n"
                    f"- **Strength**: {bd_info.strength}\n"
                    f"- **Common Uses**: {bd_info.common_uses}\n"
                    f"- **Clinical Caution**: {bd_info.warning or 'Follow registered physician guidance.'}\n\n"
                    f"⚠️ *For 24/7 National Health Consultation in Bangladesh, call: 16263*"
                )
            return MultimodalAnalysisResult(
                text=text,
                identified_items=identified,
                warning=warning,
                emergency_level="STANDARD",
                is_live_gemini=False,
            )

        # Check first-aid categories
        from src.firstaid.triage import find_first_aid_protocol

        protocol = find_first_aid_protocol(search_corpus)

        if protocol:
            identified.append(protocol.title_en)
            emergency_level = "EMERGENCY" if protocol.urgency_level == "EMERGENCY_CRITICAL" else "WARNING"
            
            if language == "bn":
                dos_list = "\n".join([f"✅ {d}" for d in protocol.dos_bn])
                donts_list = "\n".join([f"❌ {d}" for d in protocol.donts_bn])
                red_list = "\n".join([f"🚨 {r}" for r in protocol.red_flags_bn])
                text = (
                    f"### 🚑 প্রাথমিক চিকিৎসা গাইড: {protocol.title_bn}\n\n"
                    f"**সংক্ষেপ**: {protocol.quick_summary_bn}\n\n"
                    f"#### কী করবেন (Do's):\n{dos_list}\n\n"
                    f"#### কী করবেন না (Don'ts):\n{donts_list}\n\n"
                    f"#### অবিলম্বে হাসপাতালে যাওয়ার লক্ষণ (Red Flags):\n{red_list}\n\n"
                    f"📞 **জরুরি যোগাযোগ**: ৯৯৯ (অ্যাম্বুলেন্স) | ১৬২৬৩ (স্বাস্থ্য বাতায়ন)"
                )
            else:
                dos_list = "\n".join([f"✅ {d}" for d in protocol.dos_en])
                donts_list = "\n".join([f"❌ {d}" for d in protocol.donts_en])
                red_list = "\n".join([f"🚨 {r}" for r in protocol.red_flags_en])
                text = (
                    f"### 🚑 First Aid Protocol: {protocol.title_en}\n\n"
                    f"**Summary**: {protocol.quick_summary_en}\n\n"
                    f"#### Immediate Steps (Do's):\n{dos_list}\n\n"
                    f"#### Critical Mistakes to Avoid (Don'ts):\n{donts_list}\n\n"
                    f"#### Red-Flag Emergency Symptoms:\n{red_list}\n\n"
                    f"📞 **Bangladesh Emergency**: Call 999 for Ambulance | Call 16263 for 24/7 Doctor Helpline"
                )

            return MultimodalAnalysisResult(
                text=text,
                identified_items=identified,
                first_aid_steps=protocol.dos_en,
                warning=protocol.quick_summary_en,
                emergency_level=emergency_level,
                is_live_gemini=False,
            )

        # General Image/Text Consultation
        if image is not None:
            text = (
                f"### 📷 Image Inspection & Clinical Review\n\n"
                f"- **Image Received**: Successfully processed ({img_desc}).\n"
                f"- **Preliminary Observation**: Visual triage for wounds, medicine packaging, or skin conditions.\n"
                f"- **Safe First-Aid Advice**:\n"
                f"  1. Clean wounds gently with sterile saline or clean running water.\n"
                f"  2. Apply dry, clean dressing without applying chemical irritants.\n"
                f"  3. Monitor for redness, severe swelling, throbbing pain, or fever (infection signs).\n"
                f"  4. If checking a medicine strip, always confirm the expiration date and active generic name.\n\n"
                f"⚠️ *Notice: Image analysis by AI is an initial reference and cannot replace in-person physical clinical examination. For acute symptoms, visit your nearest Upazila Health Complex or Medical College Hospital.*"
            )
        else:
            if language == "bn":
                text = (
                    f"### 🩺 চিকিৎসা পরামর্শ ও প্রাথমিক তথ্য\n\n"
                    f"**আপনার প্রশ্ন**: *{prompt}*\n\n"
                    f"- **জ্বর ও সাধারণ ব্যথায়**: প্যারাসিটামল (নাপা / এইস ৫০০ মিলিগ্রাম) সাধারণত ১টি করে দিনে ২-৩ বার ভরা পেটে সেবন করা হয়। সর্বোচ্চ দৈনিক মাত্রা ৪০০০ মিলিগ্রাম (৮টি ট্যাবলেট)।\n"
                    f"- **গ্যাসের সমস্যা বা বুক জ্বালায়**: ওমিপ্রাজল (সেকলো ২০) বা এসোমিপ্রাজল (সারজেল ২০) সকালে খালি পেটে সেবন করা হয়।\n"
                    f"- **পাতলা পায়খানা বা ডায়রিয়ায়**: প্রতিবার পাতলা পায়খানার পর ১ গ্লাস খাবার স্যালাইন (ওরস্যালাইন-এন) নিয়ম মেনে পান করুন।\n\n"
                    f"⚠️ *সতর্কতা: এটি প্রাথমিক তথ্যমূলক নির্দেশিকা। কোনো ওষুধ শুরুর পূর্বে রেজিস্টার্ড চিকিৎসকের পরামর্শ নিন। জরুরি প্রয়োজনে ১৬২৬৩ নম্বরে কল করুন।*"
                )
            else:
                text = (
                    f"### 🩺 Medical Guidance & Clinical Information\n\n"
                    f"**Question**: *{prompt}*\n\n"
                    f"- **Fever & Pain Relief**: Paracetamol (Napa / Ace 500 mg) is the standard first-line choice (1 tablet every 6–8 hours after food, max 4,000 mg/day).\n"
                    f"- **Gastric Acidity & GERD**: Omeprazole (Seclo 20 mg) or Esomeprazole (Sergel 20 mg) taken 30 minutes before breakfast.\n"
                    f"- **Dehydration / Loose Stools**: Prepare Orsaline-N (1 sachet in 500 mL water) and drink after each loose stool.\n\n"
                    f"⚠️ *Notice: This guidance is for educational reference. Consult a BMDC-registered physician for tailored treatment. Call 16263 for government health consultation.*"
                )


        return MultimodalAnalysisResult(
            text=text,
            identified_items=identified,
            emergency_level=emergency_level,
            is_live_gemini=False,
        )
