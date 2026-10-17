"""Conversational Clinical Assistant with Context Memory & Multi-Turn Reasoning.

Features:
- Conversation memory & antecedent resolution ("can I take this with paracetamol?").
- Multi-intent routing: Identification, Dosage, Side Effects, Interactions, BD Alternatives.
- Multi-tier Safety Layer: Emergency (999), Pregnancy/Lactation, Pediatric alerts.
- Strict Grounding: Abstains with 'Not found in official labeling' when unverified.
- Verifiable Citations with official DailyMed package insert links.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from src.answer.engine import AnswerEngine, AnswerResult, evaluate_confidence
from src.knowledge.bangladesh_pharma import (
    BANGLADESH_BRANDS,
    BANGLADESH_EMERGENCY_HOTLINES,
    find_brand_alternatives,
    get_canonical_generic_from_local_query,
    is_emergency_query,
    lookup_bangladesh_brand,
)
from src.firstaid.triage import FIRST_AID_PROTOCOLS, find_first_aid_protocol
from src.retrieval.hybrid import detect_drugs_in_query

logger = logging.getLogger(__name__)


@dataclass
class SafetyAlert:
    alert_type: str  # "EMERGENCY" | "PREGNANCY" | "PEDIATRIC"
    title: str
    message: str
    hotlines: list[str] = field(default_factory=list)


@dataclass
class ChatTurnResponse:
    reply_text: str
    confidence: str = "high"  # "high" | "medium" | "low" | "abstained"
    sources: list[dict[str, str]] = field(default_factory=list)
    safety_alerts: list[SafetyAlert] = field(default_factory=list)
    active_drug: str | None = None
    query_type: str = "general"  # "identify" | "dosage" | "side_effects" | "interactions" | "alternatives" | "first_aid" | "rag"
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)


def build_unrecognized_guidance(query: str, effective_lang: str = "en") -> str:
    """Generate helpful guidance when a medicine name is not recognized or found in official labeling.

    Informs the user that the drug is not recognized, suggests close matches if any,
    asks what symptoms/condition they are trying to treat, and provides official online search links.
    """
    import difflib
    import urllib.parse
    from src.ingest.download import DRUG_LIST

    # Strip conversational prefixes to isolate candidate drug name
    q_clean = re.sub(
        r"^(tell me about|what is the dosage of|what is|what are|info on|information about|how to use|side effects of|can i take|how much|what does|dose of|is)\s+",
        "",
        query.strip(),
        flags=re.IGNORECASE,
    ).strip(" ?.,!\"'")

    # Build known names vocabulary
    all_names: dict[str, str] = {}
    for k, v in BANGLADESH_BRANDS.items():
        all_names[k.lower()] = f"{v.brand_name} ({v.generic_name.title()}) — {v.common_uses}"
        all_names[v.generic_name.lower()] = f"{v.generic_name.title()} (Generic Active Ingredient)"
    for d in DRUG_LIST:
        if d.lower() not in all_names:
            all_names[d.lower()] = f"{d.title()} (Generic Active Ingredient)"

    # Find close matches for the full cleaned phrase or individual words
    words = [w for w in re.split(r"\s+", q_clean) if len(w) >= 3]
    matches: list[tuple[str, str]] = []
    for cand in [q_clean] + words:
        cand_clean = re.sub(r"\d+", "", cand).strip().lower()
        if len(cand_clean) >= 3:
            close = difflib.get_close_matches(cand_clean, list(all_names.keys()), n=3, cutoff=0.55)
            for m in close:
                if m not in [x[0] for x in matches]:
                    matches.append((m, all_names[m]))

    top_matches = matches[:3]
    encoded_display = q_clean if q_clean else query.strip()
    encoded_q = urllib.parse.quote_plus(encoded_display)

    if effective_lang == "bn":
        suggestions_md = ""
        if top_matches:
            sug_items = "\n".join([f"- **{display}**" for _, display in top_matches])
            suggestions_md = f"#### 🔍 আপনি কি নিচের কোনো ওষুধের কথা জানতে চেয়েছেন?\n{sug_items}\n\n"
        else:
            suggestions_md = "অফিসিয়াল তালিকায় এর কাছাকাছি কোনো ওষুধ পাওয়া যায়নি (দয়া করে প্রেসক্রিপশন বা প্যাকেটের বানানটি পুনরায় যাচাই করুন)।\n\n"

        return (
            f"### ⚠️ ওষুধটি অফিসিয়াল তালিকায় পাওয়া যায়নি (Information Not Found in Official Drug Labeling)\n\n"
            f"**\"{encoded_display}\"** কোনো অফিসিয়াল এফডিএ (FDA) অনুমোদিত বা বাংলাদেশ ডিজিডিএ (DGDA) নিবন্ধিত ওষুধ হিসেবে পাওয়া যায়নি।\n\n"
            f"{suggestions_md}"
            f"#### 💬 আপনাকে কীভাবে সাহায্য করতে পারি?\n"
            f"- **আপনার কী সমস্যা বা লক্ষণ হচ্ছে?** (যেমনঃ জ্বর, মাথাব্যথা, পেটব্যথা, গ্যাস্ট্রিক, ইনফেকশন ইত্যাদি — আপনার লক্ষণ জানালে সঠিক নির্দেশনা দেওয়া যাবে)।\n"
            f"- আপনার কাছে ওষুধের পাতা বা প্রেসক্রিপশন থাকলে উপরের **📷 ক্যামেরা অপশন** দিয়ে ছবি আপলোড করতে পারেন।\n\n"
            f"#### 🌐 অনলাইনে ওষুধটি যাচাই করার অফিসিয়াল লিংক:\n"
            f"- 🔍 [ডেইলিমেড (FDA DailyMed Database) এ খুঁজুন](https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query={encoded_q})\n"
            f"- 🔍 [মেডেক্স বাংলাদেশ (Medex BD DGDA Directory) এ খুঁজুন](https://medex.com.bd/search?q={encoded_q})\n"
            f"- 🔍 [মেডলাইন প্লাস (MedlinePlus NIH Drug Information)](https://medlineplus.gov/druginformation.html)\n"
            f"- 📞 **বিনামূল্যে সরকারি ডাক্তার সেবা**: ২৪/৭ কল করুন **১৬২৬৩** (স্বাস্থ্য বাতায়ন) অথবা জরুরি অ্যাম্বুলেন্সের জন্য **৯৯৯**।"
        )
    else:
        suggestions_md = ""
        if top_matches:
            sug_items = "\n".join([f"- **{display}**" for _, display in top_matches])
            suggestions_md = f"#### 🔍 Did you mean one of these medications?\n{sug_items}\n\n"
        else:
            suggestions_md = "No close pharmaceutical matches were found in official directories (please double-check the spelling on your prescription or packaging).\n\n"

        return (
            f"### ⚠️ Medication Not Recognized in Official Registries (Information Not Found in Official Drug Labeling)\n\n"
            f"**\"{encoded_display}\"** is **not a recognized medication** in official FDA-approved package inserts or Directorate General of Drug Administration (DGDA) Bangladesh listings.\n\n"
            f"{suggestions_md}"
            f"#### 💬 How can I help you find what you need?\n"
            f"- **What symptoms or medical condition are you looking to treat?** (e.g., headache, fever, cough, gastric acidity, infection, or body pain? Tell me what you're experiencing and I can point you to verified medical guidance).\n"
            f"- **Have packaging or a prescription?** You can upload a photo of the medicine strip or bottle using the camera expander above for visual OCR identification.\n\n"
            f"#### 🌐 Direct Online Search & Clinical Resources:\n"
            f"- 🔍 [Search DailyMed (Official US FDA Drug Database)](https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query={encoded_q})\n"
            f"- 🔍 [Search Medex Bangladesh (DGDA Medicine Index)](https://medex.com.bd/search?q={encoded_q})\n"
            f"- 🔍 [Search MedlinePlus (U.S. National Library of Medicine)](https://medlineplus.gov/druginformation.html)\n"
            f"- 📞 **24/7 Bangladesh Medical Helpline**: Dial **16263** (Shastho Batayan — free doctor consultation) or **999** for emergency ambulance."
        )


class ConversationalMedicalAssistant:
    """Multi-turn medical conversational assistant with safety guardrails."""

    def __init__(self, answer_engine: AnswerEngine | None = None) -> None:
        self.engine = answer_engine or AnswerEngine()

    def detect_safety_escalations(self, text: str, language: str = "en") -> list[SafetyAlert]:
        """Scan input text for emergencies, pregnancy, or pediatric mentions."""
        alerts: list[SafetyAlert] = []
        t_lower = text.lower()

        # 1. Acute Life-Threatening Emergency
        is_emerg, reason = is_emergency_query(text)
        if is_emerg:
            title = "🚨 EMERGENCY ESCALATION (জরুরি সতর্কতা)" if language == "bn" else "🚨 CRITICAL EMERGENCY ALERT"
            msg = (
                "এই লক্ষণটি অত্যন্ত গুরুতর হতে পারে। অবিলম্বে ৯৯৯ নম্বরে কল করে অ্যাম্বুলেন্স ডাকুন অথবা নিকটস্থ সরকারি মেডিকেল কলেজ হাসপাতালের জরুরি বিভাগে যান।"
                if language == "bn"
                else "This condition requires immediate emergency medical attention. Call 999 immediately for an ambulance or rush to the nearest hospital emergency room."
            )
            alerts.append(SafetyAlert(alert_type="EMERGENCY", title=title, message=msg, hotlines=["999", "16263"]))

        # 2. Pregnancy & Lactation Warning
        preg_patterns = r"\b(pregnant|pregnancy|lactating|breastfeeding|trimester|গর্ভবতী|গর্ভকালীন|দুগ্ধদানকারী|স্তন্যপান)\b"
        if re.search(preg_patterns, t_lower):
            title = "🤰 PREGNANCY & LACTATION SAFETY (গর্ভকালীন সতর্কতা)" if language == "bn" else "🤰 PREGNANCY & LACTATION NOTICE"
            msg = (
                "গর্ভাবস্থায় বা স্তন্যদানকালে যেকোনো ওষুধ সেবনের পূর্বে অবশ্যই একজন রেজিস্টার্ড স্ত্রীরোগ বিশেষজ্ঞ (OB/GYN) চিকিৎসকের পরামর্শ গ্রহণ করুন। অনেক ওষুধ ভ্রূণের জন্য ঝুঁকিপূর্ণ হতে পারে।"
                if language == "bn"
                else "Medication safety during pregnancy or breastfeeding requires personalized clinical evaluation by an OB/GYN physician. Many drugs cross the placenta or pass into breast milk."
            )
            alerts.append(SafetyAlert(alert_type="PREGNANCY", title=title, message=msg, hotlines=["16263"]))

        # 3. Pediatric & Children Warning
        ped_patterns = r"\b(child|children|baby|infant|kid|toddler|pediatric|বাচ্চা|শিশু|নবজাতক|বছর)\b"
        if re.search(ped_patterns, t_lower):
            title = "👶 PEDIATRIC DOSING WARNING (শিশুদের নিরাপত্তা)" if language == "bn" else "👶 PEDIATRIC DOSAGE WARNING"
            msg = (
                "শিশুদের ওষুধের মাত্রা ওজন ও বয়স অনুযায়ী নির্ধারিত হয়। চিকিৎসকের পরামর্শ ছাড়া শিশুদের প্রাপ্তবয়স্কদের ওষুধ বা অ্যাসপিরিন সেবন করাবেন না (রেই সিন্ড্রোমের ঝুঁকি রয়েছে)।"
                if language == "bn"
                else "Pediatric medication dosing is strictly weight-based. Never give adult medications or aspirin to children without a pediatrician's prescription (risk of Reye's syndrome)."
            )
            alerts.append(SafetyAlert(alert_type="PEDIATRIC", title=title, message=msg, hotlines=["16263"]))

        return alerts

    def resolve_conversation_context(
        self,
        current_query: str,
        conversation_history: list[dict[str, Any]],
    ) -> tuple[str, str | None]:
        """Resolve pronouns ('this', 'it', 'the medicine') to the active drug in conversation."""
        q_lower = current_query.lower()
        active_drug = None

        # 1. Check if current query explicitly mentions a drug
        detected = detect_drugs_in_query(current_query)
        bd_match = lookup_bangladesh_brand(current_query)
        if detected:
            active_drug = detected[0]
        elif bd_match:
            active_drug = bd_match.generic_name

        # 2. If current query uses pronouns ("this", "it", "the drug", "can i take this with..."), find antecedent
        has_pronoun = bool(re.search(r"\b(this|it|the medicine|the drug|এটা|এই ওষুধ|ওষুধটি)\b", q_lower))
        if has_pronoun or not active_drug:
            # Look backwards through history for the last referenced drug
            for msg in reversed(conversation_history):
                prev_drug = msg.get("active_drug")
                if prev_drug:
                    active_drug = prev_drug
                    break

        # 3. Synthesize resolved query if antecedent was substituted
        resolved_query = current_query
        if active_drug and has_pronoun:
            # Substitute "this" with the active drug name for retrieval
            resolved_query = re.sub(r"\b(this|it|the medicine|the drug)\b", active_drug, current_query, flags=re.IGNORECASE)

        return resolved_query, active_drug

    def respond(
        self,
        user_message: str,
        image: Image.Image | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        language: str = "en",
    ) -> ChatTurnResponse:
        """Process conversational turn with context memory, safety gating, and strict grounding."""
        history = conversation_history or []
        safety_alerts = self.detect_safety_escalations(user_message, language=language)

        # 1. OCR Extraction if image provided
        extracted_ocr = ""
        if image is not None:
            try:
                import pytesseract
                raw_ocr = pytesseract.image_to_string(image).strip()
                if raw_ocr:
                    extracted_ocr = raw_ocr
            except Exception:
                pass

        combined_input = f"{user_message} {extracted_ocr}".strip()

        # 2. Resolve Antecedent & Context
        resolved_query, active_drug = self.resolve_conversation_context(combined_input, history)
        q_lower = resolved_query.lower()

        # Detect if input is in Bengali
        has_bengali_script = bool(re.search(r"[\u0980-\u09FF]", user_message))
        effective_lang = "bn" if (has_bengali_script or language == "bn") else "en"

        # Check specific clinical sub-intents
        is_dosage_intent = any(w in q_lower for w in ["dosage", "dose", "how much", "how many", "how often", "frequency", "ডোজ", "মাত্রা", "কতটুকু", "mg", "milligram", "খাওয়ার নিয়ম"])
        is_side_effects_intent = any(w in q_lower for w in ["side effect", "adverse", "risk", "harm", "reaction", "contraindication", "danger", "পার্শ্বপ্রতিক্রিয়া", "ক্ষতি", "ঝুঁকি", "ভয়", "সমস্যা"])
        is_interactions_intent = any(w in q_lower for w in ["interaction", "take with", "together with", "with", "mix", "একসাথে", "মিশিয়ে", "খাওয়া যাবে", "খাওয়া যায়"])

        # 3. Intent Detection: Bangladeshi Brand Alternatives
        is_alternative_intent = bool(
            re.search(
                r"(\b(alternative|alternatives|substitute|substitutes|বিকল্প|অন্য ব্র্যান্ড|অন্যান্য ব্র্যান্ড)\b)"
                r"|(\b(other|all|different|more|available|local|list)\s+(bangladeshi\s+)?brands?\b)"
                r"|(\b(list|show|name|tell me|what are)\s+(the\s+|all\s+|other\s+)?(bangladeshi\s+)?brands?\b)"
                r"|(\bbrands?\s+(of|for|sharing|like|in)\b)",
                q_lower,
            )
        )
        if is_alternative_intent:
            target_drug = active_drug
            detected = detect_drugs_in_query(combined_input)
            bd_match = lookup_bangladesh_brand(combined_input)
            canon = get_canonical_generic_from_local_query(combined_input)
            if detected:
                target_drug = detected[0]
            elif bd_match:
                target_drug = bd_match.generic_name
            elif canon:
                target_drug = canon
            elif not target_drug:
                for g in [
                    "paracetamol",
                    "acetaminophen",
                    "omeprazole",
                    "esomeprazole",
                    "pantoprazole",
                    "cetirizine",
                    "fexofenadine",
                    "montelukast",
                    "metformin",
                    "azithromycin",
                    "ciprofloxacin",
                    "amoxicillin",
                    "cefixime",
                ]:
                    if re.search(rf"\b{g}\b", q_lower):
                        target_drug = g
                        break

            if target_drug:
                alternatives = find_brand_alternatives(target_drug)
                if alternatives:
                    target_generic = alternatives[0].generic_name.title()
                    alt_lines = []
                    for alt in alternatives:
                        otc_text = "OTC (প্রেসক্রিপশন ছাড়া)" if alt.otc else "Prescription (প্রেসক্রিপশন আবশ্যক)"
                        alt_lines.append(f"- **{alt.brand_name}** ({alt.strength}, {alt.dosage_form}) — *{alt.manufacturer}* | `{otc_text}`")
                    
                    alt_list_str = "\n".join(alt_lines)
                    if effective_lang == "bn":
                        reply = (
                            f"### 💊 বাংলাদেশে **{target_drug.title()}** ({target_generic})-এর সমমানের ব্র্যান্ডসমূহ:\n\n"
                            f"- **মূল সক্রিয় উপাদান (Active Molecule)**: {target_generic}\n\n"
                            f"এদের সবার মূল উপাদান একই ({target_generic}):\n\n"
                            f"{alt_list_str}\n\n"
                            f"⚠️ *একটি ব্র্যান্ড পরিবর্তন করার পূর্বে আপনার ফার্মাসিস্ট বা চিকিৎসকের সাথে মাত্রা (strength) নিশ্চিত করুন।*"
                        )
                    else:
                        reply = (
                            f"### 💊 Bangladeshi Brand Alternatives for **{target_drug.title()}**:\n\n"
                            f"- **Active Molecule**: {target_generic}\n\n"
                            f"These medications share the identical active pharmaceutical ingredient ({target_generic}):\n\n"
                            f"{alt_list_str}\n\n"
                            f"⚠️ *Always verify the exact strength (e.g. 500 mg vs 650 mg) with a registered pharmacist before substituting.*"
                        )
                    return ChatTurnResponse(
                        reply_text=reply,
                        confidence="high",
                        sources=[{"title": f"DGDA Bangladesh National Formulary ({target_generic})", "url": "https://dgda.gov.bd"}],
                        safety_alerts=safety_alerts,
                        active_drug=target_drug,
                        query_type="alternatives",
                    )

        # 4. Intent Detection: Brand Identification / Packaging Query
        bd_brand = lookup_bangladesh_brand(combined_input)

        logger.info(
            "Query routing debug: %s",
            {
                "lang": effective_lang,
                "entity": active_drug,
                "match": bd_brand.brand_name if bd_brand else None,
                "intent": "alternatives" if is_alternative_intent else ("identify" if bd_brand else "evaluating"),
            },
        )
        is_id_inquiry = any(w in q_lower for w in ["identify", "what is", "generic", "কার ওষুধ", "কী ওষুধ", "কি ওষুধ", "কিসের", "কাজ কি", "কাজ কী", "জেনেরিক", "পরিচয়", "ব্যবহার", "বিস্তারিত", "ঔষধ"]) or extracted_ocr
        is_short_brand_inquiry = False
        if bd_brand:
            word_count = len([w for w in re.split(r"\s+", q_lower) if w])
            if word_count <= 6:
                is_short_brand_inquiry = True

        if bd_brand and (is_id_inquiry or is_short_brand_inquiry or (not is_dosage_intent and not is_side_effects_intent and not is_interactions_intent)):
            active_drug = bd_brand.generic_name
            ocr_note = f"> 📷 *OCR identified text from packaging*: `{extracted_ocr[:90]}`\n\n" if extracted_ocr else ""
            if effective_lang == "bn":
                otc_str = "ওটিসি (প্রেসক্রিপশন ছাড়া পাওয়া যায়)" if bd_brand.otc else "প্রেসক্রিপশন আবশ্যক (ডাক্তারের পরামর্শ অনুযায়ী সেব্য)"
                reply = (
                    f"{ocr_note}"
                    f"### 💊 ওষুধ পরিচিতি: **{bd_brand.brand_name}**\n\n"
                    f"- **মূল উপাদান (Generic)**: {bd_brand.bengali_generic} ({bd_brand.generic_name.title()})\n"
                    f"- **প্রস্তুতকারক**: {bd_brand.manufacturer} (বাংলাদেশ)\n"
                    f"- **মাত্রা ও ফর্ম**: {bd_brand.strength} ({bd_brand.dosage_form})\n"
                    f"- **প্রধান ব্যবহার**: {bd_brand.bengali_uses}\n"
                    f"- **খাওয়ার নিয়ম / সতর্কতা**: {bd_brand.warning or 'প্যাকেজের নির্দেশিকা অথবা ডাক্তারের পরামর্শ মেনে চলুন।'}\n"
                    f"- **শ্রেণী**: {otc_str}\n\n"
                    f"⚠️ *জরুরি স্বাস্থ্য প্রয়োজনে কল করুন: ১৬২৬৩ (স্বাস্থ্য বাতায়ন)*"
                )
            else:
                otc_str = "Over The Counter (OTC)" if bd_brand.otc else "Prescription Only"
                reply = (
                    f"{ocr_note}"
                    f"### 💊 Medicine Identification: **{bd_brand.brand_name}**\n\n"
                    f"- **Active Molecule (Generic)**: {bd_brand.generic_name.title()} ({bd_brand.bengali_generic})\n"
                    f"- **Manufacturer**: {bd_brand.manufacturer} (Bangladesh)\n"
                    f"- **Strength & Form**: {bd_brand.strength} ({bd_brand.dosage_form})\n"
                    f"- **Common Indications**: {bd_brand.common_uses}\n"
                    f"- **Clinical Precaution**: {bd_brand.warning or 'Follow registered physician instructions.'}\n"
                    f"- **Classification**: {otc_str}\n\n"
                    f"⚠️ *For 24/7 National Health Consultation in Bangladesh, dial 16263.*"
                )
            return ChatTurnResponse(
                reply_text=reply,
                confidence="high",
                sources=[{"title": f"DGDA Bangladesh National Formulary ({bd_brand.brand_name})", "url": "https://dgda.gov.bd"}],
                safety_alerts=safety_alerts,
                active_drug=active_drug,
                query_type="identify",
            )

        # 5. First-Aid Trauma Protocol Match
        first_aid = find_first_aid_protocol(combined_input)
        if first_aid:
            if language == "bn":
                dos = "\n".join([f"✅ {d}" for d in first_aid.dos_bn])
                donts = "\n".join([f"❌ {d}" for d in first_aid.donts_bn])
                reds = "\n".join([f"🚨 {r}" for r in first_aid.red_flags_bn])
                reply = (
                    f"### 🚑 জরুরি প্রাথমিক চিকিৎসা: **{first_aid.title_bn}**\n\n"
                    f"**সারসংক্ষেপ**: {first_aid.quick_summary_bn}\n\n"
                    f"#### কী করবেন:\n{dos}\n\n"
                    f"#### যা কখনোই করবেন না:\n{donts}\n\n"
                    f"#### বিপদচিহ্ন (অবিলম্বে হাসপাতালে যান):\n{reds}\n\n"
                    f"📞 **জরুরি যোগাযোগ**: ৯৯৯ (অ্যাম্বুলেন্স) | ১৬২৬৩ (স্বাস্থ্য বাতায়ন)"
                )
            else:
                dos = "\n".join([f"✅ {d}" for d in first_aid.dos_en])
                donts = "\n".join([f"❌ {d}" for d in first_aid.donts_en])
                reds = "\n".join([f"🚨 {r}" for r in first_aid.red_flags_en])
                reply = (
                    f"### 🚑 First-Aid Protocol: **{first_aid.title_en}**\n\n"
                    f"**Summary**: {first_aid.quick_summary_en}\n\n"
                    f"#### Immediate Action Steps:\n{dos}\n\n"
                    f"#### Critical Mistakes to Avoid:\n{donts}\n\n"
                    f"#### Red-Flag Emergency Indicators:\n{reds}\n\n"
                    f"📞 **Bangladesh Emergency Contacts**: 999 (Ambulance) | 16263 (Health Helpline)"
                )
            return ChatTurnResponse(
                reply_text=reply,
                confidence="high",
                sources=[{"title": f"Directorate General of Health Services (DGHS) - {first_aid.title_en}", "url": "https://dghs.gov.bd"}],
                safety_alerts=safety_alerts,
                active_drug=active_drug,
                query_type="first_aid",
            )

        # 6. Specific clinical query classification
        clinical_intent = "rag"
        if is_dosage_intent:
            clinical_intent = "dosage"
        elif is_side_effects_intent:
            clinical_intent = "side_effects"
        elif is_interactions_intent:
            clinical_intent = "interactions"

        # Formulate optimal English clinical query for FDA label retrieval
        clinical_search_query = resolved_query
        if active_drug:
            if clinical_intent == "dosage":
                clinical_search_query = f"{active_drug} dosage and administration recommended dose"
            elif clinical_intent == "side_effects":
                clinical_search_query = f"{active_drug} adverse reactions side effects warnings"
            elif clinical_intent == "interactions":
                other_drugs = [d for d in detect_drugs_in_query(combined_input) if d != active_drug]
                other_str = f" {other_drugs[0]}" if other_drugs else ""
                clinical_search_query = f"{active_drug}{other_str} drug interactions concomitant use"
            elif has_bengali_script:
                clinical_search_query = f"{active_drug} indications and usage"

        # Strict RAG Pipeline for Drug Label Queries
        rag_res: AnswerResult = self.engine.answer_question(clinical_search_query)

        # Strict Grounding Check: If abstained or score is low
        if rag_res.abstained or rag_res.confidence == "low":
            not_found_msg = build_unrecognized_guidance(user_message, effective_lang=effective_lang)
            return ChatTurnResponse(
                reply_text=not_found_msg,
                confidence="abstained",
                sources=[],
                safety_alerts=safety_alerts,
                active_drug=active_drug,
                query_type=f"{clinical_intent}_abstained",
                retrieved_chunks=rag_res.retrieved_chunks,
            )

        # Prepare formatted sources
        formatted_sources = []
        for s in rag_res.sources:
            formatted_sources.append({
                "title": f"{s.drug.title()} — Section: {s.section.title()}",
                "url": s.url or "https://dailymed.nlm.nih.gov",
            })

        if effective_lang == "bn":
            reply_body = (
                f"### 📋 অফিসিয়াল ড্রাগ নির্দেশিকা তথ্য ({active_drug.title() if active_drug else ''}):\n\n"
                f"{rag_res.answer}\n\n"
                f"⚠️ *সঠিক চিকিৎসা ও নির্দেশনার জন্য রেজিস্টার্ড চিকিৎসকের পরামর্শ নিন।* (জরুরি সেবা: ১৬২৬৩)"
            )
        else:
            reply_body = rag_res.answer

        return ChatTurnResponse(
            reply_text=reply_body,
            confidence=rag_res.confidence,
            sources=formatted_sources,
            safety_alerts=safety_alerts,
            active_drug=active_drug,
            query_type=clinical_intent,
            retrieved_chunks=rag_res.retrieved_chunks,
        )
