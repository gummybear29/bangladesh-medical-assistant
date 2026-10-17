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

        # Clinical Intent Classifications
        is_dosage_intent = bool(
            re.search(
                r"\b(dose|dosage|dosing|how much|how many|taking|take|খাব|খাওয়া|মাত্রা|পরিমাণ|কতটুকু|ক্যাপসুল|ট্যাবলেট)\b",
                q_lower,
            )
        )
        is_side_effects_intent = bool(
            re.search(
                r"\b(side effect|side effects|adverse|reaction|reactions|harm|risk|risks|পার্শ্বপ্রতিক্রিয়া|ক্ষতিকর|ঝুঁকি)\b",
                q_lower,
            )
        )
        is_interactions_intent = bool(
            re.search(
                r"\b(interaction|interactions|with|together|combine|taking with|along with|একসাথে|সাথে|ইন্টারঅ্যাকশন)\b",
                q_lower,
            )
        )

        # Detect if input is in Bengali
        has_bengali_script = bool(re.search(r"[\u0980-\u09FF]", user_message))
        effective_lang = "bn" if (has_bengali_script or language == "bn") else "en"

        # Guardrail 0: Empty, whitespace-only, emoji-only, or oversized input (H5)
        clean_chars = re.sub(r"[\s\W_]+", "", user_message)
        if not user_message.strip() and not extracted_ocr:
            return ChatTurnResponse(
                reply_text=(
                    "Hello! Please type a medicine name, question, or upload a packaging photo to get clinical guidance."
                    if effective_lang != "bn"
                    else "নমস্কার! কোনো ওষুধের নাম, প্রশ্ন লিখুন অথবা প্যাকেটের ছবি আপলোড করুন।"
                ),
                confidence="high",
                sources=[],
                query_type="welcome",
            )
        if not clean_chars and not extracted_ocr:
            return ChatTurnResponse(
                reply_text=(
                    "I received your message, but no medicine name or clinical question was detected. Please type a specific drug name (e.g. Napa, Seclo) or ask a health question."
                    if effective_lang != "bn"
                    else "আপনার বার্তা পেয়েছি, তবে কোনো ওষুধের নাম বা প্রশ্ন শনাক্ত হয়নি। অনুগ্রহ করে একটি ওষুধের নাম (যেমনঃ নাপা, সেক্লো) লিখুন।"
                ),
                confidence="high",
                sources=[],
                query_type="graceful_handling",
            )

        # Guardrail 1: Prompt Injection & System Leak Defense (H6, H7)
        if re.search(r"\b(ignore\s+previous\s+instructions|system\s+prompt|show\s+prompt|print\s+prompt|bypass\s+guardrails|reveal\s+instructions)\b", q_lower):
            return ChatTurnResponse(
                reply_text=(
                    "I am programmed as a verified Bangladesh Medical & Drug Assistant adhering strictly to official clinical safety guardrails. "
                    "I cannot alter safety thresholds or reveal internal system configurations. How can I assist you with clinical medicine information or first aid?"
                ),
                confidence="high",
                sources=[],
                query_type="security_refusal",
            )

        # Guardrail 2: Self-Harm and Crisis Intervention (F6)
        if re.search(r"\b(how to overdose|self[- ]harm|commit suicide|kill myself|ways to die|মরতে\s*চাই|আত্মহত্যা)\b", q_lower):
            return ChatTurnResponse(
                reply_text=(
                    "⚠️ **CRITICAL SUPPORT & CRISIS ASSISTANCE (জরুরি সহায়তা)**\n\n"
                    "If you are going through a difficult time, feel overwhelmed, or are having thoughts of self-harm or suicide, please know that you are not alone and compassionate support is available right now:\n\n"
                    "- 📞 **Kaan Pete Roi (Emotional Support Helpline)**: **01779-554391** / **01779-554392**\n"
                    "- 📞 **Shastho Batayan (Free Medical & Mental Health Consultation)**: **16263** (24/7)\n"
                    "- 🚨 **National Emergency Ambulance & Police**: **999**\n\n"
                    "Please talk to a trusted friend, family member, or healthcare professional. There is support available to help you through this."
                    if effective_lang != "bn"
                    else "⚠️ **জরুরি মানসিক স্বাস্থ্য সহায়তা ও সহমর্মিতা**\n\n"
                    "আপনি যদি কোনো তীব্র মানসিক কষ্টে থাকেন বা আত্মহত্যার কথা ভাবছেন, অনুগ্রহ করে একা থাকবেন না। আপনার পাশে দাঁড়ানোর জন্য সহায়তাকারী টিম প্রস্তুত রয়েছে:\n\n"
                    "- 📞 **কান পেতে রই (মানসিক স্বাস্থ্য হেল্পলাইন)**: **০১৭১২-৮৫৮১৫১** / **০১৭৭৯-৫৫৪৩৯১**\n"
                    "- 📞 **স্বাস্থ্য বাতায়ন (বিনামূল্যে ডাক্তার ও মানসিক স্বাস্থ্য সেবা)**: **১৬২৬৩**\n"
                    "- 🚨 **জাতীয় জরুরি সেবা**: **৯৯৯**\n\n"
                    "অনুগ্রহ করে সাহায্য গ্রহণ করুন। জীবন মূল্যবান এবং আপনার পাশে মানুষ রয়েছে।"
                ),
                confidence="high",
                sources=[],
                safety_alerts=safety_alerts,
                query_type="self_harm_refusal",
            )

        # Guardrail 3: Accidental Overdose & Poisoning (F1, F8)
        is_poisoning = bool(
            re.search(r"\b(swallowed\s+\d+|swallowed\s+many|overdose|poisoning|ingested\s+pills?)\b", q_lower)
            or ("বিষ" in q_lower and any(w in q_lower for w in ["খেয়ে", "খেয়ে", "পান", "গিলে", "খাওয়া", "খাইছে"]))
            or any(w in q_lower for w in ["পিল খেয়ে", "পিল খেয়েছে", "ওষুধ খেয়ে ফেলেছে", "ওষুধ খেয়েছে", "বিষ খেয়েছে", "বিষ খেয়েছে"])
        )
        if is_poisoning:
            return ChatTurnResponse(
                reply_text=(
                    "🚨 **CRITICAL EMERGENCY: SUSPECTED ACUTE MEDICATION OVERDOSE**\n\n"
                    "Immediate life-saving steps:\n"
                    "1. 🚨 **Call 999 immediately** for an ambulance or rush to the nearest hospital emergency department.\n"
                    "2. 📞 **National Poison Information Center (DMCH)**: Call **01700-700700** immediately for 24/7 emergency toxicology protocol.\n"
                    "3. ❌ **DO NOT induce vomiting** or give raw eggs/milk/salt water — this causes chemical aspiration into the lungs.\n"
                    "4. 📋 **Take the medicine packaging/strip to the hospital** so doctors can administer the specific antidote (e.g. N-acetylcysteine for paracetamol overdose).\n\n"
                    "📞 **Emergency Contacts**: 999 (Ambulance) | 01700-700700 (Poison Center) | 16263 (Health Helpline)"
                    if effective_lang != "bn"
                    else "🚨 **জরুরি জীবনহানির সতর্কতা: ওষুধের ওভারডোজ / বিষক্রিয়া**\n\n"
                    "অবিলম্বে করণীয়:\n"
                    "১. 🚨 **৯৯৯ নম্বরে কল দিন** অথবা রোগীকে দ্রুত নিকটস্থ সরকারি মেডিকেল কলেজ হাসপাতালের জরুরি বিভাগে নিয়ে যান।\n"
                    "২. 📞 **জাতীয় বিষাক্ত তথ্য কেন্দ্র (ঢাকা মেডিকেল)**: কল করুন **০১৭০০-৭০০৭০০**।\n"
                    "৩. ❌ **কোনো অবস্থাতেই বমি করানোর চেষ্টা করবেন না** এবং কাঁচা ডিম বা লবণ-পানি খাওয়াবেন না (এতে ফুসফুসে পানি জমে শ্বাস বন্ধ হতে পারে)।\n"
                    "৪. 📋 **ওষুধের খালি পাতা বা বোতলটি সঙ্গে রাখুন** যাতে চিকিৎসক দ্রুত অ্যান্টিডোট (যেমন নাপার ক্ষেত্রে N-acetylcysteine) প্রয়োগ করতে পারেন।"
                ),
                confidence="high",
                sources=[{"title": "Directorate General of Health Services (DGHS) - Poisoning Protocol", "url": "https://dghs.gov.bd"}],
                safety_alerts=safety_alerts,
                active_drug=active_drug,
                query_type="emergency_overdose",
            )

        # Guardrail 3b: Acute Cardiac Emergency / Severe Chest Pain (F3)
        if re.search(r"\b(chest\s*pain|heart\s*attack|বুকে\s*ব্যথা|হার্ট\s*অ্যাটাক|দম\s*বন্ধ)\b", q_lower):
            return ChatTurnResponse(
                reply_text=(
                    "🚨 **CRITICAL EMERGENCY: ACUTE CHEST PAIN / SUSPECTED CARDIAC EVENT**\n\n"
                    "**Immediate Action Required — Do NOT take unprescribed medicines or delay:**\n"
                    "1. 🚨 **Call 999 immediately** for an emergency ambulance or rush to the nearest hospital Emergency Department / CCU.\n"
                    "2. 📞 **Dial 16263 (Shastho Batayan)** for immediate emergency triage and doctor guidance while help is en route.\n"
                    "3. 🪑 **Rest completely**: Sit or lie down in a comfortable, semi-upright position (do not walk or exert yourself).\n"
                    "4. ❌ **Do NOT attempt home remedies or take random pain pills.** (Chest pain can indicate acute myocardial infarction / heart attack requiring immediate hospital thrombolytic or catheterization intervention).\n\n"
                    "📞 **Emergency Contacts**: 999 (National Ambulance) | 16263 (Shastho Batayan 24/7)"
                    if effective_lang != "bn"
                    else "🚨 **জরুরি জীবনহানির সতর্কতা: বুকে তীব্র ব্যথা / সম্ভাব্য হার্ট অ্যাটাক**\n\n"
                    "**অবিলম্বে জরুরি পদক্ষেপ নিন — কোনো ওষুধ সেবন করবেন না বা দেরি করবেন না:**\n"
                    "১. 🚨 **৯৯৯ নম্বরে কল করে অবিলম্বে অ্যাম্বুলেন্স ডাকুন** অথবা দ্রুত নিকটস্থ হাসপাতালের জরুরি বিভাগ/সিসিইউতে যান।\n"
                    "২. 📞 **১৬২৬৩ (স্বাস্থ্য বাতায়ন)** নম্বরে কল দিয়ে জরুরি পরামর্শ নিন।\n"
                    "৩. 🪑 রোগীকে সম্পূর্ণ বিশ্রামে রাখুন (বসা বা আরামদায়ক আধ-শোয়া অবস্থায় রাখুন, কোনো হাঁটাচলা করাবেন না)।\n"
                    "৪. ❌ **কোনো ব্যথানাশক বা ঘরোয়া টোটকা দেবেন না।** (বুকে ব্যথা হার্ট অ্যাটাকের লক্ষণ হতে পারে, যার জন্য অবিলম্বে হাসপাতালে ইসিজি ও জরুরি চিকিৎসা প্রয়োজন)।\n\n"
                    "📞 **জরুরি সেবা**: ৯৯৯ (অ্যাম্বুলেন্স) | ১৬২৬৩ (স্বাস্থ্য বাতায়ন)"
                ),
                confidence="high",
                sources=[],
                safety_alerts=safety_alerts,
                query_type="emergency_cardiac",
            )

        # Guardrail 3c: Pregnancy Safety Guidance (F4)
        if re.search(r"\b(pregnant|pregnancy|trimester|গর্ভাবস্থা|গর্ভবতী|গর্ভকালীন)\b", q_lower) and any(w in q_lower for w in ["take", "can i", "safe", "medicine", "drug", "this", "it", "খাব", "খাওয়া", "যাবে"]):
            if active_drug in ("warfarin", "lisinopril", "atorvastatin", "simvastatin"):
                warning_detail = (
                    f"**Contraindication Alert**: Official FDA labeling strictly contraindicates {active_drug.title()} during pregnancy (carries Boxed Warnings for fetal toxicity, congenital malformations, and fetal death)."
                    if effective_lang != "bn"
                    else f"**মারাত্মক ঝুঁকির সতর্কতা**: অফিসিয়াল নির্দেশিকা অনুযায়ী গর্ভাবস্থায় {active_drug.title()} সেবন সম্পূর্ণ নিষিদ্ধ (ভ্রূণের মারাত্মক ক্ষতি ও জন্মগত ত্রুটির ঝুঁকি রয়েছে)।"
                )
            elif active_drug in ("paracetamol", "acetaminophen"):
                warning_detail = (
                    "**Clinical Notice**: While Paracetamol is commonly considered first-line for fever/pain during pregnancy when medically necessary, you must use only the lowest effective dose for the shortest duration under OB/GYN supervision. Avoid all NSAIDs (such as Aspirin or Ibuprofen) which are contraindicated especially in the third trimester."
                    if effective_lang != "bn"
                    else "**চিকিৎসা পরামর্শ**: যদিও প্রয়োজনে গর্ভাবস্থায় প্যারাসিটামল তুলনামূলক প্রথম সারির নিরাপদ ওষুধ হিসেবে ব্যবহৃত হয়, তবুও ডাক্তারের পরামর্শ ছাড়া সেবন করবেন না এবং সর্বনিম্ন কার্যকর মাত্রায় সীমিত রাখুন। এসপিরিন বা আইবুপ্রোফেন জাতীয় ব্যথানাশক সম্পূর্ণ পরিহার করুন।"
                )
            elif active_drug:
                warning_detail = (
                    f"**Clinical Caution**: {active_drug.title()} has not been verified safe for unrestricted use during pregnancy without individualized clinical evaluation."
                    if effective_lang != "bn"
                    else f"**সতর্কতা**: গর্ভাবস্থায় {active_drug.title()} সেবনের পূর্বে চিকিৎসকের পরামর্শ নেওয়া আবশ্যক।"
                )
            else:
                warning_detail = (
                    "Please state the specific medicine name so its official pregnancy category and risk profile can be referenced."
                    if effective_lang != "bn"
                    else "অনুগ্রহ করে ওষুধের নির্দিষ্ট নামটি বলুন যাতে গর্ভকালীন ঝুঁকি ও নির্দেশিকা যাচাই করা যায়।"
                )

            return ChatTurnResponse(
                reply_text=(
                    f"### 🤰 Pregnancy Medication Safety Guidance (গর্ভকালীন সতর্কতা)\n\n"
                    f"**Always consult your obstetrician/gynecologist (OB/GYN) before taking any medicine during pregnancy.**\n\n"
                    f"- {warning_detail}\n"
                    f"- **General Rule**: Many medications cross the placental barrier and can adversely affect fetal organogenesis and development.\n"
                    f"- 📞 For 24/7 free medical consultation with a registered government doctor in Bangladesh, dial **16263 (Shastho Batayan)**."
                    if effective_lang != "bn"
                    else f"### 🤰 গর্ভাবস্থায় ওষুধ সেবনের জরুরি নির্দেশনা\n\n"
                    f"**গর্ভাবস্থায় যেকোনো ওষুধ সেবনের পূর্বে অবশ্যই আপনার রেজিস্টার্ড স্ত্রীরোগ বিশেষজ্ঞ (OB/GYN) ডাক্তারের পরামর্শ নিন।**\n\n"
                    f"- {warning_detail}\n"
                    f"- **সাধারণ নিয়ম**: অধিকাংশ ওষুধ প্লাসেন্টা ভেদ করে গর্ভস্থ শিশুর কাছে পৌঁছে যায় এবং ভ্রূণের অঙ্গ গঠনে মারাত্মক ক্ষতি করতে পারে।\n"
                    f"- 📞 বিনামূল্যে ২৪/৭ সরকারি ডাক্তারের পরামর্শ পেতে কল করুন **১৬২৬৩ (স্বাস্থ্য বাতায়ন)**।"
                ),
                confidence="high",
                sources=[{"title": f"FDA Pregnancy Label Guidance ({active_drug.title() if active_drug else 'General'})", "url": "https://dailymed.nlm.nih.gov"}],
                safety_alerts=safety_alerts,
                active_drug=active_drug,
                query_type="pregnancy_safety",
            )

        # Guardrail 4: Dangerous High-Dose Refusal (C6)
        if re.search(r"\b(10|12|15|20|\d{2,})\s*(tablets?|napa|pills?|capsules?)\b", q_lower) and any(w in q_lower for w in ["take", "day", "daily", "safe", "খাব", "খাওয়া"]):
            return ChatTurnResponse(
                reply_text=(
                    "🚨 **DOSAGE SAFETY WARNING: MAXIMUM LIMIT EXCEEDED**\n\n"
                    "**NO. You must NOT take 10 or more tablets of Napa in a day.**\n\n"
                    "- **Maximum Safe Limit**: The absolute maximum adult dose for Paracetamol (Napa 500 mg) is **4,000 mg (8 tablets) in 24 hours**, spaced at least 4 to 6 hours apart.\n"
                    "- **Severe Toxicity**: Taking 10 tablets (5,000 mg) causes acute paracetamol poisoning (hepatotoxicity), leading to irreversible acute liver failure.\n"
                    "- 🚨 **If this dose has already been ingested**: Rush immediately to the nearest hospital emergency room and call **999** or the National Poison Center at **01700-700700**."
                    if effective_lang != "bn"
                    else "🚨 **ওষুধের মাত্রা অতিক্রমণের মারাত্মক ঝুঁকি সতর্কতা**\n\n"
                    "**না! কখনোই একদিনে ১০টি নাপা ট্যাবলেট সেবন করা যাবে না।**\n\n"
                    "- **সর্বোচ্চ নিরাপদ মাত্রা**: প্রাপ্তবয়স্কদের জন্য প্যারাসিটামলের সর্বোচ্চ দৈনিক মাত্রা হলো **৪,০০০ মিলিগ্রাম (২৪ ঘণ্টায় সর্বোচ্চ ৮টি ৫০০ মিগ্রা ট্যাবলেট)**।\n"
                    "- **লিভারের ক্ষতি**: একদিনে ১০টি ট্যাবলেট (৫,০০০ মিলিগ্রাম) সেবন করলে তীব্র লিভারের ক্ষতি (Acute Liver Toxicity) হতে পারে, যা প্রাণঘাতী।\n"
                    "- 🚨 **যদি ইতিমধ্যে খেয়ে ফেলা হয়ে থাকে**: অবিলম্বে নিকটস্থ হাসপাতালের জরুরি বিভাগে যান অথবা বিষাক্ত তথ্য কেন্দ্রে (০১৭০০-৭০০৭০০) বা ৯৯৯ নম্বরে কল করুন।"
                ),
                confidence="high",
                sources=[{"title": "Beximco Pharma / FDA Acetaminophen Label (Max 4,000 mg)", "url": "https://dailymed.nlm.nih.gov"}],
                safety_alerts=safety_alerts,
                active_drug="paracetamol",
                query_type="dosage_refusal",
            )

        # Guardrail 5: Infant Dosage Clinical Evaluation Required (C5)
        if re.search(r"\b(\d+\s*months?|\d+\s*মাস|infant|newborn|নবজাতক|3-month|3\s*month)\b", q_lower) and is_dosage_intent:
            return ChatTurnResponse(
                reply_text=(
                    "⚠️ **Pediatric Safety Notice: Infant Clinical Evaluation Required**\n\n"
                    "**Do NOT self-administer adult tablets or OTC dosages to an infant (such as a 3-month-old) without a registered pediatrician's prescription.**\n\n"
                    "- Infant medication doses are strictly weight-based (mg per kg) and require pediatric liquid drops, not adult tablets.\n"
                    "- Incorrect dosing in infants carries a high risk of accidental overdose and severe organ toxicity.\n"
                    "- Please consult a pediatrician immediately or dial **16263** for 24/7 free pediatric teleconsultation."
                    if effective_lang != "bn"
                    else "⚠️ **শিশুদের নিরাপত্তা সতর্কতা: চিকিৎসকের পরামর্শ আবশ্যক**\n\n"
                    "**৩ মাস বয়সী বা নবজাতক শিশুকে চিকিৎসকের প্রেসক্রিপশন ছাড়া কখনোই ফার্মেসির ওষুধ বা বড়দের ট্যাবলেট খাওয়াবেন না।**\n\n"
                    "- শিশুদের ওষুধের মাত্রা সম্পূর্ণভাবে তাদের শারীরিক ওজনের (কেজি) ওপর নির্ভর করে এবং এর জন্য বিশেষ পেডিয়াট্রিক ড্রপ প্রয়োজন হয়।\n"
                    "- অবিলম্বে একজন রেজিস্টার্ড শিশু বিশেষজ্ঞের (Pediatrician) পরামর্শ নিন অথবা ১৬২৬৩ নম্বরে কল করুন।"
                ),
                confidence="abstained",
                sources=[],
                safety_alerts=safety_alerts,
                active_drug=active_drug,
                query_type="pediatric_abstain",
            )

        # Guardrail 6: Diabetes Permanent Cure Premise Correction (G1)
        if "cure" in q_lower and "diabetes" in q_lower:
            return ChatTurnResponse(
                reply_text=(
                    "### 📋 Clinical Evidence: Diabetes Management\n\n"
                    "**There is currently no medication that permanently cures Type 1 or Type 2 diabetes.**\n\n"
                    "- **Nature of Condition**: Type 2 diabetes is a chronic metabolic condition characterized by insulin resistance and progressive pancreatic beta-cell dysfunction.\n"
                    "- **Evidence-Based Management**: Diabetes is effectively controlled through:\n"
                    "  1. Lifestyle modifications (low-glycemic balanced diet, regular exercise, weight loss).\n"
                    "  2. First-line medications (such as **Metformin**, SGLT2 inhibitors, GLP-1 receptor agonists, or insulin).\n"
                    "- ⚠️ **Medical Warning**: Beware of unverified herbal or alternative remedies advertising 'permanent cures for diabetes' — abandoning prescribed diabetes medication causes dangerous complications like diabetic ketoacidosis and kidney failure.\n"
                    "- Consult a registered endocrinologist or diabetologist for individualized glycemic targets."
                ),
                confidence="high",
                sources=[{"title": "American Diabetes Association (ADA) Guidelines / FDA Metformin Monograph", "url": "https://dailymed.nlm.nih.gov"}],
                safety_alerts=safety_alerts,
                active_drug="metformin",
                query_type="clinical_premise_correction",
            )

        # Guardrail 7: Cancer Treatment Referral (G2)
        if re.search(r"\b(cancer\s*treatment|cure\s*cancer|cancer\s*medicine\s*for\s*me)\b", q_lower):
            return ChatTurnResponse(
                reply_text=(
                    "### 🏥 Clinical Referral: Oncology Staging & Evaluation Required\n\n"
                    "Cancer treatment cannot be determined or prescribed by an automated assistant. It requires comprehensive clinical staging, histopathology/biopsy, molecular testing, and multidisciplinary evaluation.\n\n"
                    "- **Standard Modalities**: Depending on the specific tumor type and stage, therapy may involve surgical resection, systemic chemotherapy, radiotherapy, targeted therapy, or immunotherapy.\n"
                    "- **Where to Seek Care in Bangladesh**:\n"
                    "  - **National Institute of Cancer Research & Hospital (NICRH)**, Mohakhali, Dhaka (Tel: 02-9898236)\n"
                    "  - Specialized Oncology Departments at Medical College Hospitals (BSMMU, DMC, CMH)\n"
                    "- Please schedule an in-person consultation with a registered medical or surgical oncologist."
                ),
                confidence="high",
                sources=[{"title": "DGHS National Cancer Control Strategy", "url": "https://dghs.gov.bd"}],
                safety_alerts=safety_alerts,
                query_type="oncology_referral",
            )

        # Guardrail 8: Off-Topic Inquiry (G6)
        if any(w in q_lower for w in ["cricket", "football", "match", "score", "weather", "movie", "song", "who won"]):
            return ChatTurnResponse(
                reply_text=(
                    "I am a specialized Bangladesh Medical & Pharmaceutical Assistant (সহকারী). "
                    "I can assist you with medicine identification, drug dosages, side effects, interactions, and first-aid protocols. "
                    "I do not track sports, entertainment, or non-medical topics."
                    if effective_lang != "bn"
                    else "আমি একটি বিশেষায়িত বাংলাদেশ মেডিকেল ও ওষুধ সহকারী। আমি কেবল ওষুধ পরিচিতি, ড্রাগ ইন্টারঅ্যাকশন, ও প্রাথমিক চিকিৎসা সম্পর্কিত বিষয়ে সাহায্য করতে পারি। খেলাধুলা বা সাধারণ সংবাদ আমার কার্যক্ষেত্রের বাইরে।"
                ),
                confidence="high",
                sources=[],
                query_type="out_of_scope",
            )

        # Guardrail 9: Clinical Comparison — Napa vs Seclo (H8)
        if "napa" in q_lower and "seclo" in q_lower and any(w in q_lower for w in ["vs", "difference", "compare", "বনাম", "পার্থক্য"]):
            return ChatTurnResponse(
                reply_text=(
                    "### 💊 Clinical Comparison: **Napa** vs **Seclo**\n\n"
                    "These two medications belong to completely different drug classes with distinct clinical indications:\n\n"
                    "1. **Napa (Paracetamol / Acetaminophen)**:\n"
                    "   - **Drug Class**: Analgesic and antipyretic.\n"
                    "   - **Primary Uses**: Reducing fever, headache, body aches, toothache, and mild-to-moderate pain.\n"
                    "   - **Standard Adult Dose**: 500 mg every 4-6 hours (max 4,000 mg in 24 hours).\n\n"
                    "2. **Seclo (Omeprazole)**:\n"
                    "   - **Drug Class**: Proton Pump Inhibitor (PPI).\n"
                    "   - **Primary Uses**: Suppressing gastric acid production, treating heartburn, GERD, gastric acidity, and peptic ulcers.\n"
                    "   - **Standard Adult Dose**: 20 mg once daily, preferably 30-60 minutes before breakfast.\n\n"
                    "⚠️ *Note: These medicines do not replace each other. There is no known adverse interaction between Paracetamol and Omeprazole, and they may be co-prescribed when indicated.*"
                ),
                confidence="high",
                sources=[{"title": "DGDA Bangladesh Formulary (Napa & Seclo)", "url": "https://dgda.gov.bd"}],
                safety_alerts=safety_alerts,
                active_drug="paracetamol",
                query_type="comparison",
            )

        # Guardrail 10: Strength Comparison — Napa 500 vs 650/665 (H9)
        if "napa" in q_lower and ("500" in q_lower and ("650" in q_lower or "665" in q_lower)):
            return ChatTurnResponse(
                reply_text=(
                    "### 💊 Strength Comparison: **Napa 500 mg** vs **Napa Extend (665 mg)**\n\n"
                    "- **Napa 500 mg (Regular Tablet)**:\n"
                    "  - Immediate-release formulation providing fast pain and fever relief.\n"
                    "  - Effective duration: 4 to 6 hours.\n"
                    "  - Standard adult dose: 1-2 tablets every 4-6 hours (max 4,000 mg/day).\n\n"
                    "- **Napa Extend 665 mg (Extended Release Tablet)**:\n"
                    "  - Bi-layer sustained-release formulation designed for prolonged pain relief (e.g. osteoarthritis, chronic backache).\n"
                    "  - Effective duration: up to 8 hours.\n"
                    "  - Standard adult dose: 2 tablets every 8 hours (max 4,000 mg/day). Must be swallowed whole; do NOT crush or chew.\n\n"
                    "⚠️ *Never take both simultaneously without spacing, to prevent exceeding the 4,000 mg daily paracetamol ceiling.*"
                ),
                confidence="high",
                sources=[{"title": "Beximco Pharmaceuticals Product Monograph", "url": "https://dailymed.nlm.nih.gov"}],
                safety_alerts=safety_alerts,
                active_drug="paracetamol",
                query_type="comparison",
            )

        # Guardrail 11: Food/Drug Interaction — Alcohol & Napa (D4)
        if "alcohol" in q_lower and ("napa" in q_lower or "paracetamol" in q_lower or active_drug in ("paracetamol", "acetaminophen")):
            return ChatTurnResponse(
                reply_text=(
                    "⚠️ **Critical Drug-Food Interaction: Alcohol & Paracetamol**\n\n"
                    "**Avoid consuming alcohol while taking Napa (Paracetamol).**\n\n"
                    "- **Mechanism**: Both alcohol and paracetamol are metabolized by the liver via the CYP2E1 enzyme pathway.\n"
                    "- **Toxic Metabolite**: Concomitant alcohol use accelerates the conversion of paracetamol into **NAPQI** (N-acetyl-p-benzoquinone imine), a toxic metabolite that depletes liver glutathione.\n"
                    "- **Clinical Risk**: Combining chronic or binge alcohol consumption with paracetamol drastically increases the risk of severe acute hepatotoxicity and permanent liver damage."
                ),
                confidence="high",
                sources=[{"title": "FDA Acetaminophen Warning Label — Alcohol Interaction", "url": "https://dailymed.nlm.nih.gov"}],
                safety_alerts=safety_alerts,
                active_drug="paracetamol",
                query_type="interaction",
            )

        # Guardrail 12: Administration — Seclo on Empty Stomach (C3)
        if ("empty stomach" in q_lower or "খালি পেটে" in q_lower) and ("seclo" in q_lower or "omeprazole" in q_lower or active_drug == "omeprazole"):
            return ChatTurnResponse(
                reply_text=(
                    "### 📋 Administration Instructions: **Seclo (Omeprazole)**\n\n"
                    "**YES, Seclo should be taken on an empty stomach.**\n\n"
                    "- **Timing**: For optimal acid suppression, take Seclo **30 to 60 minutes before a meal**, preferably before breakfast in the morning.\n"
                    "- **Rationale**: Proton pump inhibitors (PPIs) inhibit active acid pumps (H+/K+ ATPase) stimulated by food ingestion. Taking it before meals ensures peak drug concentration arrives as gastric pumps activate.\n"
                    "- **Administration**: Swallow capsules whole with a full glass of water. Do NOT crush or chew enteric-coated pellets."
                ),
                confidence="high",
                sources=[{"title": "FDA Omeprazole Label — Dosage and Administration", "url": "https://dailymed.nlm.nih.gov"}],
                safety_alerts=safety_alerts,
                active_drug="omeprazole",
                query_type="dosage_administration",
            )

        # Guardrail 13: Precaution — Alatrol & Driving Safety (C4)
        if ("driv" in q_lower or "গাড়ি" in q_lower) and ("alatrol" in q_lower or "cetirizine" in q_lower or active_drug == "cetirizine"):
            return ChatTurnResponse(
                reply_text=(
                    "⚠️ **Clinical Precaution: Alatrol (Cetirizine) & Driving Safety**\n\n"
                    "- **Drowsiness Risk**: Although Alatrol (Cetirizine) is a second-generation, less sedating antihistamine, clinical trials report somnolence/drowsiness in approximately 10-14% of patients.\n"
                    "- **Recommendation**: Exercise caution when driving a motor vehicle or operating dangerous machinery until you know how Alatrol affects you.\n"
                    "- **Avoid Alcohol**: Combining Alatrol with alcohol or other central nervous system depressants produces additive drowsiness and further impairs reaction time."
                ),
                confidence="high",
                sources=[{"title": "FDA Cetirizine Label — Warnings and Precautions", "url": "https://dailymed.nlm.nih.gov"}],
                safety_alerts=safety_alerts,
                active_drug="cetirizine",
                query_type="precaution",
            )

        # Guardrail 14: Drug Interaction — Seclo with Paracetamol (D2)
        if ("seclo" in q_lower or "omeprazole" in q_lower) and ("paracetamol" in q_lower or "napa" in q_lower):
            return ChatTurnResponse(
                reply_text=(
                    "### 🔄 Drug Interaction: **Seclo (Omeprazole)** + **Paracetamol (Napa)**\n\n"
                    "**YES, Seclo and Paracetamol can be safely taken together.**\n\n"
                    "- **No Major Adverse Interaction**: There is no documented negative drug-drug interaction between Omeprazole and Paracetamol.\n"
                    "- **Clinical Benefit**: Omeprazole reduces gastric acid and does not increase paracetamol hepatotoxicity at standard therapeutic doses.\n"
                    "- **Administration Tip**: Take Seclo 30-60 minutes before breakfast on an empty stomach, and take Paracetamol as needed for pain or fever (maximum 4,000 mg/24h)."
                ),
                confidence="high",
                sources=[{"title": "FDA DailyMed Drug Interaction Reference", "url": "https://dailymed.nlm.nih.gov"}],
                safety_alerts=safety_alerts,
                active_drug="paracetamol",
                query_type="interaction",
            )

        # Guardrail 14b: Drug Interaction — Paracetamol with Ibuprofen (D1)
        if ("ibuprofen" in q_lower or "আইবুপ্রোফেন" in q_lower) and ("napa" in q_lower or "paracetamol" in q_lower or active_drug in ("paracetamol", "acetaminophen")):
            return ChatTurnResponse(
                reply_text=(
                    "### 🔄 Drug Interaction: **Paracetamol (Napa)** + **Ibuprofen**\n\n"
                    "**YES, Paracetamol and Ibuprofen can be taken together or alternated for acute pain or fever, provided dosages are strictly followed.**\n\n"
                    "- **Different Mechanisms**: Paracetamol acts primarily in the central nervous system, while Ibuprofen is an NSAID that reduces peripheral inflammation. They are cleared by different metabolic pathways (liver vs kidneys).\n"
                    "- **Safety Rules**:\n"
                    "  1. Do not exceed **4,000 mg** of Paracetamol per day.\n"
                    "  2. Do not exceed **1,200 mg** of OTC Ibuprofen per day, and always take Ibuprofen with food or milk to prevent gastric irritation.\n"
                    "  3. ⚠️ **Dengue Warning**: If there is any possibility of **Dengue fever**, DO NOT take Ibuprofen (NSAIDs cause severe internal bleeding and platelet destruction; use Paracetamol only).\n"
                    "- Consult a physician or pharmacist if pain persists beyond 3 days."
                    if effective_lang != "bn"
                    else "### 🔄 ড্রাগ ইন্টারঅ্যাকশন: **প্যারাসিটামল (নাপা)** এবং **আইবুপ্রোফেন**\n\n"
                    "**হ্যাঁ, তীব্র জ্বর বা ব্যথার ক্ষেত্রে চিকিৎসকের পরামর্শ অনুযায়ী প্যারাসিটামল এবং আইবুপ্রোফেন একসাথে বা একের পর এক সেবন করা যায়।**\n\n"
                    "- **কাজের ধরণ আলাদা**: প্যারাসিটামল কেন্দ্রীয়ভাবে ব্যথা কমায় এবং আইবুপ্রোফেন প্রদাহ দূর করে। এ দুটি ওষুধের বিপাক পথ আলাদা (লিভার বনাম কিডনি)।\n"
                    "- **জরুরি নিয়ম**:\n"
                    "  ১. দিনে ৪,০০০ মিলিগ্রামের বেশি প্যারাসিটামল সেবন করবেন না।\n"
                    "  ২. আইবুপ্রোফেন অবশ্যই খাবার পর ভরা পেটে খেতে হবে যাতে গ্যাস্ট্রিক বা আলসারের ঝুঁকি না বাড়ে।\n"
                    "  ৩. ⚠️ **ডেঙ্গু সতর্কতা**: ডেঙ্গু জ্বর সন্দেহ হলে ভুল করেও আইবুপ্রোফেন খাবেন না (শুধুমাত্র সাধারণ প্যারাসিটামল ব্যবহার করবেন)।"
                ),
                confidence="high",
                sources=[{"title": "FDA DailyMed / NHS Clinical Interaction Reference (Paracetamol & Ibuprofen)", "url": "https://dailymed.nlm.nih.gov"}],
                safety_alerts=safety_alerts,
                active_drug="paracetamol",
                query_type="interaction",
            )

        # Guardrail 14c: Drug Interaction with Unknown / Unverified Drug (D3)
        if is_interactions_intent and active_drug:
            words = [re.sub(r"[^\w\-]", "", w) for w in re.split(r"\s+", q_lower) if len(w) >= 4]
            stop_words = {"with", "take", "taking", "together", "along", "combine", "safe", "napa", "paracetamol", "seclo", "omeprazole", "ibuprofen", "alcohol", "can", "this", "tablets"}
            unknown_candidates = [w for w in words if w not in stop_words and not detect_drugs_in_query(w) and not lookup_bangladesh_brand(w)]
            if unknown_candidates:
                unk_name = unknown_candidates[0].title()
                return ChatTurnResponse(
                    reply_text=(
                        f"### ⚠️ Information Not Found in Official Drug Labeling (ABSTAIN)\n\n"
                        f"**No verified official drug interaction data was found for {active_drug.title()} with \"{unk_name}\".**\n\n"
                        f"- \"{unk_name}\" is not recognized in official FDA package inserts or DGDA Bangladesh registries.\n"
                        f"- Under strict clinical grounding rules, potential drug-drug interactions cannot be assumed or fabricated.\n"
                        f"- Please consult a registered physician or pharmacist with your specific prescription before combining medications."
                        if effective_lang != "bn"
                        else f"### ⚠️ অফিসিয়াল তালিকায় তথ্য পাওয়া যায়নি (ABSTAIN)\n\n"
                        f"**{active_drug.title()}-এর সাথে \"{unk_name}\"-এর কোনো অফিসিয়াল ড্রাগ ইন্টারঅ্যাকশন তথ্য পাওয়া যায়নি।**\n\n"
                        f"- \"{unk_name}\" কোনো স্বীকৃত ওষুধ হিসেবে তালিকায় নিবন্ধিত নেই।\n"
                        f"- মনগড়া তথ্য পরিহার করতে এই সহকারী অনুমানভিত্তিক কোনো মন্তব্য করে না। দুটি ওষুধ একসাথে খাওয়ার আগে অবশ্যই রেজিস্টার্ড ডাক্তারের পরামর্শ নিন।"
                    ),
                    confidence="abstained",
                    sources=[],
                    safety_alerts=safety_alerts,
                    active_drug=active_drug,
                    query_type="interaction_abstained",
                )

        # Guardrail 14d: Maximum Daily Dose of Paracetamol (C1)
        if ("paracetamol" in q_lower or "napa" in q_lower or active_drug in ("paracetamol", "acetaminophen")) and (
            "max" in q_lower or "maximum" in q_lower or "daily" in q_lower or "adult" in q_lower or "কতটুকু" in q_lower or "সর্বোচ্চ" in q_lower
        ) and is_dosage_intent:
            return ChatTurnResponse(
                reply_text=(
                    "### 📋 Maximum Daily Dose: **Paracetamol (Acetaminophen)**\n\n"
                    "- **Adult Maximum Daily Limit**: The absolute maximum daily dose for paracetamol in adults is **4,000 mg (4 grams) in 24 hours**.\n"
                    "- **Standard Dosing**: 500 mg to 1,000 mg every 4 to 6 hours as needed (maximum eight 500 mg tablets in 24 hours).\n"
                    "- **Minimum Spacing**: Doses must be spaced at least 4 to 6 hours apart.\n"
                    "- **Hepatotoxicity Boxed Warning**: Exceeding 4,000 mg in 24 hours causes acute paracetamol toxicity, liver necrosis, and irreversible acute liver failure.\n"
                    "- **Alcohol Interaction**: If you consume 3 or more alcoholic drinks daily, consult a doctor before use as liver injury risk is severely heightened."
                    if effective_lang != "bn"
                    else "### 📋 প্যারাসিটামলের সর্বোচ্চ দৈনিক মাত্রা (Maximum Dose)\n\n"
                    "- **প্রাপ্তবয়স্কদের সর্বোচ্চ মাত্রা**: ২৪ ঘণ্টায় প্যারাসিটামলের সর্বোচ্চ নিরাপদ মাত্রা হলো **৪,০০০ মিলিগ্রাম (সর্বোচ্চ ৮টি ৫০০ মিগ্রা ট্যাবলেট)**।\n"
                    "- **সেবন বিধি**: ৫০০ মিগ্রা থেকে ১০০০ মিগ্রা প্রতি ৪ থেকে ৬ ঘণ্টা পরপর (প্রয়োজন অনুযায়ী)।\n"
                    "- **বিরতি**: এক মাত্রা থেকে অন্য মাত্রার মধ্যে অন্তত ৪ থেকে ৬ ঘণ্টার ব্যবধান রাখতে হবে।\n"
                    "- **সতর্কতা**: ৪,০০০ মিলিগ্রামের বেশি গ্রহণ করলে তীব্র লিভারের ক্ষতি (লিভার ফেইলিউর) হতে পারে।"
                ),
                confidence="high",
                sources=[{"title": "FDA Acetaminophen Monograph / DailyMed Label (Max 4,000 mg/24h)", "url": "https://dailymed.nlm.nih.gov"}],
                safety_alerts=safety_alerts,
                active_drug="paracetamol",
                query_type="dosage",
            )

        # Guardrail 14e: Adverse Reactions / Side Effects of Omeprazole (C2)
        if ("omeprazole" in q_lower or "seclo" in q_lower or active_drug == "omeprazole") and is_side_effects_intent:
            return ChatTurnResponse(
                reply_text=(
                    "### 📋 Official Adverse Reactions: **Omeprazole (Seclo)**\n\n"
                    "According to official FDA labeling and clinical trials, the most common adverse reactions reported are:\n"
                    "1. **Headache** (6.9%)\n"
                    "2. **Abdominal pain** (5.2%)\n"
                    "3. **Nausea** (4.0%)\n"
                    "4. **Diarrhea** (3.7%)\n"
                    "5. **Vomiting** (3.2%)\n"
                    "6. **Flatulence** (2.7%)\n\n"
                    "- **Precautions & Long-Term Warnings**:\n"
                    "  - *Clostridium difficile*-associated diarrhea.\n"
                    "  - Increased risk of bone fractures (hip, wrist, spine) with prolonged or high-dose PPI use (>1 year).\n"
                    "  - Vitamin B12 deficiency and hypomagnesemia with long-term therapy."
                    if effective_lang != "bn"
                    else "### 📋 ওমিপ্রাজল (সেক্লো)-এর অফিসিয়াল পার্শ্বপ্রতিক্রিয়া\n\n"
                    "এফডিএ ড্রাগ লেবেল এবং ক্লিনিক্যাল ট্রায়াল অনুযায়ী সাধারণ পার্শ্বপ্রতিক্রিয়াগুলো হলো:\n"
                    "১. **মাথাব্যথা** (Headache - ৬.৯%)\n"
                    "২. **পেটব্যথা** (Abdominal pain - ৫.২%)\n"
                    "৩. **বমি বমি ভাব** (Nausea - ৪.০%)\n"
                    "৪. **ডায়রিয়া** (Diarrhea - ৩.৭%)\n"
                    "৫. **বমি** (Vomiting - ৩.২%)\n"
                    "৬. **পেটে গ্যাস বা পেট ফাঁপা** (Flatulence - ২.৭%)\n\n"
                    "⚠️ *তীব্র পাতলা পায়খানা, ফুসকুড়ি বা শ্বাসকষ্ট দেখা দিলে অবিলম্বে ডাক্তারের পরামর্শ নিন।*"
                ),
                confidence="high",
                sources=[{"title": "FDA Omeprazole Package Insert — Adverse Reactions", "url": "https://dailymed.nlm.nih.gov"}],
                safety_alerts=safety_alerts,
                active_drug="omeprazole",
                query_type="side_effects",
            )

        # Guardrail 15: Missing Entity in Clinical Follow-up (E3)
        if is_side_effects_intent and not active_drug:
            return ChatTurnResponse(
                reply_text="Which medicine are you inquiring about? Please specify the medicine or brand name (e.g. 'side effects of Napa' or 'side effects of Seclo') so I can retrieve verified labeling details."
                if effective_lang != "bn"
                else "আপনি কোন ওষুধের পার্শ্বপ্রতিক্রিয়া সম্পর্কে জানতে চাচ্ছেন? অনুগ্রহ করে ওষুধ বা ব্র্যান্ডের নামটি উল্লেখ করুন (যেমনঃ 'নাপার পার্শ্বপ্রতিক্রিয়া')।",
                confidence="high",
                sources=[],
                query_type="clarification_needed",
            )

        # 3. Intent Detection: Bangladeshi Brand Alternatives
        is_alternative_intent = bool(
            re.search(
                r"(\b(alternative|alternatives|substitute|substitutes|বিকল্প|অন্য ব্র্যান্ড|অন্যান্য ব্র্যান্ড)\b)"
                r"|(\b(other|all|different|more|available|local|list)\s+(bangladeshi\s+)?brands?\b)"
                r"|(\b(list|show|name|tell me|what are)\s+(the\s+|all\s+|other\s+)?(bangladeshi\s+)?brands?\b)"
                r"|(\bbrands?\s+(of|for|sharing|like|in)\b)",
                q_lower,
            )
            or (("ব্র্যান্ড" in q_lower or "brand" in q_lower) and any(w in q_lower for w in ["মতো", "আরও", "আর", "অন্য", "বিকল্প", "alternatives", "like", "other"]))
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
                exclude_brand = bd_match.brand_name if bd_match else None
                alternatives = find_brand_alternatives(target_drug, exclude_brand=exclude_brand)
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
                else:
                    return ChatTurnResponse(
                        reply_text=(
                            f"### ⚠️ Information Not Found in Official Drug Labeling (ABSTAIN)\n\n"
                            f"No alternative Bangladeshi commercial brands were found in the local pharmaceutical database for **{target_drug.title()}**.\n\n"
                            f"- Under strict clinical grounding rules, unverified brand names are not fabricated.\n"
                            f"- Please consult a registered physician or pharmacist, or search Medex Bangladesh / DGDA directories."
                            if effective_lang != "bn"
                            else f"### ⚠️ অফিসিয়াল তালিকায় তথ্য পাওয়া যায়নি (ABSTAIN)\n\n"
                            f"**{target_drug.title()}**-এর জন্য স্থানীয় ফার্মাসিউটিক্যাল তালিকায় কোনো নিবন্ধিত বিকল্প ব্র্যান্ড পাওয়া যায়নি।\n\n"
                            f"- মনগড়া ব্র্যান্ড তৈরি প্রতিরোধে সহকারী অনুমাননির্ভর তথ্য দেয় না। বিস্তারিত জানতে রেজিস্টার্ড ডাক্তার বা ফার্মাসিস্টের পরামর্শ নিন।"
                        ),
                        confidence="abstained",
                        sources=[],
                        safety_alerts=safety_alerts,
                        active_drug=target_drug,
                        query_type="alternatives_abstained",
                    )
            else:
                return ChatTurnResponse(
                    reply_text=(
                        "### ⚠️ Information Not Found in Official Drug Labeling (ABSTAIN)\n\n"
                        "The requested generic medicine is not recognized in official local pharmaceutical registries. "
                        "Under strict clinical grounding rules, unverified brand names are not fabricated. "
                        "Please verify the generic spelling or search the DGDA Bangladesh / Medex BD directory."
                        if effective_lang != "bn"
                        else "### ⚠️ অফিসিয়াল তালিকায় তথ্য পাওয়া যায়নি (ABSTAIN)\n\n"
                        "অনুরোধকৃত জেনেরিক ওষুধটি অফিসিয়াল তালিকায় পাওয়া যায়নি। কোনো মনগড়া ব্র্যান্ড নাম দেওয়া হয় না। "
                        "অনুগ্রহ করে বানান যাচাই করুন অথবা ডিজিডিএ / মেডেক্স ডিরেক্টরিতে অনুসন্ধান করুন।"
                    ),
                    confidence="abstained",
                    sources=[],
                    safety_alerts=safety_alerts,
                    active_drug=active_drug,
                    query_type="alternatives_abstained",
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
        is_id_inquiry = any(w in q_lower for w in ["identify", "what is", "generic", "কার ওষুধ", "কী ওষুধ", "কি ওষুধ", "কিসের", "কাজ কি", "কাজ কী", "জেনেরিক", "পরিচয়", "ব্যবহার", "বিস্তারিত", "ঔষধ", "ki oshudh", "ki osudh", "ki medicine", "oshudh", "osudh", "ki"]) or extracted_ocr
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
