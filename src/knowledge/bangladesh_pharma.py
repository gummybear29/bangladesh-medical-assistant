"""Bangladesh Pharmaceutical Directory & Emergency Directory.

Maps popular Bangladeshi commercial brand names (Square, Beximco, Incepta,
Renata, ACI, Eskayef, Aristopharma, etc.) to canonical generic ingredients,
manufacturers, standard indications, and emergency contact registries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# ── Emergency Hotlines (Bangladesh) ───────────────────────────────────

BANGLADESH_EMERGENCY_HOTLINES = {
    "999": {
        "title": "National Emergency Service (জাতীয় জরুরি সেবা)",
        "number": "999",
        "description": "Ambulance, Police, and Fire Rescue throughout Bangladesh.",
        "type": "all_emergency",
    },
    "16263": {
        "title": "Shastho Batayan (স্বাস্থ্য বাতায়ন)",
        "number": "16263",
        "description": "Government 24/7 health helpline for free medical doctor consultation and hospital triage.",
        "type": "health",
    },
    "10655": {
        "title": "IEDCR Disease Control Helpline (আইইডিসিআর)",
        "number": "10655",
        "description": "Institute of Epidemiology, Disease Control and Research for viral outbreaks (Dengue, Nipah, COVID).",
        "type": "epidemic",
    },
    "01700-700700": {
        "title": "National Poison Information Center (জাতীয় বিষাক্ত তথ্য কেন্দ্র)",
        "number": "01700-700700",
        "description": "Dhaka Medical College Hospital 24/7 emergency poison, snakebite, and toxicology hotline.",
        "type": "toxicology",
    },
}


# ── Brand Data Model ──────────────────────────────────────────────────


@dataclass
class BangladeshDrugInfo:
    brand_name: str
    generic_name: str
    manufacturer: str
    strength: str
    dosage_form: str
    common_uses: str
    bengali_generic: str
    bengali_uses: str
    otc: bool
    warning: str = ""


# ── Top Bangladeshi Brands Dictionary ─────────────────────────────────

BANGLADESH_BRANDS: dict[str, BangladeshDrugInfo] = {
    # Paracetamol / Acetaminophen
    "napa": BangladeshDrugInfo(
        brand_name="Napa",
        generic_name="paracetamol",
        manufacturer="Beximco Pharmaceuticals",
        strength="500 mg",
        dosage_form="Tablet / Syrup / Suppository",
        common_uses="Fever, headache, body ache, mild to moderate pain",
        bengali_generic="প্যারাসিটামল",
        bengali_uses="জ্বর, মাথাব্যথা, শরীর ব্যথা",
        otc=True,
        warning="Max 4,000 mg (8 tablets) in 24 hours. Avoid alcohol. In Dengue, use ONLY Paracetamol.",
    ),
    "napa extra": BangladeshDrugInfo(
        brand_name="Napa Extra",
        generic_name="paracetamol + caffeine",
        manufacturer="Beximco Pharmaceuticals",
        strength="500 mg + 65 mg",
        dosage_form="Tablet",
        common_uses="Severe headache, migraine, toothache, acute fever",
        bengali_generic="প্যারাসিটামল + ক্যাফেইন",
        bengali_uses="তীব্র মাথাব্যথা, মাইগ্রেন, দাঁতের ব্যথা",
        otc=True,
        warning="Contains caffeine. Avoid taking close to bedtime or with high-caffeine beverages.",
    ),
    "napa extend": BangladeshDrugInfo(
        brand_name="Napa Extend",
        generic_name="paracetamol",
        manufacturer="Beximco Pharmaceuticals",
        strength="665 mg sustained release",
        dosage_form="Extended Release Tablet",
        common_uses="Joint pain, osteoarthritis, persistent backache",
        bengali_generic="প্যারাসিটামল (এক্সটেন্ডেড রিলিজ)",
        bengali_uses="জয়েন্টের ব্যথা, দীর্ঘস্থায়ী শরীর ব্যথা",
        otc=True,
    ),
    "ace": BangladeshDrugInfo(
        brand_name="Ace",
        generic_name="paracetamol",
        manufacturer="Square Pharmaceuticals",
        strength="500 mg",
        dosage_form="Tablet / Syrup / Suspension",
        common_uses="Fever, pain relief, common cold symptoms",
        bengali_generic="প্যারাসিটামল",
        bengali_uses="জ্বর, ব্যথা উপশম",
        otc=True,
    ),
    "ace plus": BangladeshDrugInfo(
        brand_name="Ace Plus",
        generic_name="paracetamol + caffeine",
        manufacturer="Square Pharmaceuticals",
        strength="500 mg + 65 mg",
        dosage_form="Tablet",
        common_uses="Tension headache, migraine, toothache, fever",
        bengali_generic="প্যারাসিটামল + ক্যাফেইন",
        bengali_uses="তীব্র মাথাব্যথা, মাইগ্রেন",
        otc=True,
    ),
    "fast": BangladeshDrugInfo(
        brand_name="Fast",
        generic_name="paracetamol",
        manufacturer="Acme Laboratories",
        strength="500 mg",
        dosage_form="Tablet",
        common_uses="Fever, headache, general body pain",
        bengali_generic="প্যারাসিটামল",
        bengali_uses="জ্বর এবং ব্যথা",
        otc=True,
    ),
    "renova": BangladeshDrugInfo(
        brand_name="Renova",
        generic_name="paracetamol",
        manufacturer="Opso Saline",
        strength="500 mg",
        dosage_form="Tablet / Syrup",
        common_uses="Fever, analgesic",
        bengali_generic="প্যারাসিটামল",
        bengali_uses="জ্বর এবং ব্যথা",
        otc=True,
    ),

    # Gastric / Acid Suppressants (PPIs)
    "seclo": BangladeshDrugInfo(
        brand_name="Seclo",
        generic_name="omeprazole",
        manufacturer="Square Pharmaceuticals",
        strength="20 mg / 40 mg",
        dosage_form="Capsule / Injection",
        common_uses="Gastric ulcer, heartburn, acid reflux (GERD), indigestion",
        bengali_generic="ওমিপ্রাজল",
        bengali_uses="গ্যাস্ট্রিকের ব্যথা, বুকজ্বালা, অম্লতা",
        otc=True,
        warning="Take 30-60 minutes before breakfast or a meal on an empty stomach.",
    ),
    "losectil": BangladeshDrugInfo(
        brand_name="Losectil",
        generic_name="omeprazole",
        manufacturer="Eskayef Pharmaceuticals",
        strength="20 mg / 40 mg",
        dosage_form="Capsule",
        common_uses="Gastric acidity, peptic ulcer, acid reflux",
        bengali_generic="ওমিপ্রাজল",
        bengali_uses="গ্যাস্ট্রিক এবং পেপটিক আলসার",
        otc=True,
    ),
    "sergel": BangladeshDrugInfo(
        brand_name="Sergel",
        generic_name="esomeprazole",
        manufacturer="Healthcare Pharmaceuticals",
        strength="20 mg / 40 mg",
        dosage_form="Capsule / Tablet",
        common_uses="Acid reflux, erosive esophagitis, hyperacidity",
        bengali_generic="এসোমিপ্রাজল",
        bengali_uses="অ্যাসিড রিফ্লাক্স, বুকজ্বালা",
        otc=True,
    ),
    "nexum": BangladeshDrugInfo(
        brand_name="Nexum",
        generic_name="esomeprazole",
        manufacturer="Incepta Pharmaceuticals",
        strength="20 mg / 40 mg",
        dosage_form="Capsule / Tablet",
        common_uses="Severe acidity, GERD, gastric ulcer healing",
        bengali_generic="এসোমিপ্রাজল",
        bengali_uses="তীব্র অম্লতা, আলসার নিরাময়",
        otc=True,
    ),
    "pantonix": BangladeshDrugInfo(
        brand_name="Pantonix",
        generic_name="pantoprazole",
        manufacturer="Incepta Pharmaceuticals",
        strength="20 mg / 40 mg",
        dosage_form="Tablet / Injection",
        common_uses="Acidity, gastric ulcer, NSAID-induced ulcer prevention",
        bengali_generic="প্যান্টোপ্রাজল",
        bengali_uses="পেটের গ্যাস্ট্রিক এবং আলসার প্রতিরোধ",
        otc=True,
    ),
    "trupan": BangladeshDrugInfo(
        brand_name="Trupan",
        generic_name="pantoprazole",
        manufacturer="Beximco Pharmaceuticals",
        strength="20 mg / 40 mg",
        dosage_form="Tablet",
        common_uses="Acid-related stomach complaints",
        bengali_generic="প্যান্টোপ্রাজল",
        bengali_uses="গ্যাস্ট্রিকজনিত পেটের সমস্যা",
        otc=True,
    ),
    "finix": BangladeshDrugInfo(
        brand_name="Finix",
        generic_name="rabeprazole",
        manufacturer="Opsonin Pharma",
        strength="20 mg",
        dosage_form="Tablet",
        common_uses="Rapid relief of acid reflux and gastritis",
        bengali_generic="রেবেপ্রাজল",
        bengali_uses="গ্যাস্ট্রাইটিস এবং অম্লতা",
        otc=True,
    ),

    # Antihistamines / Allergy
    "alatrol": BangladeshDrugInfo(
        brand_name="Alatrol",
        generic_name="cetirizine",
        manufacturer="Square Pharmaceuticals",
        strength="10 mg",
        dosage_form="Tablet / Syrup",
        common_uses="Allergic rhinitis, sneezing, runny nose, itching, hives (urticaria)",
        bengali_generic="সেটিরিজিন",
        bengali_uses="অ্যালার্জি, হাঁচি, চুলকানি, নাক দিয়ে জল পড়া",
        otc=True,
        warning="May cause mild drowsiness. Avoid driving or alcohol after taking.",
    ),
    "cetriz": BangladeshDrugInfo(
        brand_name="Cetriz",
        generic_name="cetirizine",
        manufacturer="Incepta Pharmaceuticals",
        strength="10 mg",
        dosage_form="Tablet",
        common_uses="Cold allergy, skin allergy, itchiness",
        bengali_generic="সেটিরিজিন",
        bengali_uses="ঠান্ডা অ্যালার্জি এবং চুলকানি",
        otc=True,
    ),
    "fenadin": BangladeshDrugInfo(
        brand_name="Fenadin",
        generic_name="fexofenadine",
        manufacturer="Square Pharmaceuticals",
        strength="120 mg / 180 mg",
        dosage_form="Tablet / Suspension",
        common_uses="Non-sedating antihistamine for seasonal allergic rhinitis, chronic urticaria",
        bengali_generic="ফেক্সোফেনাডিন",
        bengali_uses="অ্যালার্জি, হাঁচি, ত্বকের চুলকানি (ঘুম হয় না)",
        otc=True,
    ),
    "fexo": BangladeshDrugInfo(
        brand_name="Fexo",
        generic_name="fexofenadine",
        manufacturer="Incepta Pharmaceuticals",
        strength="120 mg / 180 mg",
        dosage_form="Tablet",
        common_uses="Allergy, skin rash, allergic conjunctivitis",
        bengali_generic="ফেক্সোফেনাডিন",
        bengali_uses="মৌসুমি অ্যালার্জি এবং ত্বকের ফুসকুড়ি",
        otc=True,
    ),

    # Oral Rehydration & Diarrhea
    "orsaline": BangladeshDrugInfo(
        brand_name="Orsaline-N",
        generic_name="oral rehydration salts (ORS)",
        manufacturer="SMC (Social Marketing Company)",
        strength="Standard WHO formula",
        dosage_form="Powder packet",
        common_uses="Diarrhea, cholera, vomiting, dehydration, heat exhaustion",
        bengali_generic="খাবার স্যালাইন (ওআরএস)",
        bengali_uses="ডায়রিয়া, পাতলা পায়খানা, বমি, পানিশূন্যতা",
        otc=True,
        warning="CRITICAL: Dissolve 1 whole packet in exactly 500 mL (half liter) of clean drinking water. DO NOT boil after mixing. Discard after 12-24 hours.",
    ),
    "motigut": BangladeshDrugInfo(
        brand_name="Motigut",
        generic_name="domperidone",
        manufacturer="Square Pharmaceuticals",
        strength="10 mg",
        dosage_form="Tablet / Suspension",
        common_uses="Nausea, vomiting, bloating, gastric fullness",
        bengali_generic="ডম্পেরিডন",
        bengali_uses="বমি বমি ভাব, বমি, পেট ফাঁপা",
        otc=True,
        warning="Take 15-30 minutes before meals.",
    ),
    "omidon": BangladeshDrugInfo(
        brand_name="Omidon",
        generic_name="domperidone",
        manufacturer="Incepta Pharmaceuticals",
        strength="10 mg",
        dosage_form="Tablet",
        common_uses="Nausea, post-meal fullness, dyspepsia",
        bengali_generic="ডম্পেরিডন",
        bengali_uses="বমি ভাব এবং বদহজম",
        otc=True,
    ),

    # Antibiotics
    "zimax": BangladeshDrugInfo(
        brand_name="Zimax",
        generic_name="azithromycin",
        manufacturer="Square Pharmaceuticals",
        strength="500 mg",
        dosage_form="Tablet / Suspension",
        common_uses="Respiratory tract infections, throat infection (pharyngitis), pneumonia, typhoid",
        bengali_generic="অ্যাজিথ্রোমাইসিন",
        bengali_uses="শ্বাসতন্ত্রের সংক্রমণ, গলাব্যথা, নিউমোনিয়া, টাইফয়েড",
        otc=False,
        warning="Prescription only. Complete the full course (typically 3 to 5 days). Take 1 hour before or 2 hours after meals.",
    ),
    "tridosil": BangladeshDrugInfo(
        brand_name="Tridosil",
        generic_name="azithromycin",
        manufacturer="Beximco Pharmaceuticals",
        strength="500 mg",
        dosage_form="Tablet",
        common_uses="Bacterial infections of throat, chest, and skin",
        bengali_generic="অ্যাজিথ্রোমাইসিন",
        bengali_uses="গলা ও ফুসফুসের ব্যাকটেরিয়াল সংক্রমণ",
        otc=False,
    ),
    "cefix-3": BangladeshDrugInfo(
        brand_name="Cef-3",
        generic_name="cefixime",
        manufacturer="Square Pharmaceuticals",
        strength="200 mg / 400 mg",
        dosage_form="Capsule / Suspension",
        common_uses="Typhoid fever, urinary tract infection (UTI), bronchitis",
        bengali_generic="সেফিক্সিম",
        bengali_uses="টাইফয়েড জ্বর, প্রস্রাবে ইনফেকশন",
        otc=False,
    ),
    "ciprocin": BangladeshDrugInfo(
        brand_name="Ciprocin",
        generic_name="ciprofloxacin",
        manufacturer="Square Pharmaceuticals",
        strength="500 mg",
        dosage_form="Tablet / Eye Drop",
        common_uses="Bacterial diarrhea (dysentery), UTI, bone and soft tissue infection",
        bengali_generic="সিপ্রোফ্লক্সাসিন",
        bengali_uses="আমাশয়, মূত্রনালীর সংক্রমণ",
        otc=False,
    ),
    "moxacil": BangladeshDrugInfo(
        brand_name="Moxacil",
        generic_name="amoxicillin",
        manufacturer="Square Pharmaceuticals",
        strength="500 mg",
        dosage_form="Capsule / Syrup",
        common_uses="Ear-nose-throat infections, dental abscess, chest infections",
        bengali_generic="অ্যামোক্সিসিলিন",
        bengali_uses="দাঁতের ইনফেকশন, কান ও গলার সংক্রমণ",
        otc=False,
    ),

    # Asthma / Allergy / Respiratory
    "monas": BangladeshDrugInfo(
        brand_name="Monas",
        generic_name="montelukast",
        manufacturer="Acme Laboratories",
        strength="10 mg / 4 mg / 5 mg",
        dosage_form="Chewable Tablet / Tablet",
        common_uses="Asthma prevention, chronic allergic rhinitis, nocturnal breathing difficulty",
        bengali_generic="মন্টেলুকাস্ট",
        bengali_uses="অ্যাজমা, হাঁপানি এবং অ্যালার্জি প্রতিরোধ",
        otc=True,
        warning="Take once daily in the evening.",
    ),
    "montene": BangladeshDrugInfo(
        brand_name="Montene",
        generic_name="montelukast",
        manufacturer="Square Pharmaceuticals",
        strength="10 mg",
        dosage_form="Tablet",
        common_uses="Asthma control, dust allergy",
        bengali_generic="মন্টেলুকাস্ট",
        bengali_uses="হাঁপানি এবং শ্বাসকষ্ট নিয়ন্ত্রণ",
        otc=True,
    ),

    # Diabetes
    "comet": BangladeshDrugInfo(
        brand_name="Comet",
        generic_name="metformin",
        manufacturer="Square Pharmaceuticals",
        strength="500 mg / 850 mg",
        dosage_form="Tablet",
        common_uses="Type 2 Diabetes Mellitus management",
        bengali_generic="মেটফরমিন",
        bengali_uses="টাইপ ২ ডায়াবেটিস নিয়ন্ত্রণ",
        otc=False,
        warning="Take with meals to reduce gastrointestinal upset. Monitor kidney function (eGFR).",
    ),
}


# ── Canonical Generic Alias Mapping ───────────────────────────────────

# Map local brand names and common typos directly to canonical generic names
BANGLADESH_BRAND_TO_GENERIC: dict[str, str] = {
    # Paracetamol / Acetaminophen
    "napa": "acetaminophen",
    "napa extra": "acetaminophen",
    "napa extend": "acetaminophen",
    "napa rapid": "acetaminophen",
    "ace": "acetaminophen",
    "ace plus": "acetaminophen",
    "fast": "acetaminophen",
    "renova": "acetaminophen",
    "reset": "acetaminophen",
    "fevnil": "acetaminophen",
    "paracitemol": "acetaminophen",  # Typo support
    "paracetamol": "acetaminophen",
    "প্যারাসিটামল": "acetaminophen",
    "নাপা": "acetaminophen",
    "এইস": "acetaminophen",

    # Omeprazole / PPIs
    "seclo": "omeprazole",
    "losectil": "omeprazole",
    "omecon": "omeprazole",
    "সার্জেল": "omeprazole",
    "সেক্লো": "omeprazole",

    # Pantoprazole / Esomeprazole
    "pantonix": "pantoprazole",
    "trupan": "pantoprazole",
    "pantodac": "pantoprazole",
    "sergel": "esomeprazole",
    "nexum": "esomeprazole",
    "maxpro": "esomeprazole",

    # Antihistamines
    "alatrol": "cetirizine",
    "cetriz": "cetirizine",
    "fenadin": "fexofenadine",
    "fexo": "fexofenadine",
    "আলাট্রোল": "cetirizine",

    # ORS
    "orsaline": "oral rehydration salts",
    "orsaline-n": "oral rehydration salts",
    "saline": "oral rehydration salts",
    "ওরস্যালাইন": "oral rehydration salts",
    "স্যালাইন": "oral rehydration salts",

    # Antibiotics
    "zimax": "azithromycin",
    "tridosil": "azithromycin",
    "ciprocin": "ciprofloxacin",
    "moxacil": "amoxicillin",
    "cef-3": "cefixime",

    # Asthma
    "monas": "montelukast",
    "montene": "montelukast",

    # Diabetes
    "comet": "metformin",
    "informet": "metformin",
}


def lookup_bangladesh_brand(query: str) -> BangladeshDrugInfo | None:
    """Find a Bangladesh drug info record matching the query string."""
    q_lower = query.strip().lower()

    # 1. Check Bengali brand names first (longer names first)
    bengali_to_brand = {
        "নাপা এক্সট্রা": "napa extra",
        "নাপা": "napa",
        "এইস প্লাস": "ace plus",
        "এইস": "ace",
        "ফাস্ট": "fast",
        "রেনোভা": "renova",
        "সেকলো": "seclo",
        "সারজেল": "sergel",
        "সারজিল": "sergel",
        "প্যান্টোনিক্স": "pantonix",
        "প্যান্টোবেক্স": "pantobex",
        "আলাট্রোল": "alatrol",
        "অ্যালাট্রোল": "alatrol",
        "ওরস্যালাইন": "orsaline-n",
        "স্যালাইন": "orsaline-n",
        "মোনাস": "monas",
        "মন্টিন": "montene",
        "জিমেক্স": "zimax",
        "প্যারাসিটামল": "napa",
    }
    for bn_name, brand_key in sorted(bengali_to_brand.items(), key=lambda x: len(x[0]), reverse=True):
        if bn_name in q_lower:
            return BANGLADESH_BRANDS.get(brand_key)

    # 2. Check English brand keys (longer names first)
    sorted_brands = sorted(BANGLADESH_BRANDS.keys(), key=len, reverse=True)
    for brand_key in sorted_brands:
        if re.search(rf"\b{re.escape(brand_key)}\b", q_lower) or brand_key in q_lower:
            return BANGLADESH_BRANDS[brand_key]

    return None



def get_canonical_generic_from_local_query(query: str) -> str | None:
    """Extract canonical active ingredient from a query containing local Bangladeshi brands or typos."""
    q_lower = query.strip().lower()
    # Check multi-word brands first (e.g. napa extra)
    sorted_brands = sorted(BANGLADESH_BRAND_TO_GENERIC.keys(), key=len, reverse=True)
    for b in sorted_brands:
        if re.search(rf"\b{re.escape(b)}\b", q_lower):
            return BANGLADESH_BRAND_TO_GENERIC[b]
    return None


def is_emergency_query(query: str) -> tuple[bool, str]:
    """Detect if the query represents an acute life-threatening emergency requiring 999 or immediate hospital dispatch."""
    q_lower = query.lower()
    emergency_patterns = [
        (r"\b(snake\s*bite|সাপের\s*কামড়|বিষধর|russell's\s*viper|কালো\s*সাপ)\b", "snakebite"),
        (r"\b(chest\s*pain|heart\s*attack|বুকে\s*ব্যথা|হার্ট\s*অ্যাটাক|দম\s*বন্ধ)\b", "cardiac"),
        (r"\b(unconscious|fainted|unresponsive|অজ্ঞান|জ্ঞান\s*হারানো|পড়ে\s*গেছে|অচেতন)\b", "unconscious"),
        (r"\b(severe\s*burn|আগুনে\s*পুড়ে|third\s*degree\s*burn|বিদ্যুৎস্পৃষ্ট|electrocution)\b", "severe_burn"),
        (r"\b(dengue\s*bleeding|প্লাটিলেট\s*কমে|রক্তক্ষরণ|কালো\s*পায়খানা|dengue\s*shock|bleeding\s*heavily|massive\s*bleeding|hemorrhage|প্রচুর\s*রক্তপাত)\b", "severe_bleeding"),
        (r"\b(severe\s*choking|গলায়\s*আটকে|দম\s*নিতে\s*পারছে\s*না)\b", "choking"),
        (r"\b(poison|বিষ\s*খেয়েছে|কীটনাশক|হারপিক|bleach\s*poisoning)\b", "poisoning"),
    ]
    for pattern, reason in emergency_patterns:
        if re.search(pattern, q_lower):
            return True, reason
    return False, ""


def find_brand_alternatives(brand_or_generic: str) -> list[BangladeshDrugInfo]:
    """Find all alternative Bangladeshi pharmaceutical brands sharing the same generic molecule."""
    q_clean = brand_or_generic.strip().lower()
    target_generic = None

    # Check if query directly matches a brand
    info = lookup_bangladesh_brand(q_clean)
    if info:
        target_generic = info.generic_name.lower()
    else:
        # Check canonical mapping
        canonical = get_canonical_generic_from_local_query(q_clean)
        if canonical:
            target_generic = canonical.lower()
        else:
            # Check if query itself is generic name
            for drug in BANGLADESH_BRANDS.values():
                if drug.generic_name.lower() == q_clean:
                    target_generic = q_clean
                    break

    if not target_generic:
        return []

    alternatives: list[BangladeshDrugInfo] = []
    seen_brands: set[str] = set()

    for brand_key, drug in BANGLADESH_BRANDS.items():
        if drug.generic_name.lower() == target_generic and drug.brand_name.lower() not in seen_brands:
            seen_brands.add(drug.brand_name.lower())
            alternatives.append(drug)

    return alternatives

