"""First-Aid & Emergency Medical Triage Protocols for Bangladesh.

Provides structured, evidence-based first-aid steps compliant with WHO,
Bangladesh Red Crescent Society, and DGDA clinical safety guidelines.
Debunks common harmful home remedies (e.g., toothpaste on burns, tourniquets
on snakebites, NSAIDs in dengue fever).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class FirstAidProtocol:
    id: str
    title_en: str
    title_bn: str
    category: str
    urgency_level: str  # "EMERGENCY_CRITICAL" | "HIGH_PRIORITY" | "STANDARD_CARE"
    quick_summary_en: str
    quick_summary_bn: str
    dos_en: list[str]
    dos_bn: list[str]
    donts_en: list[str]
    donts_bn: list[str]
    red_flags_en: list[str]
    red_flags_bn: list[str]
    emergency_contacts: list[str]


FIRST_AID_PROTOCOLS: dict[str, FirstAidProtocol] = {
    "dengue": FirstAidProtocol(
        id="dengue",
        title_en="Dengue Fever Triage & Management",
        title_bn="ডেঙ্গু জ্বরের প্রাথমিক পরিচর্যা ও সতর্কতা",
        category="Infectious Disease / Fever",
        urgency_level="HIGH_PRIORITY",
        quick_summary_en="Use ONLY Paracetamol for fever. Drink plenty of fluids (ORS, coconut water). NEVER use NSAIDs (Aspirin, Ibuprofen, Napa Extra/Ace Plus) which trigger life-threatening internal bleeding.",
        quick_summary_bn="জ্বরের জন্য শুধুমাত্র প্যারাসিটামল ব্যবহার করুন। প্রচুর তরল ও ওরস্যালাইন খান। ভুল করেও এসপিরিন বা আইবুপ্রোফেন খাবেন না, এতে অভ্যন্তরীণ রক্তক্ষরণ হতে পারে।",
        dos_en=[
            "Take plain Paracetamol (Napa / Ace 500 mg) every 6-8 hours for high fever (maximum 4 grams per day for adults).",
            "Drink plenty of fluids: Oral Rehydration Salts (Orsaline-N), tender coconut water, soup, and fresh juices.",
            "Complete blood count (CBC) test by Day 3 to monitor platelet count and hematocrit.",
            "Use mosquito nets (মশারি) day and night to prevent spreading to family members.",
            "Rest completely in bed; avoid physical exertion.",
        ],
        dos_bn=[
            "জ্বরের জন্য শুধুমাত্র সাধারণ প্যারাসিটামল (নাপা/এইস ৫০০ মিগ্রা) ৬-৮ ঘণ্টা পরপর গ্রহণ করুন।",
            "প্রচুর তরল খাবার খান: ওরস্যালাইন (Orsaline-N), ডাবের পানি, স্যুপ এবং ফলের রস।",
            "জ্বরের ৩য় দিনে রক্তের সিবিসি (CBC) পরীক্ষা করে প্লাটিলেট কাউন্ট পর্যবেক্ষণ করুন।",
            "রোগীকে দিন-রাত মশারি টাঙিয়ে রাখুন যাতে অন্য সদস্যে ডেঙ্গু না ছড়ায়।",
        ],
        donts_en=[
            "DO NOT TAKE Aspirin, Ibuprofen, Diclofenac, Naproxen, or Napa Extra (Caffeine/NSAIDs) — they cause severe bleeding and platelet failure!",
            "DO NOT administer antibiotics — Dengue is a viral infection, antibiotics have zero effect.",
            "DO NOT inject intramuscular medicines.",
            "DO NOT ignore warning signs during the afebrile phase (when fever suddenly drops on Day 3-5; this is when shock occurs).",
        ],
        donts_bn=[
            "ভুল করেও অ্যাসপিরিন, আইবুপ্রোফেন, ডাইক্লোফেনাক বা ব্রুফেন জাতীয় ব্যথানাশক খাবেন না—এতে তীব্র রক্তক্ষরণ হতে পারে!",
            "অ্যান্টিবায়োটিক খাবেন না—ডেঙ্গু একটি ভাইরাসজনিত রোগ, অ্যান্টিবায়োটিকের কোনো ভূমিকা নেই।",
            "জ্বর কমার দিনগুলোতে বেশি সতর্ক থাকুন (৩য় থেকে ৫ম দিন ডেঙ্গু শক সিন্ড্রোমের ঝুঁকি বেশি থাকে)।",
        ],
        red_flags_en=[
            "Severe persistent abdominal pain or tenderness",
            "Persistent vomiting (unable to keep fluids down)",
            "Bleeding from gums, nose, vomit, or black tarry stools",
            "Extreme lethargy, confusion, or restlessness",
            "Cold, clammy skin or sudden drop in blood pressure",
        ],
        red_flags_bn=[
            "পেটে তীব্র ব্যথা বা ক্রমাগত বমি হওয়া",
            "দাঁতের মাড়ি বা নাক দিয়ে রক্তপাত, বমিতে রক্ত বা কালো পায়খানা",
            "অতিরিক্ত দুর্বলতা, অস্থিরতা বা হাত-পা ঠান্ডা হয়ে যাওয়া",
            "প্রস্রাবের পরিমাণ মারাত্মক কমে যাওয়া (৬ ঘণ্টার বেশি প্রস্রাব না হওয়া)",
        ],
        emergency_contacts=["999 (National Ambulance)", "16263 (Shastho Batayan 24/7 Health Doctor)"],
    ),
    "burns": FirstAidProtocol(
        id="burns",
        title_en="Burns & Scalds First Aid",
        title_bn="আগুনে বা গরম পানিতে পোড়ার প্রাথমিক চিকিৎসা",
        category="Trauma / Burns",
        urgency_level="HIGH_PRIORITY",
        quick_summary_en="Immediately cool the burn under gently running cool tap water for 15-20 minutes. NEVER apply toothpaste, egg, butter, or ice.",
        quick_summary_bn="পোড়া জায়গায় সাথে সাথে স্বাভাবিক তাপমাত্রার চলমান ট্যাপের পানি অন্তত ১৫-২০ মিনিট ঢালুন। টুথপেস্ট, ডিম, মাখন বা বরফ লাগাবেন না।",
        dos_en=[
            "Cool the burn IMMEDIATELY under cool running tap water for at least 15 to 20 minutes.",
            "Carefully remove rings, watches, or tight clothing around the burned area before swelling starts.",
            "Cover loosely with a clean, lint-free cloth or sterile plastic cling wrap.",
            "Give plain Paracetamol for acute pain relief.",
            "Seek immediate medical care for burns larger than the patient's palm, on face, hands, joints, or genitals.",
        ],
        dos_bn=[
            "অবিলম্বে পোড়া অংশে স্বাভাবিক তাপমাত্রার পরিষ্কার চলমান পানি অন্তত ১৫ থেকে ২০ মিনিট ঢালুন।",
            "পোড়া অংশের আশেপাশের আংটি, ঘড়ি বা টাইট কাপড় দ্রুত খুলে ফেলুন (ফোলার আগেই)।",
            "পরিষ্কার পাতলা সুতি কাপড় বা প্লাস্টিক র‍্যাপ দিয়ে আলতো করে ঢেকে রাখুন।",
            "ব্যথা কমানোর জন্য প্যারাসিটামল খেতে পারেন।",
        ],
        donts_en=[
            "DO NOT apply toothpaste, raw egg, butter, flour, or turmeric — these trap heat and cause severe bacterial infection!",
            "DO NOT use direct ice or iced water — ice causes secondary tissue necrosis and frostbite.",
            "DO NOT burst or pop blisters — the intact blister skin is the body's sterile barrier.",
            "DO NOT peel off clothing that is melted or stuck to the burn tissue.",
        ],
        donts_bn=[
            "টুথপেস্ট, ডিমের সাদা অংশ, মাখন, আটা বা হলুদ লাগাবেন না—এতে ইনফেকশনের ঝুঁকি বাড়ে!",
            "বরফ বা বরফ-শীতল পানি ব্যবহার করবেন না—এতে চামড়ার কোষ ধ্বংস হয়ে যায়।",
            "ফোসকা (Blister) গলাবেন না বা খোঁচাবেন না।",
            "পোড়া অংশে আটকে যাওয়া কাপড় জোর করে টান দিয়ে খুলবেন না।",
        ],
        red_flags_en=[
            "Burns involving the face, airway, hands, feet, groin, or major joints",
            "Deep burns where skin looks charred, white, or leather-like (3rd degree)",
            "Electrical burns or chemical burns (deep internal tissue damage)",
            "Burns covering more than 10% of the body surface area",
        ],
        red_flags_bn=[
            "মুখমণ্ডল, শ্বাসনালী, হাত, পা বা যৌনাঙ্গের পোড়া",
            "চামড়া পুড়ে সাদা, শক্ত বা কয়লার মতো কালো হয়ে যাওয়া",
            "বৈদ্যুতিক পোড়া বা রাসায়নিক অ্যাসিড/ক্ষার দ্বারা পোড়া",
        ],
        emergency_contacts=["Sheikh Hasina National Institute of Burn and Plastic Surgery (SHNIBPS): 02-41052601", "999"],
    ),
    "snakebite": FirstAidProtocol(
        id="snakebite",
        title_en="Snakebite Emergency First Aid",
        title_bn="সাপের কামড়ের জরুরি প্রাথমিক চিকিৎসা",
        category="Toxicology / Envenomation",
        urgency_level="EMERGENCY_CRITICAL",
        quick_summary_en="Keep the victim calm and strictly STILL. Immobilize the limb with a splint. NEVER cut, suck venom, or tie tight rope tourniquets. Rush to Upazila Health Complex / Medical College Hospital immediately for Anti-Venom (AVS).",
        quick_summary_bn="রোগীকে শান্ত ও সম্পূর্ণ স্থির রাখুন। কামড়ানো হাত/পা নড়াচড়া করতে দেবেন না (স্প্লিন্ট দিয়ে বাঁধুন)। ব্লেড দিয়ে কাটা, মুখ দিয়ে রক্ত টানা বা শক্ত রশি দিয়ে গিঁট দেওয়া যাবে না। দ্রুত উপজেলা স্বাস্থ্য কমপ্লেক্স বা হাসপাতালে অ্যান্টি-ভেনমের জন্য যান।",
        dos_en=[
            "Keep the patient calm, reassure them, and demand they stay completely motionless (movement accelerates venom spread).",
            "Immobilize the bitten limb using a rigid splint (stick or bamboo strip) and broad crepe bandage, exactly like a bone fracture.",
            "Remove rings, bangles, and tight clothing from the bitten limb immediately before swelling develops.",
            "Transport the patient on a stretcher, motorcycle, or vehicle to the nearest hospital with Anti-Venom Serum (AVS).",
            "Note the snake's color and head shape if safely seen, or photograph it from a safe distance.",
        ],
        dos_bn=[
            "রোগীকে কোনোভাবেই দৌড়াদৌড়ি বা হাঁটাহাঁটি করতে দেবেন না, সম্পূর্ণ স্থির রাখুন।",
            "কামড়ানো অঙ্গটি ভাঙা হাড়ের মতো বাঁশের চটা বা শক্ত কাঠি এবং কাপড় দিয়ে নড়াচড়াহীন করে রাখুন।",
            "হাতে বা পায়ে থাকা আংটি, চুড়ি বা টাইট কাপড় অবিলম্বে খুলে ফেলুন।",
            "দ্রুত নিকটস্থ উপজেলা স্বাস্থ্য কমপ্লেক্স বা জেলা সদর হাসপাতালে নিয়ে যান যেখানে অ্যান্টি-ভেনম পাওয়া যায়।",
        ],
        donts_en=[
            "DO NOT cut the wound with a blade or knife — causes hemorrhage and tetanus!",
            "DO NOT suck venom with mouth — ineffective and poisons the rescuer.",
            "DO NOT tie tight cord, wire, or bicycle tube tourniquets — cuts arterial blood flow and leads to gangrene and amputation!",
            "DO NOT visit traditional healers (ওঝা/Ozha) — delays critical antivenom administration.",
            "DO NOT give aspirin or pain medicines that increase bleeding.",
        ],
        donts_bn=[
            "ব্লেড বা ছুরি দিয়ে কামড়ানো স্থান কাটবেন না!",
            "মুখ দিয়ে বিষ চুষে বের করার চেষ্টা করবেন না!",
            "দড়ি, তার বা সাইকেলের টিউব দিয়ে শক্ত করে বাঁধবেন না—এতে রক্ত চলাচল বন্ধ হয়ে পচন ধরে অঙ্গ কেটে ফেলতে হয়!",
            "ওঝা বা কবিরাজের কাছে নিয়ে সময় নষ্ট করবেন না—অ্যান্টি-ভেনম ছাড়া বিষের কোনো কার্যকর চিকিৎসা নেই!",
        ],
        red_flags_en=[
            "Difficulty speaking, swallowing, or drooping eyelids (ptosis - neurotoxic)",
            "Bleeding from bite marks, gums, or skin (Russell's viper hematotoxicity)",
            "Severe swelling spreading rapidly up the limb",
            "Difficulty breathing or respiratory paralysis",
        ],
        red_flags_bn=[
            "চোখের পাতা ভারী হয়ে আসা, কথা বলতে বা গিলতে সমস্যা (স্নায়বিক বিষ)",
            "কামড়ের জায়গা বা দাঁতের মাড়ি দিয়ে রক্তক্ষরণ (রাসেলস ভাইপার / চন্দ্রবোড়ার লক্ষণ)",
            "শ্বাসকষ্ট বা নিস্তেজ হয়ে পড়া",
        ],
        emergency_contacts=["01700-700700 (National Poison Center, DMC)", "999 (Emergency Ambulance)", "Nearest Upazila Health Complex"],
    ),
    "diarrhea": FirstAidProtocol(
        id="diarrhea",
        title_en="Acute Diarrhea & Dehydration (ORS Protocol)",
        title_bn="তীব্র ডায়রিয়া ও পাতলা পায়খানায় ওরস্যালাইন প্রস্তুত প্রণালী",
        category="Gastrointestinal / Dehydration",
        urgency_level="STANDARD_CARE",
        quick_summary_en="Prevent dehydration with properly prepared Orsaline-N: dissolve 1 full packet in exactly 500 mL of clean water. Drink after every loose stool.",
        quick_summary_bn="পানিশূন্যতা রোধে সঠিকভাবে খাবার স্যালাইন বানান: আধা লিটার (৫০০ মিলি) পরিষ্কার পানিতে ১ প্যাকেট ওরস্যালাইন সম্পূর্ণ গুলিয়ে নিন। প্রতিবার পাতলা পায়খানার পর খান।",
        dos_en=[
            "Prepare Orsaline-N correctly: 1 entire sachet dissolved into exactly 500 mL (two standard glasses) of drinking water.",
            "Drink 1/2 to 1 glass of prepared ORS after EVERY loose bowel movement.",
            "For children under 5: Give Zinc tablet (20 mg) daily for 10 to 14 consecutive days to heal intestinal mucosa.",
            "Continue normal feeding: breast milk for infants, rice porridge (জাউ ভাত), green bananas, boiled potatoes.",
            "Discard any leftover prepared ORS after 12 to 24 hours and prepare a fresh batch.",
        ],
        dos_bn=[
            "ওরস্যালাইন বানানোর সঠিক নিয়ম: পুরো ১ প্যাকেট ওরস্যালাইন ঠিক ৫০০ মিলি (আধা লিটার বা ২ পোয়া) বিশুদ্ধ পানিতে মিশিয়ে নিন।",
            "প্রতিবার পাতলা পায়খানা হওয়ার পর আধা গ্লাস থেকে ১ গ্লাস স্যালাইন পান করুন।",
            "শিশুদের ক্ষেত্রে টানা ১০-১৪ দিন জিংক (Zinc) সিরাপ বা ট্যাবলেট খাওয়াতে হবে।",
            "স্বাভাবিক খাবার (বুকের দুধ, কাঁচাকলার জাউ ভাত) চালিয়ে যান।",
            "বানানো স্যালাইন ১২ ঘণ্টার বেশি পুরনো হলে ফেলে দিয়ে নতুন স্যালাইন বানান।",
        ],
        donts_en=[
            "DO NOT mix half a packet in half a glass of water — incorrect concentration causes fatal electrolyte imbalance!",
            "DO NOT boil or heat the water AFTER mixing the ORS powder.",
            "DO NOT stop breastfeeding or feeding infants.",
            "DO NOT take anti-motility drugs (like Loperamide) without a physician's advice if there is bloody diarrhea.",
        ],
        donts_bn=[
            "অল্প পানিতে আধা প্যাকেট স্যালাইন গুলাবেন না—এতে লবণের ঘনত্ব বেড়ে শিশুর মৃত্যুর ঝুঁকি থাকে!",
            "স্যালাইন গুলানোর পর পানি ফুটানো যাবে না।",
            "শিশুর বুকের দুধ বা খাবার বন্ধ করবেন না।",
            "রক্ত আমাশয়ে চিকিৎসকের পরামর্শ ছাড়া নিজে নিজে ওষুধ খাবেন না।",
        ],
        red_flags_en=[
            "Severe dehydration: sunken eyes, dry mouth, skin pinch goes back very slowly (>2 seconds)",
            "Inability to drink or persistent vomiting",
            "High fever accompanied by blood or mucus in stool (Dysentery)",
            "No urination for more than 6-8 hours",
        ],
        red_flags_bn=[
            "তীব্র পানিশূন্যতা: চোখ গর্তে ঢুকে যাওয়া, জিহ্বা শুকিয়ে যাওয়া, চামড়া টানলে আস্তে আস্তে স্বাভাবিক হওয়া",
            "রোগী অতিরিক্ত দুর্বল বা অচেতন হয়ে পড়া",
            "মলের সাথে রক্ত যাওয়া বা তীব্র জ্বর",
            "৬-৮ ঘণ্টার বেশি সময় প্রস্রাব বন্ধ থাকা",
        ],
        emergency_contacts=["ICDDR,B (Cholera Hospital, Mohakhali): 02-9827001", "16263", "999"],
    ),
    "bleeding": FirstAidProtocol(
        id="bleeding",
        title_en="Severe Bleeding & Wound Management",
        title_bn="রক্তপাত ও জখমের প্রাথমিক চিকিৎসা",
        category="Trauma / Wound Care",
        urgency_level="HIGH_PRIORITY",
        quick_summary_en="Apply direct, continuous, firm pressure over the wound using a clean cloth or sterile gauze for 10 minutes without lifting. Elevate above heart level.",
        quick_summary_bn="পরিষ্কার কাপড় বা গজ দিয়ে কাটার ওপর একটানা শক্তভাবে ১০ মিনিট চেপে ধরে রাখুন। হাত বা পা হলে হৃৎপিণ্ডের চেয়ে উঁচুতে তুলে রাখুন।",
        dos_en=[
            "Apply direct, uninterrupted pressure directly on the bleeding site with a sterile gauze or clean cloth for at least 10 minutes.",
            "If blood soaks through, add another cloth on top — DO NOT lift the first layer (disrupts clot formation).",
            "Elevate the injured limb above heart level if no fracture is suspected.",
            "Bandage securely once bleeding slows down.",
            "Receive a Tetanus Toxoid (TT) injection within 48 hours if not vaccinated in the past 5-10 years.",
        ],
        dos_bn=[
            "কাটা স্থানে পরিষ্কার কাপড় বা গজ চেপে ধরে একটানা ১০ মিনিট চাপ দিয়ে রাখুন।",
            "কাপড় রক্তে ভিজে গেলে সেটা না সরিয়ে তার ওপর আরও কাপড় দিয়ে চাপ দিন।",
            "আহত হাত বা পা হৃৎপিণ্ডের চেয়ে উঁচুতে রাখুন।",
            "রক্তপাত বন্ধ হলে পরিষ্কার ব্যান্ডেজ দিয়ে বেঁধে দিন।",
            "মরচে ধরা লোহা বা নোংরা বস্তু দিয়ে কাটলে ৪৮ ঘণ্টার মধ্যে টিটেনাস (টিটি) টিকা নিন।",
        ],
        donts_en=[
            "DO NOT remove embedded foreign objects (e.g., deep knife, glass, metal shard) — leave in place and pad around it to avoid massive hemorrhage!",
            "DO NOT apply soil, cow dung, coffee powder, or kerosene to the wound.",
            "DO NOT lift the dressing repeatedly to check if bleeding has stopped.",
        ],
        donts_bn=[
            "শরীরে কোনো ছুরি, কাচ বা লোহার টুকরো গেঁথে থাকলে তা নিজে টেনে বের করবেন না—হাসপাতালে নিন!",
            "কাটা ঘায়ে মাটি, চুন বা কেরোসিন লাগাবেন না—এতে তীব্র ধনুষ্টঙ্কার (টিটেনাস) ও ইনফেকশন হয়।",
            "বারবার কাপড় তুলে রক্ত বন্ধ হয়েছে কিনা তা পরীক্ষা করবেন না।",
        ],
        red_flags_en=[
            "Blood spurting rhythmically in pulses (arterial bleed)",
            "Bleeding fails to stop after 15 minutes of direct continuous pressure",
            "Victim feels faint, dizzy, pale, cold, or loses consciousness (hemorrhagic shock)",
        ],
        red_flags_bn=[
            "রক্ত ফিনকি দিয়ে বের হওয়া (ধমনী কাটা পড়ার লক্ষণ)",
            "১৫ মিনিট একটানা চেপে রাখার পরও রক্তপাত বন্ধ না হওয়া",
            "রক্তক্ষরণে রোগী ফ্যাকাশে হয়ে যাওয়া বা অজ্ঞান হয়ে পড়া",
        ],
        emergency_contacts=["999", "National Institute of Traumatology & Rehabilitation (NITOR / পঙ্গু হাসপাতাল): 02-9144190"],
    ),
    "choking": FirstAidProtocol(
        id="choking",
        title_en="Choking First Aid (Heimlich Maneuver)",
        title_bn="শ্বাসনালীতে খাবার বা বস্তু আটকে দম বন্ধ হওয়া (হেইমলিক ম্যানুভার)",
        category="Airway Emergency",
        urgency_level="EMERGENCY_CRITICAL",
        quick_summary_en="If the person cannot speak or cough, stand behind them, wrap your arms around their waist, and deliver 5 upward abdominal thrusts (Heimlich maneuver).",
        quick_summary_bn="যদি ব্যক্তি কথা বলতে বা কাশতে না পারে, তবে তার পেছনে দাঁড়িয়ে পেটের ওপর দুই হাত দিয়ে ভেতরের ও ওপরের দিকে ৫ বার চাপ (Heimlich Maneuver) দিন।",
        dos_en=[
            "Encourage them to cough forcefully if they are able to cough or speak.",
            "If unable to breathe, speak, or cough: Stand behind them, lean them forward, and give 5 firm back blows between shoulder blades.",
            "If back blows fail: Make a fist just above the navel, grasp with other hand, and give 5 quick inward and upward abdominal thrusts.",
            "For infants (<1 year): Place facedown on your forearm supporting head, give 5 back blows, then turn face-up and give 5 chest thrusts with 2 fingers.",
            "Call 999 immediately if the obstruction is not expelled.",
        ],
        dos_bn=[
            "রোগী যদি কথা বলতে বা কাশতে পারে, তবে তাকে জোরে জোরে কাশতে উৎসাহিত করুন।",
            "যদি দম নিতে বা কথা বলতে না পারে: পেছনে দাঁড়িয়ে একটু সামনে ঝুঁকিয়ে দুই কাঁধের মাঝখানে হাতের তালু দিয়ে ৫ বার চাপড় দিন।",
            "এতে না নামলে: নাভির ঠিক ওপরে এক হাত মুষ্টিবদ্ধ করে ওপরের দিকে পেটে ৫ বার দ্রুত চাপ (Abdominal thrusts) দিন।",
            "শিশুদের ক্ষেত্রে: কোলে উপুড় করে পিঠে ৫ বার আলতো চাপড় এবং উল্টে বুকে ২ আঙুল দিয়ে ৫ বার চাপ দিন।",
        ],
        donts_en=[
            "DO NOT perform blind finger sweeps in the mouth — pushes the object deeper into the larynx!",
            "DO NOT slap the back of a person who is sitting upright (can cause the object to slip further down).",
            "DO NOT offer water to drink while choking.",
        ],
        donts_bn=[
            "আন্দাজে আঙুল দিয়ে মুখের ভেতর খোঁটাখুঁটি করবেন না—এতে বস্তু আরও গভীরে চলে যেতে পারে!",
            "শ্বাস বন্ধ থাকা অবস্থায় পানি পান করাবেন না।",
        ],
        red_flags_en=[
            "Inability to speak, vocalize, or cough",
            "Lips, ears, or fingernails turning blue (cyanosis - oxygen deprivation)",
            "Loss of consciousness (start CPR immediately)",
        ],
        red_flags_bn=[
            "কথা বলতে বা আওয়াজ করতে সম্পূর্ণ অক্ষম হওয়া",
            "ঠোঁট বা নখ নীলচে হয়ে যাওয়া (অক্সিজেন স্বল্পতা)",
            "অজ্ঞান হয়ে পড়া (অবিলম্বে সিপিআর শুরু করুন)",
        ],
        emergency_contacts=["999", "16263"],
    ),
    "fracture": FirstAidProtocol(
        id="fracture",
        title_en="Broken Bone, Fracture & Sprain First Aid",
        title_bn="হাড় ভাঙা, ফ্র্যাকচার ও মচকে যাওয়ার জরুরি প্রাথমিক চিকিৎসা",
        category="Trauma / Orthopedics",
        urgency_level="EMERGENCY_CRITICAL",
        quick_summary_en="DO NOT attempt to push, realign, or 'fix' a broken bone yourself! Keep the limb completely STILL, immobilize with a splint/padding, apply ice wrapped in cloth, and transport immediately to NITOR (পঙ্গু হাসপাতাল) or nearest hospital.",
        quick_summary_bn="কখনোই নিজে নিজে ভাঙা হাড় সোজা করার বা 'জোড়া লাগানোর' চেষ্টা করবেন না! আক্রান্ত অঙ্গটি যেভাবে আছে সেভাবেই কাঠি বা কাপড় দিয়ে বেঁধে স্থির রাখুন (Immobilize)। বরফের সেঁক দিন এবং অবিলম্বে পঙ্গু হাসপাতাল (NITOR) বা হাসপাতালে নিন।",
        dos_en=[
            "CRITICAL: Keep the injured limb completely STILL in the exact position it was found.",
            "Immobilize the fracture site using a rigid splint (cardboard, wood, folded newspaper) extending beyond the joints above and below the break.",
            "Apply an ice pack or cold compress wrapped in a towel for 15-20 minutes to reduce acute swelling and pain.",
            "If an open fracture (bone pierced through skin): Cover the wound with a clean sterile cloth. Do NOT push the bone back in!",
            "Call 999 or transport carefully to the National Institute of Traumatology & Orthopaedics (NITOR / পঙ্গু হাসপাতাল) or nearest hospital.",
        ],
        dos_bn=[
            "ভাঙা হাত বা পা যেভাবে আছে সেভাবেই স্থির রাখুন, কোনো অবস্থাতেই জোরাজুরি করবেন না।",
            "ভাঙা অংশের ওপরে ও নিচে শক্ত কাঠি, স্কেল বা পিচবোর্ড দিয়ে আলতো করে বেঁধে দিন (Splinting)।",
            "তোয়ালেতে বরফ পেঁচিয়ে আলতো করে সেঁক দিন যাতে ফোলা ও ব্যথা কমে।",
            "হাড় চামড়া ভেদ করে বাইরে বের হয়ে থাকলে পরিষ্কার কাপড় দিয়ে ঢেকে রাখুন, কখনোই হাড় ভেতরে ঢোকানোর চেষ্টা করবেন না!",
            "দ্রুত জাতীয় অর্থোপেডিক হাসপাতাল (পঙ্গু হাসপাতাল - NITOR) বা নিকটস্থ হাসপাতালে নিন।",
        ],
        donts_en=[
            "DO NOT attempt to push, force, massage, or straighten a broken or dislocated bone!",
            "DO NOT visit untrained bone-setters (হাড্ডি কবিরাজ/হাতুড়ে) who forcefully twist or massage fractured limbs — causes permanent disability and nerve damage!",
            "DO NOT allow the patient to walk or bear weight on a suspected leg/ankle/hip fracture.",
            "DO NOT apply hot packs or massage immediately after injury.",
        ],
        donts_bn=[
            "ভুল করেও ভাঙা হাড় টেনে সোজা করার বা মালিশ করার চেষ্টা করবেন না!",
            "হাতুড়ে কবিরাজ বা তথাকথিত 'হাড় জোড়া দেওয়া কবিরাজ'-এর কাছে যাবেন না—এতে পঙ্গু হওয়ার মারাত্মক ঝুঁকি থাকে!",
            "পায়ে আঘাত থাকলে রোগীকে হাঁটানোর চেষ্টা করবেন না।",
            "আঘাতের পরপরই গরম সেঁক বা তেল মালিশ করবেন না।",
        ],
        red_flags_en=[
            "Bone piercing through the skin or visible deformity/unnatural angle",
            "Numbness, tingling, pale/blue skin, or coldness in fingers or toes below the break",
            "Suspected neck, spine, pelvis, or skull injury (DO NOT move patient, call 999 immediately)",
            "Intolerable, excruciating pain or rapid severe swelling",
        ],
        red_flags_bn=[
            "হাড় চামড়া ভেদ করে বাইরে বের হওয়া বা অঙ্গ অস্বাভাবিকভাবে বেঁকে থাকা",
            "আক্রান্ত অংশের নিচের আঙুল নীল বা ফ্যাকাশে হয়ে যাওয়া ও অবশ লাগা (রক্ত চলাচল বন্ধের লক্ষণ)",
            "ঘাড়, মেরুদণ্ড বা কোমরে আঘাত (রোগীকে নড়াচড়া করাবেন না, অবিলম্বে ৯৯৯-এ কল দিন)",
        ],
        emergency_contacts=[
            "National Institute of Traumatology & Orthopaedic Rehabilitation (NITOR / পঙ্গু হাসপাতাল): 02-9144190",
            "999 (National Ambulance)",
            "16263 (Shastho Batayan)",
        ],
    ),
}


def find_first_aid_protocol(query: str) -> FirstAidProtocol | None:
    """Identify if the user query asks about a known first-aid protocol."""
    q_lower = query.lower()
    mapping = [
        (r"\b(dengue|ডেঙ্গু|প্লেটলেট|platelet|dengue\s*fever|haddi\s*bhanga\s*jor)\b", "dengue"),
        (r"\b(burn|burns|scalding|পুড়ে|পোড়া|আগুনে|গরম\s*পানি|blister)\b", "burns"),
        (r"\b(snake|snakebite|সাপ|সাপের\s*কামড়|বিষধর|রাসেলস|চন্দ্রবোড়া|গোখরো)\b", "snakebite"),
        (r"\b(diarrhea|cholera|ডায়রিয়া|পাতলা\s*পায়খানা|ওরস্যালাইন|orsaline|dehydration|কলেরা)\b", "diarrhea"),
        (r"\b(bleed|bleeding|cut|রক্তপাত|কেটে\s*গেছে|কাটা|জখম|ক্ষত|wound|bandage)\b", "bleeding"),
        (r"\b(chok|choking|দম\s*বন্ধ|গলায়\s*আটকে|heimlich|শ্বাস\s*আটকে)\b", "choking"),
        (r"\b(bone|bones|fracture|fractures|fractured|broken\s*bone|broken\s*arm|broken\s*leg|broken\s*hand|broken\s*wrist|broken\s*ankle|haddi|হাড়|হাড়\s*ভাঙা|হাড়\s*ভেঙে|মচকে|dislocation|dislocated|sprain|sprained|twisted\s*ankle|পঙ্গু|orthopedic)\b", "fracture"),
    ]
    for pattern, protocol_key in mapping:
        if re.search(pattern, q_lower):
            return FIRST_AID_PROTOCOLS.get(protocol_key)
    return None
