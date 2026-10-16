"""Report on downloaded drug label completeness.

Prints a per-drug table showing which of the 5 sections are present,
and flags any drug with fewer than 4 sections populated.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import LABELS_DIR

SECTIONS = ["indications", "dosage", "warnings", "interactions", "contraindications"]
SHORT = {"indications": "IND", "dosage": "DOS", "warnings": "WRN",
         "interactions": "INT", "contraindications": "CTR"}


def generate_report(labels_dir: Path = LABELS_DIR) -> dict:
    """Scan all label JSONs and print a completeness report.

    Returns a summary dict with per-drug details and totals.
    """
    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        print("No label files found in", labels_dir)
        return {"drugs": [], "total": 0, "total_missing": 0, "flagged": 0,
                "flagged_drugs": []}

    # Header
    col = "  ".join(f"{SHORT[s]:>3}" for s in SECTIONS)
    header = f"{'Drug':<25}  {col}   #/5  Flag"
    sep = "-" * len(header)
    print(header)
    print(sep)

    drugs: list[dict] = []
    total_missing = 0
    flagged_drugs: list[str] = []

    for fpath in label_files:
        data = json.loads(fpath.read_text(encoding="utf-8"))
        drug_name = data["drug_name"]
        sections = data.get("sections", {})

        present: list[str] = []
        marks: list[str] = []
        for s in SECTIONS:
            if sections.get(s) is not None:
                marks.append("  Y")
                present.append(s)
            else:
                marks.append("  -")
                total_missing += 1

        count = len(present)
        flag = " <<" if count < 4 else ""
        row = f"{drug_name:<25}  {'  '.join(marks)}   {count}/5{flag}"
        print(row)

        drugs.append({
            "drug_name": drug_name,
            "present": present,
            "count": count,
            "flagged": count < 4,
        })
        if count < 4:
            flagged_drugs.append(drug_name)

    # Summary
    print(sep)
    print(f"Total drugs:            {len(drugs)}")
    print(f"Total missing sections: {total_missing}")
    print(f"Flagged (< 4 sections): {len(flagged_drugs)}")
    if flagged_drugs:
        for d in flagged_drugs:
            drug_info = next(x for x in drugs if x["drug_name"] == d)
            missing = [s for s in SECTIONS if s not in drug_info["present"]]
            print(f"  >> {d}: missing {', '.join(missing)}")

    return {
        "drugs": drugs,
        "total": len(drugs),
        "total_missing": total_missing,
        "flagged": len(flagged_drugs),
        "flagged_drugs": flagged_drugs,
    }


if __name__ == "__main__":
    generate_report()
