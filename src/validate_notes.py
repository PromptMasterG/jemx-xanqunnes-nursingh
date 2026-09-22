"""Validation for the note classifier: a hand-labelled sample checked
against both the plain keyword classifier and the fuzzy-fallback version.

The 63-note sample below was labelled by reading every note and its
context (not by re-running any rule) and is stratified across all seven
categories, oversampling other_operational deliberately because that is
where a keyword classifier's mistakes concentrate. See NOTES.md for what
this found.
"""
from __future__ import annotations

import pandas as pd

# shift_id -> hand-labelled true category
HAND_LABELS: dict[str, str] = {
    "S104644": "no_signal",
    "S103599": "late_handover",
    "S104005": "relief_no_show",
    "S106818": "no_signal",
    "S104413": "staff_leave_covering",
    "S106467": "client_requested",
    "S105217": "late_handover",
    "S106394": "client_requested",
    "S109304": "equipment_failure",
    "S102185": "equipment_failure",
    "S101555": "client_requested",
    "S104188": "relief_no_show",
    "S107191": "staff_leave_covering",
    "S103251": "no_signal",
    "S103726": "staff_leave_covering",
    "S103391": "relief_no_show",
    "S106130": "client_requested",
    "S102397": "relief_no_show",
    "S103077": "relief_no_show",
    "S101101": "equipment_failure",
    "S100716": "staff_leave_covering",
    "S103065": "late_handover",
    "S100315": "relief_no_show",
    "S104812": "staff_leave_covering",
    "S101019": "late_handover",
    "S107989": "relief_no_show",
    "S101551": "relief_no_show",
    "S106213": "relief_no_show",
    "S103026": "relief_no_show",
    "S107717": "relief_no_show",
    "S102143": "no_signal",
    "S107873": "staff_leave_covering",
    "S104784": "equipment_failure",
    "S109351": "late_handover",
    "S108359": "equipment_failure",
    "S102309": "no_signal",
    "S109257": "relief_no_show",
    "S106100": "late_handover",
    "S106105": "client_requested",
    "S105143": "equipment_failure",
    "S102234": "no_signal",
    "S103560": "no_signal",
    "S103769": "staff_leave_covering",
    "S108964": "equipment_failure",
    "S108079": "late_handover",
    "S101597": "staff_leave_covering",
    "S104029": "late_handover",
    "S103269": "no_signal",
    "S100812": "client_requested",
    "S108349": "client_requested",
    "S105683": "client_requested",
    "S107743": "staff_leave_covering",
    "S105053": "client_requested",
    "S102351": "late_handover",
    "S108797": "client_requested",
    "S103981": "equipment_failure",
    "S106021": "client_requested",
    "S108104": "equipment_failure",
    "S101805": "equipment_failure",
    "S100159": "equipment_failure",
    "S106925": "equipment_failure",
    "S108830": "client_requested",
    "S107980": "relief_no_show",
}


def run_validation(notes: pd.DataFrame) -> dict:
    from classify_notes import classify_notes, classify_notes_with_fuzzy_fallback

    plain = classify_notes(notes).set_index("shift_id")
    fuzzy = classify_notes_with_fuzzy_fallback(notes).set_index("shift_id")

    rows = []
    for shift_id, true_cat in HAND_LABELS.items():
        rows.append(
            {
                "shift_id": shift_id,
                "note": plain.loc[shift_id, "note"],
                "true_category": true_cat,
                "keyword_category": plain.loc[shift_id, "category"],
                "fuzzy_category": fuzzy.loc[shift_id, "category"],
            }
        )
    df = pd.DataFrame(rows)
    df["keyword_correct"] = df["true_category"] == df["keyword_category"]
    df["fuzzy_correct"] = df["true_category"] == df["fuzzy_category"]

    return {
        "n": len(df),
        "keyword_accuracy": float(df["keyword_correct"].mean()),
        "fuzzy_accuracy": float(df["fuzzy_correct"].mean()),
        "keyword_errors": df[~df["keyword_correct"]][
            ["shift_id", "note", "true_category", "keyword_category"]
        ].to_dict("records"),
        "fuzzy_errors": df[~df["fuzzy_correct"]][
            ["shift_id", "note", "true_category", "fuzzy_category"]
        ].to_dict("records"),
    }


if __name__ == "__main__":
    import json
    import sys

    sys.path.insert(0, "src")
    notes = pd.read_csv("data/shift_notes.csv", keep_default_na=False)
    report = run_validation(notes)
    print(f"n={report['n']}")
    print(f"keyword-only accuracy: {report['keyword_accuracy']:.3f}")
    print(f"keyword+fuzzy accuracy: {report['fuzzy_accuracy']:.3f}")
    print("\n-- keyword-only errors --")
    for e in report["keyword_errors"]:
        print(e)
    print("\n-- keyword+fuzzy errors --")
    for e in report["fuzzy_errors"]:
        print(e)
