"""Rule-based classifier for shift_notes.csv.

Why rules and not an LLM: the notes are heavily templated -- 846 unique
strings cover 2,117 rows, and the top ~150 templates (with typo variants
folded in via substring matching) cover the overwhelming majority. A
keyword classifier is exactly as accurate as an LLM would be here, every
decision is inspectable in one line, it costs nothing, and it never
hallucinates a category. See NOTES.md for the argument in full.

Categories:
  client_requested      -- client/site management asked for the extra work
                           (billable, not a failure)
  relief_no_show        -- a colleague failed to arrive (no call/no show),
                           or relief never came, and someone had to cover
  staff_leave_covering  -- a colleague was on sick/family-responsibility
                           leave and someone had to cover (legitimate
                           absence, but still an operational cost)
  equipment_failure     -- broken machinery forced manual/slower work
  late_handover         -- handover delays, usually tied to missing keys
                           or an unsigned occurrence book
  no_signal             -- filler: "quiet shift", "ntr", "-", empty, etc.
  other_operational     -- a real note that names a cause but doesn't fit
                           any bucket above (catch-all, kept deliberately
                           small -- see validate_notes.py)

CATEGORY_BUCKET maps each category to the client_requested vs
operational_failure split required by the brief. no_signal is excluded
from that split entirely -- it is not evidence of anything.
"""
from __future__ import annotations

import difflib
import re

import pandas as pd

CATEGORY_BUCKET = {
    "client_requested": "client_requested",
    "relief_no_show": "operational_failure",
    "staff_leave_covering": "operational_failure",
    "equipment_failure": "operational_failure",
    "late_handover": "operational_failure",
    "other_operational": "operational_failure",
    "no_signal": "no_signal",
}

# Exact-match (after normalizing) filler phrases -- checked by full-string
# equality, never substring, so "no issues on site" can't be caught by a
# stray "no" inside a real note elsewhere.
NO_SIGNAL_PHRASES = {
    "", "-", ".", "..", "ok", "okay", "ntr", "n/a", "na", "fine", "all fine",
    "all good", "all ok", "all quiet", "quiet shift", "quiet night", "sharp",
    "nothing to report", "nothing much", "no incidents", "no issues",
    "no issues on site", "as per normal", "as per usual",
    "akukho lutho",  # isiZulu: "there is nothing"
    "niks om te rapporteer nie",  # Afrikaans: "nothing to report"
}

# Ordered (category, [substrings]) -- first match wins. Order matters: the
# "real reason" override must run before client_requested, and
# staff_leave_covering (explicit sick/leave wording) must run before the
# more generic relief_no_show "covering" keywords.
REAL_REASON_OVERRIDE = "real reason"

CLIENT_REQUESTED_KEYWORDS = [
    "client requested", "client wanted", "client asked", "client email",
    "client says stay", "requested by centre management",
    "requested by site manager", "requested by site mgr",
    "centre manager requested", "centre mgr requested",
    "approved by office", "ok'd by centre mgmt", "okd by centre mgmt",
    "approved by client", "klient het ekstra ure gevra",
    "signed off", "approved",
]

STAFF_LEAVE_KEYWORDS = [
    "booked off sick", "off sick", "siek gemeld", "siek geemld",
    "at the clinic", "family responsibility", "gedek vir", "geddek vir",
]

RELIEF_NO_SHOW_KEYWORDS = [
    "no show", "no-show", "didn't come", "didnt come", "did not pitch",
    "never pitched", "never arrived", "nobody came", "no replacement",
    "no relief", "akezanga", "akafikanga", "aflos het nie opgedaag",
    "stood in for", "covering", "control room says relief",
    "relief was suppose to come", "relief only arrived", "2 posts 1 guard",
    "2 pots 1 guard", "shift as well", "ngimele yena", "absent, took",
    "relief no show", "relief never", "covered for", "covered the post",
    "come in, covered", "again - 3rd time", "no relief",
]

EQUIPMENT_FAILURE_KEYWORDS = [
    "generator fault", "machine down", "machine kaput", "buffer machine",
    "scrubber broke", "lift out of order", "gate motor failed",
    "masjien is stukkend", "machine broke", "broke down", "out of order",
    "kaput", "malfunction", "faulty", "generator",
]

LATE_HANDOVER_KEYWORDS = [
    "handover late", "handover delayed", "late handover",
    "oorhandiging was laat", "gewag vir sleutels", "keys missing",
    "ob book not signed", "waiting on paperwork", "waited for handover",
    "hanodver late", "handovver", "min for handover",
]


def _normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[!.]+$", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t


def classify_note(text: str) -> str:
    norm = _normalize(text)

    if norm in NO_SIGNAL_PHRASES or len(norm) <= 2:
        return "no_signal"

    if REAL_REASON_OVERRIDE in norm:
        # e.g. "client signed for the extra hrs but real reason is relief
        # no show again" -- the paperwork says client-approved, the note
        # itself says otherwise. Classify on the stated real cause, not
        # the surface framing.
        tail = norm.split(REAL_REASON_OVERRIDE, 1)[1]
        if any(k in tail for k in ["no show", "no-show", "relief"]):
            return "relief_no_show"
        return "other_operational"

    if any(k in norm for k in CLIENT_REQUESTED_KEYWORDS):
        return "client_requested"

    if any(k in norm for k in STAFF_LEAVE_KEYWORDS):
        return "staff_leave_covering"

    if any(k in norm for k in RELIEF_NO_SHOW_KEYWORDS):
        return "relief_no_show"

    if any(k in norm for k in EQUIPMENT_FAILURE_KEYWORDS):
        return "equipment_failure"

    if any(k in norm for k in LATE_HANDOVER_KEYWORDS):
        return "late_handover"

    return "other_operational"


def classify_notes(notes: pd.DataFrame) -> pd.DataFrame:
    out = notes[["shift_id", "note"]].copy()
    out["category"] = out["note"].apply(classify_note)
    return out[["shift_id", "category", "note"]]


def classify_notes_with_fuzzy_fallback(notes: pd.DataFrame, ratio_threshold: float = 0.82) -> pd.DataFrame:
    """Second method, for comparison: identical keyword pass, but notes
    that land in other_operational get one more chance via nearest-
    neighbor string matching against every note that WAS confidently
    keyword-matched. This recovers typo drift ("shift handovver dealyed
    by 40 min") without hand-listing every misspelling -- the fix
    generalizes to typos we have not seen, instead of overfitting to the
    ones we happened to spot.
    """
    out = classify_notes(notes)
    normalized = out["note"].apply(_normalize)
    confident_mask = out["category"] != "other_operational"
    reference = dict(zip(normalized[confident_mask], out.loc[confident_mask, "category"]))
    reference_texts = list(reference.keys())

    fallback_mask = out["category"] == "other_operational"
    for idx in out.index[fallback_mask]:
        norm = normalized.loc[idx]
        matches = difflib.get_close_matches(norm, reference_texts, n=1, cutoff=ratio_threshold)
        if matches:
            out.loc[idx, "category"] = reference[matches[0]]
    return out


def hours_by_category(notes_classified: pd.DataFrame, shifts_with_hours: pd.DataFrame) -> pd.DataFrame:
    """Join each note's category to its shift's corrected hours and site,
    for the 'where is operational-failure time concentrated' analysis."""
    merged = notes_classified.merge(shifts_with_hours, on="shift_id", how="left")
    merged["bucket"] = merged["category"].map(CATEGORY_BUCKET)
    return merged
