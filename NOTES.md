# Notes

## Assumptions

- **Missing clock-outs** (184 of 8,863 rows, spread across 61 dates): imputed from the employee's own median shift length elsewhere in the data, not dropped. Dropping — what `weekly_summary.csv` does — makes anyone who forgot to punch out on a long shift look like they worked nothing that day.
- **Public holiday hours count toward the 45h/10h caps.** National Women's Day (2026-08-10) is the Monday of the target week. The BCEA caps limit hours worked; the 2x pay multiplier only changes what those hours cost the client, so it doesn't change whether someone breaches.
- **`weekly_summary.csv` is not ground truth.** A rollover-corrected, non-deduplicated sum of shift rows matches its `total_hours` to the cent on all 2,122 rows — it has never once caught the bug below. Every hour here is recomputed from `shifts.csv` directly.
- **Overlapping double-booked shifts** (126 employee-days: 76 fully-contained, 50 partial, 0 sequential) are relief-covering events logged twice under two site codes, not real double duty — a person can't work two sites at once. We take the union of the two time windows, not the sum. Only 29 of the 126 have a supporting note; the other 97 were found by checking every employee/date pair for overlapping windows, not by reading text.
- **Forecasts are capped at 16h/day for the 4 remaining days.** A thin sample of historical weeks can otherwise multiply a personal ratio out to 100+ predicted hours for someone whose pace genuinely does run high — physically impossible, so it's capped before being reported.

## Checking the note-sorting

Hand-labelled 63 notes by reading them, stratified across all 7 categories and oversampling the keyword classifier's catch-all bucket (`other_operational`, 83 of 2,117 notes) since that's where a keyword classifier's mistakes concentrate. Keyword-only agreement: 48/63 (76.2%). Every miss was the same failure mode: a typo broke a substring match — "shift handover **deelayed** by 40 min", "**gaate** motor failed", "relief no **shwo** AGAIN" — and the note fell into the catch-all bucket. Never a confident wrong category.

Second method, for comparison: nearest-neighbour matching of catch-all notes against the ~2,034 notes the keyword pass already trusted. That fixed all 15 errors on the hand-labelled sample (63/63) and 82 of the 83 catch-all notes corpus-wide, because 846 unique note strings cover all 2,117 rows — this dataset is heavily templated, so a typo'd note is usually one edit away from a common one. That near-100% is a property of this corpus's repetition, not a claim the method generalizes to a genuinely novel note it's never seen a relative of.

Worth naming one deliberately tricky case: 18 notes read "client signed for the extra hrs but real reason is relief no show again" — the paperwork says client-approved, the note itself says otherwise. Classified as `relief_no_show`, not `client_requested`, by checking for "real reason" before ever checking the approval keywords.

## What a trained model would learn here that this doesn't

A keyword classifier can't handle a note it hasn't seen a relative of, and it has no sense of degree — "waited 20 min for handover" and "waited 50 min for handover" are the same category regardless of how disruptive each one actually was. A trained model, even something as simple as TF-IDF plus logistic regression over the note text, would pick up compositional signal a keyword list can't: negation, and which words in an unfamiliar sentence are actually carrying the meaning, without anyone hand-listing every phrasing.

The breach predictor has the same ceiling. It uses one number — this week's Monday-Wednesday pace times a personal historical ratio — and ignores site, role, day-of-week shape, and whether this person's hours have been trending up for weeks. A gradient-boosted model over those features would likely beat our 8.3% precision / 51.3% recall.

To test either without fooling myself: split by time, not at random. Train the note classifier on the first 7 historical weeks, hand-label a fresh sample from the last 2 it never saw, and measure agreement there — a random split leaks, because so many notes are near-duplicates of each other. The breach backtest here already does this correctly (predict week N from week N's own Monday-Wednesday, using only ratios learned from weeks before N); a trained model would need to pass that same test, on a couple hundred real future employee-weeks, before it earns trust over this heuristic.
