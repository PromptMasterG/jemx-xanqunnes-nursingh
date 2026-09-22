"""Single entrypoint that both the CSV generator and the dashboard call,
so there is exactly one place that defines "the answer" -- no risk of the
dashboard showing something different from what predictions.csv says.

The target week is always "whatever week contains the most recent
shift_date in the data", not a hardcoded date -- this is what makes
Requirement 4 (load next week's export, no code change) actually true.
"""
from __future__ import annotations

import pandas as pd

from classify_notes import CATEGORY_BUCKET, classify_notes_with_fuzzy_fallback, hours_by_category
from pipeline import compute_daily_hours, compute_shift_intervals, compute_weekly_hours, week_starting
from predict import backtest, fit_risk_calibration, predict_target_week
from validate import run_backtest_report
from validate_notes import HAND_LABELS, run_validation as run_notes_validation


def run(dataframes: dict[str, pd.DataFrame]) -> dict:
    shifts = dataframes["shifts"]
    employees = dataframes["employees"]
    sites = dataframes["sites"]
    shift_notes = dataframes["shift_notes"]

    daily = compute_daily_hours(shifts)
    weekly = compute_weekly_hours(daily)

    max_date = pd.to_datetime(shifts["shift_date"]).max()
    target_week_start = week_starting(pd.Series([max_date]))[0]

    backtest_report, bt = run_backtest_report(daily, weekly, employees)
    calibrator = fit_risk_calibration(bt)
    predictions = predict_target_week(daily, weekly, employees, target_week_start, calibrator)

    notes_classified = classify_notes_with_fuzzy_fallback(shift_notes)

    shift_intervals = compute_shift_intervals(shifts)
    shift_intervals["row_hours"] = (shift_intervals["end"] - shift_intervals["start"]).dt.total_seconds() / 3600
    notes_with_hours = hours_by_category(
        notes_classified,
        shift_intervals[["shift_id", "site_id", "employee_id", "shift_date", "row_hours"]],
    )
    notes_with_hours = notes_with_hours.merge(sites[["site_id", "site_name"]], on="site_id", how="left")

    notes_validation = None
    overlapping_ids = set(HAND_LABELS) & set(shift_notes["shift_id"])
    if len(overlapping_ids) == len(HAND_LABELS):
        notes_validation = run_notes_validation(shift_notes)

    return {
        "target_week_start": target_week_start,
        "daily": daily,
        "weekly": weekly,
        "predictions": predictions,
        "backtest_report": backtest_report,
        "notes_classified": notes_classified,
        "notes_with_hours": notes_with_hours,
        "notes_validation": notes_validation,
        "employees": employees,
        "sites": sites,
    }


def employee_note_summary(employee_id: str, notes_with_hours: pd.DataFrame, lookback_days: int = 21) -> pd.DataFrame:
    """Recent categorized shifts for one employee, most-hours-first -- the
    material behind the dashboard's 'why' column for that person."""
    rows = notes_with_hours[notes_with_hours["employee_id"] == employee_id]
    return rows.sort_values("shift_date", ascending=False)


ACTION_BY_CATEGORY = {
    "relief_no_show": "Check relief-roster coverage for this site -- recurring no-shows are driving the hours up, not workload.",
    "staff_leave_covering": "Coverage gap from sick/family-responsibility leave. Confirm the roster has a standing backup, not just this person picking up the slack.",
    "equipment_failure": "Log a maintenance ticket. Manual workarounds for broken equipment are adding hours every time it happens.",
    "late_handover": "Flag the handover process at this site -- late handovers are a recurring, fixable process issue.",
    "client_requested": "Billable client-requested hours. Confirm the client invoice reflects this before treating it as a cost problem.",
    "other_operational": "Cause noted but not categorized cleanly -- read the raw note before acting.",
    "no_signal": "No explanation on file for these hours. Ask the site supervisor directly.",
}
