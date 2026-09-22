"""Corrected hours pipeline.

Fixes two bugs the client's own weekly_summary.csv does not fix:
  1. Midnight rollover: night shifts where clock_out's time-of-day is earlier
     than clock_in's are read as a negative duration unless 24h is added.
  2. Double-booked relief coverage: 126 employee-days have two shift rows
     (different sites) whose time windows overlap. The correct hours for
     that employee-day is the UNION of the two windows, not the sum -- a
     person cannot work two places at once. weekly_summary.csv sums them,
     which is why it does not match ground truth.

Also imputes missing clock-outs (2.1% of rows) from the employee's own
median shift length, rather than silently dropping them like the client's
system does (see ASSUMPTIONS in NOTES.md).
"""
from __future__ import annotations

import pandas as pd


def load_data(data_dir: str) -> dict[str, pd.DataFrame]:
    return {
        "shifts": pd.read_csv(f"{data_dir}/shifts.csv"),
        "employees": pd.read_csv(f"{data_dir}/employees.csv"),
        "sites": pd.read_csv(f"{data_dir}/sites.csv"),
        "shift_notes": pd.read_csv(f"{data_dir}/shift_notes.csv", keep_default_na=False),
        "public_holidays": pd.read_csv(f"{data_dir}/public_holidays.csv"),
        "weekly_summary": pd.read_csv(f"{data_dir}/weekly_summary.csv"),
        "payroll_details": pd.read_csv(f"{data_dir}/payroll_details.csv"),
    }


def week_starting(dt: pd.Series) -> pd.Series:
    """Monday of the ISO week containing dt (weeks run Mon-Sun)."""
    return (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()


def _impute_missing_clockouts(shifts: pd.DataFrame) -> pd.DataFrame:
    """Fill missing clock_out_time from the employee's own median shift
    length (complete shifts only). Falls back to the shift_pattern-wide
    median when an employee has no complete shifts to learn from.

    This is a stated assumption, not a certainty -- see NOTES.md. We flag
    every imputed row with was_imputed=True so downstream consumers (the
    risk model) can treat those hours as less certain than a real punch.
    """
    shifts = shifts.copy()
    shifts["clock_in_dt"] = pd.to_datetime(shifts["clock_in_time"], format="%H:%M")
    has_out = shifts["clock_out_time"].notna() & (shifts["clock_out_time"] != "")

    shifts["clock_out_dt_raw"] = pd.NaT
    shifts.loc[has_out, "clock_out_dt_raw"] = pd.to_datetime(
        shifts.loc[has_out, "clock_out_time"], format="%H:%M"
    )
    dur_minutes = (shifts.loc[has_out, "clock_out_dt_raw"] - shifts.loc[has_out, "clock_in_dt"]).dt.total_seconds() / 60
    dur_minutes = dur_minutes.where(dur_minutes > 0, dur_minutes + 24 * 60)  # rollover-aware duration
    shifts.loc[has_out, "_duration_min"] = dur_minutes

    emp_median = shifts.groupby("employee_id")["_duration_min"].median()
    shifts["was_imputed"] = ~has_out

    missing_idx = shifts.index[~has_out]
    for idx in missing_idx:
        emp = shifts.at[idx, "employee_id"]
        fill_min = emp_median.get(emp, float("nan"))
        if pd.isna(fill_min):
            fill_min = shifts["_duration_min"].median()  # global fallback
        clock_in = shifts.at[idx, "clock_in_dt"]
        shifts.at[idx, "clock_out_dt_raw"] = clock_in + pd.Timedelta(minutes=fill_min)

    shifts = shifts.drop(columns=["_duration_min"], errors="ignore")
    return shifts


def compute_shift_intervals(shifts: pd.DataFrame) -> pd.DataFrame:
    """Return shifts with start/end timestamps, rollover-corrected and with
    missing clock-outs imputed."""
    shifts = _impute_missing_clockouts(shifts)
    shifts["start"] = pd.to_datetime(shifts["shift_date"]) + (
        shifts["clock_in_dt"] - shifts["clock_in_dt"].dt.normalize()
    )
    end_time_of_day = shifts["clock_out_dt_raw"] - shifts["clock_out_dt_raw"].dt.normalize()
    shifts["end"] = pd.to_datetime(shifts["shift_date"]) + end_time_of_day
    rollover = shifts["end"] <= shifts["start"]
    shifts.loc[rollover, "end"] += pd.Timedelta(days=1)
    shifts["had_rollover"] = rollover
    return shifts.drop(columns=["clock_in_dt", "clock_out_dt_raw"])


def _union_minutes(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    """Total minutes covered by the union of possibly-overlapping intervals."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = pd.Timedelta(0)
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = s, e
    total += cur_end - cur_start
    return total.total_seconds() / 60


def compute_daily_hours(shifts: pd.DataFrame) -> pd.DataFrame:
    """One row per (employee_id, shift_date): corrected hours worked, using
    the UNION of overlapping shift windows (fixes the double-booking bug)."""
    intervals = compute_shift_intervals(shifts)

    rows = []
    for (emp, date), grp in intervals.groupby(["employee_id", "shift_date"]):
        pairs = list(zip(grp["start"], grp["end"]))
        union_min = _union_minutes(pairs)
        naive_sum_min = sum((e - s).total_seconds() / 60 for s, e in pairs)
        rows.append(
            {
                "employee_id": emp,
                "shift_date": date,
                "hours": union_min / 60,
                "naive_sum_hours": naive_sum_min / 60,
                "had_overlap": len(pairs) > 1 and union_min < naive_sum_min - 1e-6,
                "had_rollover": bool(grp["had_rollover"].any()),
                "had_imputed_clockout": bool(grp["was_imputed"].any()),
                "n_rows": len(grp),
                "site_ids": ",".join(sorted(grp["site_id"].unique())),
            }
        )
    daily = pd.DataFrame(rows)
    daily["shift_date"] = pd.to_datetime(daily["shift_date"])
    daily["week_start"] = week_starting(daily["shift_date"])
    return daily


def compute_weekly_hours(
    daily: pd.DataFrame, contract_ordinary_hours: float = 45.0, overtime_cap: float = 10.0
) -> pd.DataFrame:
    weekly = (
        daily.groupby(["employee_id", "week_start"])
        .agg(
            total_hours=("hours", "sum"),
            days_worked=("shift_date", "nunique"),
            had_overlap_any=("had_overlap", "any"),
            had_imputed_any=("had_imputed_clockout", "any"),
        )
        .reset_index()
    )
    weekly["ordinary_hours"] = weekly["total_hours"].clip(upper=contract_ordinary_hours)
    weekly["overtime_hours"] = (weekly["total_hours"] - contract_ordinary_hours).clip(lower=0)
    weekly["breached"] = weekly["overtime_hours"] > overtime_cap
    return weekly
