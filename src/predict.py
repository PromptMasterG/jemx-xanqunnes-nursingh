"""Predict who breaches the 10h weekly overtime cap by Sunday, from a
partial (Mon-Wed) week.

Two models, both explainable from a single number:

  naive: everyone's Mon-Wed pace is extrapolated by a flat 7/3 multiplier.
  personalized: each employee gets their OWN historical
    (full_week_hours / mon_wed_hours) ratio, learned from their prior
    complete weeks. Falls back to the global median ratio for employees
    with no usable history (e.g. brand new, or zero hours in a prior
    Mon-Wed).

Both produce a predicted full-week hours total, from which predicted
overtime and a breach flag follow directly from the BCEA 45h/10h caps.
risk_score is NOT a hardcoded 0.01/0.99 -- it comes from a 1-feature
logistic regression fit on the backtest (predicted_overtime -> actual
breach), so it reflects what actually happened historically at that
predicted overtime level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pipeline import compute_daily_hours, compute_weekly_hours, week_starting

MON_WED_DAYS = 3
FULL_WEEK_DAYS = 7
REMAINING_DAYS = FULL_WEEK_DAYS - MON_WED_DAYS
ORDINARY_CAP = 45.0
OVERTIME_CAP = 10.0
# A ratio-based extrapolation has no sense of physical limits: an employee
# with a low, variable Mon-Wed pace and one unusually strong historical
# ratio can get multiplied out to 100+ predicted hours, which nobody can
# actually work in 4 remaining days. Cap the remaining-days portion of the
# prediction at a generous but bounded daily maximum (double-shift
# territory, matching the "double duty" pattern seen in the notes) so a
# thin sample size can't produce an unphysical forecast.
MAX_REASONABLE_HOURS_PER_REMAINING_DAY = 16.0


def mon_wed_hours_by_week(daily: pd.DataFrame) -> pd.DataFrame:
    """Sum of corrected hours for Mon/Tue/Wed of each week, per employee."""
    d = daily.copy()
    d["dow"] = d["shift_date"].dt.weekday  # Mon=0
    mw = d[d["dow"] <= 2]
    out = (
        mw.groupby(["employee_id", "week_start"])["hours"]
        .sum()
        .reset_index()
        .rename(columns={"hours": "mon_wed_hours"})
    )
    return out


def _employee_ratios(history: pd.DataFrame) -> pd.Series:
    """Median (full_week_hours / mon_wed_hours) per employee, using only
    weeks with mon_wed_hours > 0."""
    usable = history[history["mon_wed_hours"] > 0].copy()
    usable["ratio"] = usable["total_hours"] / usable["mon_wed_hours"]
    return usable.groupby("employee_id")["ratio"].median()


def build_history_table(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    mw = mon_wed_hours_by_week(daily)
    hist = weekly.merge(mw, on=["employee_id", "week_start"], how="left")
    hist["mon_wed_hours"] = hist["mon_wed_hours"].fillna(0.0)
    return hist


def predict_week(
    history: pd.DataFrame,
    target_week_start: pd.Timestamp,
    employees: pd.DataFrame,
    exclude_week: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Predict full-week hours for target_week_start using every week in
    `history` strictly before it (or != exclude_week, for leave-one-out
    backtesting) to learn each employee's ratio.
    """
    if exclude_week is not None:
        prior = history[history["week_start"] != exclude_week]
    else:
        prior = history[history["week_start"] < target_week_start]

    ratios = _employee_ratios(prior)
    global_ratio = ratios.median() if len(ratios) else FULL_WEEK_DAYS / MON_WED_DAYS

    target_mw = history[history["week_start"] == target_week_start][
        ["employee_id", "mon_wed_hours"]
    ]
    all_emp = pd.DataFrame({"employee_id": employees["employee_id"].unique()})
    target_mw = all_emp.merge(target_mw, on="employee_id", how="left")
    target_mw["mon_wed_hours"] = target_mw["mon_wed_hours"].fillna(0.0)

    target_mw["employee_ratio"] = target_mw["employee_id"].map(ratios)
    target_mw["used_fallback_ratio"] = target_mw["employee_ratio"].isna()
    target_mw["employee_ratio"] = target_mw["employee_ratio"].fillna(global_ratio)

    max_remaining = REMAINING_DAYS * MAX_REASONABLE_HOURS_PER_REMAINING_DAY

    raw_naive = target_mw["mon_wed_hours"] * (FULL_WEEK_DAYS / MON_WED_DAYS)
    remaining_naive = (raw_naive - target_mw["mon_wed_hours"]).clip(upper=max_remaining)
    target_mw["predicted_hours_naive"] = target_mw["mon_wed_hours"] + remaining_naive

    raw_personalized = np.where(
        target_mw["mon_wed_hours"] > 0,
        target_mw["mon_wed_hours"] * target_mw["employee_ratio"],
        0.0,
    )
    remaining_personalized = (raw_personalized - target_mw["mon_wed_hours"]).clip(upper=max_remaining)
    target_mw["predicted_hours_personalized"] = target_mw["mon_wed_hours"] + remaining_personalized

    for col in ["predicted_hours_naive", "predicted_hours_personalized"]:
        ot_col = col.replace("predicted_hours", "predicted_overtime")
        target_mw[ot_col] = (target_mw[col] - ORDINARY_CAP).clip(lower=0)
        target_mw[ot_col.replace("predicted_overtime", "predicted_breach")] = (
            target_mw[ot_col] > OVERTIME_CAP
        )

    target_mw["week_start"] = target_week_start
    return target_mw


def backtest(daily: pd.DataFrame, weekly: pd.DataFrame, employees: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-week-out backtest across every complete historical week."""
    history = build_history_table(daily, weekly)
    complete_weeks = sorted(history["week_start"].unique())

    results = []
    for wk in complete_weeks:
        actual = weekly[weekly["week_start"] == wk][["employee_id", "breached", "overtime_hours"]]
        pred = predict_week(history, wk, employees, exclude_week=wk)
        merged = pred.merge(actual, on="employee_id", how="left")
        merged["actual_breached"] = merged["breached"].fillna(False)
        merged["week_start"] = wk
        results.append(merged)
    return pd.concat(results, ignore_index=True)


def fit_risk_calibration(backtest_df: pd.DataFrame) -> LogisticRegression:
    """1-feature logistic regression: predicted_overtime_personalized ->
    P(actual breach). This is what turns a heuristic number into a
    genuine, backtest-calibrated risk_score instead of a hardcoded 0.01/0.99.
    """
    X = backtest_df[["predicted_overtime_personalized"]]
    y = backtest_df["actual_breached"].astype(int)
    model = LogisticRegression()
    model.fit(X, y)
    return model


def predict_target_week(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    employees: pd.DataFrame,
    target_week_start: pd.Timestamp,
    calibrator: LogisticRegression,
) -> pd.DataFrame:
    history = build_history_table(daily, weekly)
    pred = predict_week(history, target_week_start, employees, exclude_week=None)
    # risk_score: a calibrated, continuous 0-1 number for ranking/triage
    # (from the backtest logistic fit -- NOT the decision rule below).
    pred["risk_score"] = calibrator.predict_proba(pred[["predicted_overtime_personalized"]])[:, 1]
    # will_breach: the actual legal call. This is a direct application of
    # the BCEA 10h cap to our best point-estimate (personalized model,
    # highest F1 in backtest -- see validate.py), not a probability cutoff.
    # A 0.5 cutoff on risk_score would flag almost nobody, because true
    # breaches are only ~3.6% of employee-weeks and a single-feature
    # logistic fit correctly refuses to push any prediction past ~0.49
    # given how weak that signal is -- using it as the decision boundary
    # would silently suppress every positive prediction.
    pred["will_breach"] = pred["predicted_breach_personalized"].astype(int)
    return pred
