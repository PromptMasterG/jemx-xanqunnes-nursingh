"""Backtest validation for the breach predictor.

There is no held-out ground truth available to us (the assessment keeps
that back deliberately), so we validate the only honest way available:
truncate each of the ~9 complete historical weeks to its own Monday-
Wednesday, predict that week's Sunday outcome the same way we predict the
real target week, and check against what actually happened in the full
week (per our own corrected pipeline, not weekly_summary.csv -- see
pipeline.py for why that file is not ground truth either).
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from predict import backtest


def run_backtest_report(daily: pd.DataFrame, weekly: pd.DataFrame, employees: pd.DataFrame) -> dict:
    bt = backtest(daily, weekly, employees)
    bt["actual_breached"] = bt["actual_breached"].astype(bool)

    report = {"n_backtest_rows": len(bt), "actual_breach_rate": float(bt["actual_breached"].mean())}

    for model in ["naive", "personalized"]:
        pred_col = f"predicted_breach_{model}"
        bt[pred_col] = bt[pred_col].astype(bool)
        tn, fp, fn, tp = confusion_matrix(bt["actual_breached"], bt[pred_col]).ravel()
        report[model] = {
            "precision": float(precision_score(bt["actual_breached"], bt[pred_col], zero_division=0)),
            "recall": float(recall_score(bt["actual_breached"], bt[pred_col], zero_division=0)),
            "f1": float(f1_score(bt["actual_breached"], bt[pred_col], zero_division=0)),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "n_flagged": int(bt[pred_col].sum()),
        }
    return report, bt


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "src")
    from pipeline import compute_daily_hours, compute_weekly_hours, load_data

    d = load_data("data")
    daily = compute_daily_hours(d["shifts"])
    weekly = compute_weekly_hours(daily)
    report, bt = run_backtest_report(daily, weekly, d["employees"])
    import json

    print(json.dumps(report, indent=2, default=str))
