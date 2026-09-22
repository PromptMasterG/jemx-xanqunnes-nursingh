"""Generate predictions.csv and note_classifications.csv at the repo root."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from engine import run
from pipeline import load_data


def main(data_dir: str = "data", out_dir: str = "."):
    dataframes = load_data(data_dir)
    result = run(dataframes)
    employees = dataframes["employees"]

    predictions = result["predictions"][["employee_id", "will_breach", "risk_score"]].copy()
    predictions["risk_score"] = predictions["risk_score"].round(4)
    predictions = predictions.sort_values("employee_id").reset_index(drop=True)

    assert len(predictions) == len(employees), "one row per employee required"
    assert set(predictions["employee_id"]) == set(employees["employee_id"])
    predictions.to_csv(f"{out_dir}/predictions.csv", index=False)
    print(f"wrote predictions.csv: {len(predictions)} rows, {predictions['will_breach'].sum()} flagged")
    print(f"target week: {result['target_week_start'].date()}")

    notes_classified = result["notes_classified"]
    assert len(notes_classified) == len(dataframes["shift_notes"])
    notes_classified.to_csv(f"{out_dir}/note_classifications.csv", index=False)
    print(f"wrote note_classifications.csv: {len(notes_classified)} rows")


if __name__ == "__main__":
    main()
