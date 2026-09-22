"""Ops room dashboard: who's going into overtime breach by Sunday, and why.

Built for a contract manager on a phone between sites, not an analyst.
Upload a fresh weekly export in the sidebar and everything below
recomputes -- no code change, no developer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import streamlit as st

from engine import ACTION_BY_CATEGORY, run
from pipeline import load_data

st.set_page_config(page_title="Jem Ops Room", page_icon="\U0001F4CB", layout="wide")

NAVY = "#051D2E"
CORAL = "#FF697F"
CORAL_DARK = "#E1506B"
CORAL_LIGHT = "#FF93A3"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {NAVY}; }}
    h1, h2, h3, h4, p, label, .stMarkdown, span {{ color: #FFFFFF; }}
    [data-testid="stMetricValue"] {{ color: {CORAL}; }}
    [data-testid="stMetricLabel"] {{ color: #FFFFFF; }}
    .stDataFrame {{ background-color: #0A2A40; }}
    div[data-testid="stSidebar"] {{ background-color: #03141F; }}
    .streamlit-expanderHeader {{ color: #FFFFFF; }}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_FILES = [
    "shifts.csv", "employees.csv", "sites.csv", "shift_notes.csv",
    "public_holidays.csv", "weekly_summary.csv", "payroll_details.csv",
]


@st.cache_data(show_spinner=False)
def load_default_data():
    return load_data("data")


def load_uploaded_overrides(uploaded_files, base: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    dataframes = dict(base)
    for f in uploaded_files or []:
        name = f.name.strip().lower()
        key = name.replace(".csv", "")
        if key in dataframes:
            keep_default_na = key == "shift_notes"
            dataframes[key] = pd.read_csv(f, keep_default_na=keep_default_na)
    return dataframes


st.markdown(
    f"<div style='background:{CORAL}; padding:14px 20px; border-radius:8px; margin-bottom:18px;'>"
    f"<span style='font-size:22px; font-weight:700; color:white;'>jem</span> "
    f"<span style='font-size:16px; color:white;'>&nbsp;Ops Room -- Overtime Risk</span></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Load a new week's export")
    st.caption("Upload any of the files below to replace the bundled data (same columns, new rows). Anything not uploaded keeps last week's file.")
    uploaded = st.file_uploader(
        "CSV files", type="csv", accept_multiple_files=True, label_visibility="collapsed"
    )
    st.caption("Expected filenames: " + ", ".join(DATA_FILES))

base_data = load_default_data()
dataframes = load_uploaded_overrides(uploaded, base_data)

with st.spinner("Computing..."):
    result = run(dataframes)

pred = result["predictions"]
weekly = result["weekly"]
employees = dataframes["employees"]
target_week = result["target_week_start"]
bt_report = result["backtest_report"]
notes_val = result["notes_validation"]
notes_hours = result["notes_with_hours"]

n_flagged = int(pred["will_breach"].sum())
n_total = len(pred)

st.markdown(f"#### Week of {target_week.date()} (Mon-Sun) -- forecast from Monday-Wednesday actuals")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees at risk of breach", n_flagged, help="predicted_overtime > 10h by our best model")
c2.metric("Total employees", n_total)
c3.metric("Historical breach rate", f"{bt_report['actual_breach_rate']*100:.1f}%", help="Share of employee-weeks that actually breached, in ~9 complete historical weeks (corrected hours, not the client's export).")
c4.metric("Model recall (backtest)", f"{bt_report['personalized']['recall']*100:.0f}%", help="Share of real historical breaches this model would have caught, tested by truncating past weeks to their own Mon-Wed.")

st.markdown("---")

left, right = st.columns([3, 2])

with left:
    st.markdown("### Who to look at today")
    st.caption("Ranked by risk_score. This is a triage list, not an accusation -- read the 'why' before acting.")

    merged = pred.merge(employees, on="employee_id", how="left")
    top = merged.sort_values("risk_score", ascending=False).head(20)

    for _, row in top.iterrows():
        emp_notes = notes_hours[notes_hours["employee_id"] == row["employee_id"]]
        op_notes = emp_notes[emp_notes["bucket"] == "operational_failure"]
        if len(op_notes):
            dominant_cat = op_notes.groupby("category")["row_hours"].sum().idxmax()
            why = f"Mostly **{dominant_cat.replace('_', ' ')}** ({op_notes['row_hours'].sum():.0f}h of noted hours across the observed weeks)."
            action = ACTION_BY_CATEGORY.get(dominant_cat, "")
        elif len(emp_notes):
            dominant_cat = emp_notes.groupby("category")["row_hours"].sum().idxmax()
            why = f"Mostly **{dominant_cat.replace('_', ' ')}**."
            action = ACTION_BY_CATEGORY.get(dominant_cat, "")
        else:
            why = "No supervisor notes on file for this person's recent shifts."
            action = ACTION_BY_CATEGORY["no_signal"]

        flag = "\U0001F534" if row["will_breach"] else "\U0001F7E1"
        with st.container(border=True):
            st.markdown(
                f"{flag} **{row['full_name']}** ({row['employee_id']}) -- {row['role']}, {row['primary_site_id']} "
                f"&nbsp;&nbsp; risk_score `{row['risk_score']:.2f}`"
            )
            st.markdown(f"Mon-Wed so far: **{row['mon_wed_hours']:.1f}h** -> predicted week total **{row['predicted_hours_personalized']:.1f}h** "
                        f"(predicted overtime **{row['predicted_overtime_personalized']:.1f}h**, cap is 10h)")
            st.markdown(f"_Why:_ {why}")
            st.markdown(f"_Do this:_ {action}")

with right:
    st.markdown("### Where the operational-failure hours are")
    op = notes_hours[notes_hours["bucket"] == "operational_failure"]
    by_cat = op.groupby("category")["row_hours"].sum().sort_values(ascending=False)
    st.bar_chart(by_cat, color=CORAL)

    st.markdown("**By site**")
    by_site = op.groupby("site_name")["row_hours"].sum().sort_values(ascending=False)
    st.bar_chart(by_site, color=CORAL_DARK)

    client_hours = notes_hours[notes_hours["bucket"] == "client_requested"]["row_hours"].sum()
    op_hours = op["row_hours"].sum()
    st.markdown(
        f"**{op_hours:,.0f}h** tagged operational-failure vs **{client_hours:,.0f}h** client-requested "
        f"(on shifts with a supervisor note). Relief no-shows are the single biggest driver."
    )

st.markdown("---")
with st.expander("How this was checked (validation)"):
    st.markdown("#### Breach prediction backtest")
    st.caption("Each of the ~9 complete historical weeks, truncated to its own Mon-Wed, predicted the same way as the real target week, checked against what actually happened.")
    rows = []
    for model in ["naive", "personalized"]:
        m = bt_report[model]
        rows.append({"model": model, "precision": round(m["precision"], 3), "recall": round(m["recall"], 3),
                     "f1": round(m["f1"], 3), "flagged": m["n_flagged"], "true_positives": m["true_positives"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"n={bt_report['n_backtest_rows']} employee-weeks, actual breach rate {bt_report['actual_breach_rate']*100:.1f}%.")

    st.markdown("#### Note classification hand-check")
    if notes_val:
        st.caption(f"n={notes_val['n']} hand-labelled notes, stratified across categories (oversampling the catch-all bucket, where errors concentrate).")
        st.write(f"Keyword-only accuracy: **{notes_val['keyword_accuracy']*100:.1f}%** | Keyword + fuzzy-fallback accuracy: **{notes_val['fuzzy_accuracy']*100:.1f}%**")
        if notes_val["keyword_errors"]:
            st.caption("Where keyword-only got it wrong (all recovered by the fuzzy fallback -- see NOTES.md):")
            st.dataframe(pd.DataFrame(notes_val["keyword_errors"]).head(10), hide_index=True)
    else:
        st.caption("Hand-labelled sample is keyed to the original dataset's shift_ids and does not apply to this uploaded data.")

st.markdown("---")
st.caption("Data: " + ", ".join(f"{k} ({len(v)} rows)" for k, v in dataframes.items()))
