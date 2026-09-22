# Jem Applied AI Intern — Technical Assessment

Who's going to breach the 10h weekly overtime cap by Sunday, and what should
somebody do about it today. See [NOTES.md](NOTES.md) for assumptions and
validation, [ORIGINAL_BRIEF.md](ORIGINAL_BRIEF.md) for the assessment brief.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a new week's CSV export in the sidebar to recompute against it —
no code change needed.

## Regenerate predictions.csv / note_classifications.csv

```bash
python src/generate_outputs.py
```

## Structure

- `data/` — the provided CSVs
- `src/pipeline.py` — corrected hours (midnight rollover + overlap union + missing clock-out handling)
- `src/predict.py` — breach forecasting (naive vs personalized-ratio models)
- `src/validate.py` — backtest against ~9 complete historical weeks
- `src/classify_notes.py` — supervisor-note taxonomy (keyword + fuzzy fallback)
- `src/validate_notes.py` — hand-labelled accuracy check
- `src/engine.py` — single entrypoint shared by the dashboard and the CSV generator
- `app.py` — the Streamlit dashboard
