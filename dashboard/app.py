"""Streamlit dashboard for SynthGate status and evaluation reports."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="SynthGate", layout="wide")
st.title("SynthGate")

st.subheader("Pipeline status")
for label, path in (
    ("Prepared data", "data/processed/train.csv"),
    ("Gaussian Copula", "models/gaussian_copula.pkl"),
    ("CTGAN", "models/ctgan.pkl"),
    ("Evaluation", "artifacts/evaluation/summary.txt"),
):
    st.write(f"{'Ready' if Path(path).is_file() else 'Missing'}: {label}")

st.subheader("Evaluation reports")
reports = sorted(Path("artifacts/evaluation").glob("*_*.json"))
if not reports:
    st.info("Run the evaluation stage first.")
else:
    rows = []
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": report["model_type"],
                "contract": report["contract"],
                "target_error": report["scenario_target_error"],
                "stability": report["stability"]["stability_score"],
                "rule_violations": sum(
                    int(value["violations"])
                    for value in report["rule_results"].values()
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
