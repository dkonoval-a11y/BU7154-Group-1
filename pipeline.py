"""Read the dataset, calculate tables and save the results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (age_results, annual, descriptive_results, monthly,
                       prepare, retention_results, seasonal)
from .checks import monthly_diagnostics, source_checks
from .database import build_database

ROOT = Path(__file__).resolve().parents[1]


def load_data(raw_path: Path | None = None):
    """Return source records, the prepared table and calculation fields."""
    path = Path(raw_path) if raw_path is not None else ROOT / "data/raw.xlsx"
    raw = pd.read_excel(path, sheet_name="DadosV3")
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed:")]
    prepared, members = prepare(raw)
    checks = source_checks(members)
    failed = checks.loc[checks.Result.eq("FAIL"), "Check"].tolist()
    if failed:
        raise ValueError("Source checks failed: " + ", ".join(failed))
    return raw, prepared, members


def save_tables(folder: Path, tables: dict[str, pd.DataFrame]) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False, float_format="%.17g",
                     date_format="%Y-%m-%dT%H:%M:%S.%f", encoding="utf-8-sig")


def preparation_tables(prepared: pd.DataFrame, members: pd.DataFrame):
    columns = ["Id", "StartDate", "FinishDate", "AgeReferenceDate", "Age",
               "EntryAgeLower", "EstimatedEntryAge", "EntryAgeUpper", "ValidAge"]
    return {"AnalysisData": prepared, "EntryAgeCalculations": members[columns],
            "SourceChecks": source_checks(members)}


def monthly_tables(members: pd.DataFrame):
    frame = monthly(members)
    tables = {"MonthlyData": frame, "AnnualSummary": annual(frame),
              "SeasonalSummary": seasonal(frame),
              **monthly_diagnostics(members, frame)}
    residuals = tables["MonthlyReconciliation"].drop(columns="MonthStart")
    if residuals.abs().to_numpy().max() > 1e-12:
        raise ValueError("Monthly membership totals do not reconcile")
    if not np.array_equal(frame.OpeningMemberships.iloc[1:],
                          frame.ActiveMemberships.iloc[:-1]):
        raise ValueError("Monthly opening and closing totals do not agree")
    for field in ("AnnualEnrolmentShare", "AnnualDropoutShare"):
        if not np.allclose(frame.groupby("Year")[field].sum(), 1):
            raise ValueError(f"Annual shares do not sum to one: {field}")
    return tables


def run_analysis(raw_path: Path, output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status_file = output / "checks.json"
    status_file.write_text('{"status": "RUNNING"}\n', encoding="utf-8")
    raw, prepared, members = load_data(raw_path)
    stages = {
        "data": preparation_tables(prepared, members),
        "monthly": monthly_tables(members),
        "age_groups": age_results(members),
        "retention": retention_results(members),
        "descriptive": descriptive_results(members),
    }
    for folder, tables in stages.items():
        save_tables(output / folder, tables)
    database = build_database(members, output / "database", ROOT / "database/schema.sql")
    result = {
        "status": "PASS",
        "source_rows": len(raw),
        "source_fields": len(raw.columns),
        "prepared_fields": len(prepared.columns),
        "source_sha256": hashlib.sha256(Path(raw_path).read_bytes()).hexdigest(),
        "reporting_period": ["2015-01-01", "2018-12-31"],
        "observation_cutoff": "2019-10-31",
        "tables_exported": sum(len(t) for t in stages.values()),
        "database": database,
    }
    status_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
