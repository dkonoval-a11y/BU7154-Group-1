#!/usr/bin/env python3
"""Calculate all results from the source dataset."""
import argparse
from pathlib import Path

from src.pipeline import ROOT, run_analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    result = run_analysis(args.raw, args.output)
    print(f"Completed: {result['source_rows']:,} customers, "
          f"{result['tables_exported']} CSV tables, database integrity OK.")
    print(f"Results: {args.output.resolve()}")
