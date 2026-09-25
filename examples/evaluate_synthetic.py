"""
Compare na_imputation's calls on the synthetic example with the mechanisms that generated it.

Run from the repository root, after make_synthetic_data.py and the workflow:
    uv run examples/make_synthetic_data.py
    uv run na_imputation.py --lfq examples/synthetic/lfq_synthetic.csv --out-dir examples/synthetic/output --report
    uv run examples/evaluate_synthetic.py

Writes examples/synthetic/output/evaluation_*.csv and prints the tables.
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent / "synthetic"
OUT = BASE / "output"
CLASS_ORDER = ["Complete", "MCAR - random", "MNAR - censored", "Mixed", "Group-structured", "Sparse (drop)"]


def main():
    truth = pd.read_csv(BASE / "truth_proteins.csv")
    cls = pd.read_csv(OUT / "protein_na_classification.csv")[["accession", "na_class"]]
    prot = truth.merge(cls, on="accession", how="left")
    prot["na_class"] = prot["na_class"].fillna("(no observed value)")

    # 1. protein level: built-in mechanism x assigned class
    by_protein = pd.crosstab(prot["mechanism"], prot["na_class"], margins=True, margins_name="total")
    by_protein = by_protein.reindex(columns=[c for c in CLASS_ORDER + ["(no observed value)", "total"]
                                             if c in by_protein.columns])

    # 2. cell level: why a value is missing x how it was handled
    why = pd.read_csv(BASE / "truth_cells.csv").set_index("accession").drop(columns="gene")
    meth = pd.read_csv(OUT / "LFQ_imputation_method_per_cell.csv").set_index("accession").drop(columns="gene")
    handled = meth.reindex(why.index)  # proteins not in the imputed matrix were dropped
    handled = handled.where(handled.notna(), "dropped")
    cells = pd.DataFrame({"why_missing": why.values.ravel(), "handled": handled[why.columns].values.ravel()})
    cells = cells[cells["why_missing"] != "observed"]
    by_cell = pd.crosstab(cells["why_missing"], cells["handled"], margins=True, margins_name="total")
    by_cell = by_cell.reindex(columns=[c for c in ["MinProb", "kNN", "dropped", "total"] if c in by_cell.columns])

    # 3. headline rates
    kept = cells[cells["handled"] != "dropped"]
    low = kept["why_missing"].isin(["censored", "absent"])
    onoff = prot["mechanism"].str.startswith("on/off")
    summary = pd.Series({
        "censored/absent cells given MinProb (of retained)": (kept.loc[low, "handled"] == "MinProb").mean(),
        "MCAR cells given kNN (of retained)": (kept.loc[~low, "handled"] == "kNN").mean(),
        "on/off proteins called Group-structured": (prot.loc[onoff, "na_class"] == "Group-structured").mean(),
        "Group-structured calls that are true on/off": onoff[prot["na_class"] == "Group-structured"].mean(),
    }).round(3)

    by_protein.to_csv(OUT / "evaluation_by_protein.csv")
    by_cell.to_csv(OUT / "evaluation_by_cell.csv")
    summary.rename("value").to_csv(OUT / "evaluation_summary.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("Proteins: built-in mechanism x assigned class\n", by_protein, "\n")
        print("Missing cells: why missing x how handled\n", by_cell, "\n")
        print(summary.to_string())


if __name__ == "__main__":
    main()
