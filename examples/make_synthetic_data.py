"""
Generate a synthetic LFQ matrix with known missingness mechanisms, for demonstrating na_imputation.

Design (all on the log2 scale):
  * 20 patients: Control01-08 and Disease01-12; four of them contribute two samples (L and R),
    giving 24 samples. Sample IDs look like "Disease05.L".
  * 800 proteins with true abundance mu_i ~ N(22, 2.5), a patient effect (shared by L/R samples)
    and technical noise.
  * 40 proteins differ in abundance (Disease +1.5 log2): ordinary differential abundance.
  * 20 "on/off" proteins are truly absent (-8 log2) in one group: 12 absent in Control,
    8 absent in Disease.
  * 30 "flaky" proteins, abundant but prone to random dropout (20% MCAR).
  * Detection: each sample has a detection limit (LOD). Most runs vary a little; three runs are
    shallow (LOD +2.5 to +4). P(detected) = logistic((y - LOD_j) / 0.5)  -> left-censoring.
  * On top of censoring, 2% of all cells drop out completely at random (MCAR).
  * Finally each sample is median-normalised on its observed values, as a typical pipeline does
    (which reproduces the down-shift of shallow runs).

Writes to examples/synthetic/:
  lfq_synthetic.csv        proteins x samples (PG.ProteinAccessions, PG.Genes, Protein.Names, samples...)
  truth_proteins.csv       per protein: true abundance and built-in mechanism
  truth_cells.csv          per cell: observed / censored / mcar / absent (why the value is missing)
  sample_sheet.csv         per sample: sample_id, patient, group, subtype (Control / DiseaseA /
                           DiseaseB), side, batch -- metadata for --sample-sheet
  truth_samples.csv        per sample: LOD offset used in the simulation (3 shallow runs)
  groups_rare.csv          two-column groups file with a 2-patient 'DiseaseRare' group
                           (for the minimum-group-size example)

Run from the repository root:  uv run examples/make_synthetic_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
OUT = Path(__file__).resolve().parent / "synthetic"

N_PROT = 800
N_CONTROL, N_DISEASE = 8, 12
BILATERAL = {"Control03", "Disease02", "Disease05", "Disease09"}
N_DE, DE_SHIFT = 40, 1.5
N_OFF_IN_CONTROL, N_OFF_IN_DISEASE, OFF_SHIFT = 12, 8, -8.0
N_FLAKY, FLAKY_MCAR = 30, 0.20
MCAR_RATE = 0.02
LOD_BASE, LOD_WIDTH = 20.0, 0.5
SHALLOW = {"Disease04": 4.0, "Control06": 3.0, "Disease11": 2.5}  # patient -> LOD offset of its first run


def main():
    rng = np.random.default_rng(SEED)

    # ---- samples
    patients = [f"Control{i:02d}" for i in range(1, N_CONTROL + 1)] + \
               [f"Disease{i:02d}" for i in range(1, N_DISEASE + 1)]
    rows = []
    for p in patients:
        sides = ["L", "R"] if p in BILATERAL else [rng.choice(["L", "R"])]
        for side in sides:
            rows.append(dict(sample_id=f"{p}.{side}", patient=p, group=p.rstrip("0123456789"), side=side))
    samples = pd.DataFrame(rows).set_index("sample_id")
    samples["lod_offset"] = rng.normal(0, 0.8, len(samples))
    for pat, off in SHALLOW.items():
        samples.loc[samples.index[samples["patient"] == pat][0], "lod_offset"] = off
    n_s = len(samples)
    is_disease = (samples["group"] == "Disease").values

    # ---- proteins
    mu = rng.normal(22, 2.5, N_PROT)
    idx = rng.permutation(N_PROT)
    de = idx[:N_DE]
    off_c = idx[N_DE:N_DE + N_OFF_IN_CONTROL]
    off_d = idx[N_DE + N_OFF_IN_CONTROL:N_DE + N_OFF_IN_CONTROL + N_OFF_IN_DISEASE]
    flaky = idx[N_DE + N_OFF_IN_CONTROL + N_OFF_IN_DISEASE:][:N_FLAKY]
    # on/off and flaky proteins are mid/high abundance, so their NAs are not explained by low abundance
    mu[off_c] = rng.normal(23.5, 1.0, len(off_c))
    mu[off_d] = rng.normal(23.5, 1.0, len(off_d))
    mu[flaky] = rng.normal(25.5, 1.0, len(flaky))

    mechanism = np.full(N_PROT, "none", dtype=object)
    mechanism[de] = f"differential abundance (Disease {DE_SHIFT:+})"
    mechanism[off_c] = "on/off: absent in Control"
    mechanism[off_d] = "on/off: absent in Disease"
    mechanism[flaky] = f"flaky: {FLAKY_MCAR:.0%} random dropout"

    # ---- true values: abundance + patient effect (shared by L/R) + technical noise + group effects
    pat_effect = {p: rng.normal(0, 0.4, N_PROT) for p in patients}
    Y = np.column_stack([mu + pat_effect[samples.at[s, "patient"]] for s in samples.index])
    Y += rng.normal(0, 0.3, Y.shape)
    Y[np.ix_(de, is_disease)] += DE_SHIFT
    absent = np.zeros_like(Y, dtype=bool)
    absent[np.ix_(off_c, ~is_disease)] = True
    absent[np.ix_(off_d, is_disease)] = True
    Y[absent] += OFF_SHIFT

    # ---- missingness: censoring below each run's LOD, then MCAR dropout
    lod = LOD_BASE + samples["lod_offset"].values
    p_detect = 1 / (1 + np.exp(-(Y - lod) / LOD_WIDTH))
    censored = rng.random(Y.shape) > p_detect
    mcar_rate = np.full(Y.shape, MCAR_RATE)
    mcar_rate[flaky] = FLAKY_MCAR
    mcar = (rng.random(Y.shape) < mcar_rate) & ~censored

    why = np.full(Y.shape, "observed", dtype=object)
    why[mcar] = "mcar"
    why[censored] = "censored"
    why[censored & absent] = "absent"
    X = np.where(why == "observed", Y, np.nan)

    # ---- per-sample median normalisation on observed values (as in a typical LFQ pipeline)
    X = X - np.nanmedian(X, axis=0, keepdims=True)

    # ---- write
    OUT.mkdir(parents=True, exist_ok=True)
    acc = [f"SYN{i + 1:04d}" for i in range(N_PROT)]
    gene = [f"P{i + 1:04d}" for i in range(N_PROT)]
    ids = pd.DataFrame({"PG.ProteinAccessions": acc, "PG.Genes": gene,
                        "Protein.Names": [f"Synthetic protein {i + 1:04d}" for i in range(N_PROT)]})
    lfq = pd.concat([ids, pd.DataFrame(X, columns=samples.index).round(4)], axis=1)
    lfq.to_csv(OUT / "lfq_synthetic.csv", index=False)

    truth = ids.rename(columns={"PG.ProteinAccessions": "accession", "PG.Genes": "gene"}).drop(columns="Protein.Names")
    truth["true_mu"] = mu.round(3)
    truth["mechanism"] = mechanism
    for k in ["censored", "mcar", "absent"]:
        truth[f"n_{k}"] = (why == k).sum(axis=1)
    truth.to_csv(OUT / "truth_proteins.csv", index=False)
    pd.concat([truth[["accession", "gene"]], pd.DataFrame(why, columns=samples.index)], axis=1) \
        .to_csv(OUT / "truth_cells.csv", index=False)
    # sample sheet: disease patients alternate between two subtypes; two processing batches
    disease = [p for p in patients if p.startswith("Disease")]
    subtype = {p: ("Control" if p.startswith("Control") else ("DiseaseA" if disease.index(p) % 2 == 0 else "DiseaseB"))
               for p in patients}
    sheet = samples[["patient", "group"]].copy()
    sheet["subtype"] = sheet["patient"].map(subtype)
    sheet["side"] = samples["side"]
    sheet["batch"] = np.where(np.arange(n_s) < n_s // 2, "B1", "B2")
    sheet.to_csv(OUT / "sample_sheet.csv", index_label="sample_id")
    samples[["lod_offset"]].round(3).to_csv(OUT / "truth_samples.csv", index_label="sample_id")
    rare = {"Disease07", "Disease12"}
    pd.DataFrame({"patient": patients,
                  "group": ["Control" if p.startswith("Control") else ("DiseaseRare" if p in rare else "Disease")
                            for p in patients]}).to_csv(OUT / "groups_rare.csv", index=False)

    miss = np.isnan(X)
    print(f"{N_PROT} proteins x {n_s} samples, {miss.mean():.1%} missing "
          f"(censored {np.mean(why == 'censored'):.1%}, mcar {np.mean(why == 'mcar'):.1%}, "
          f"absent {np.mean(why == 'absent'):.1%}); written to {OUT}")


if __name__ == "__main__":
    main()
