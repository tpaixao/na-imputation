# na-imputation — NA-profile classification and class-specific imputation

For label-free (LFQ) proteomics matrices (log2 intensities, proteins × samples). For each protein
it works out the mechanism most likely behind its missing values, then imputes (or drops) the
protein with a method suited to that mechanism. It can also produce a report of tables, figures and
a self-contained HTML page.

| File | Contents |
|---|---|
| `na_imputation.py` | classification, imputation, full workflow, command-line interface |
| `na_report.py` | report tables, figures and HTML (`--report`) |
| `examples/` | synthetic dataset with known mechanisms, its full output and report, and an evaluation against the truth |
| `tests/` | pytest suite (`uv run pytest`) |
| `pyproject.toml` | dependencies |

All numbers below come from the **synthetic example** in `examples/` (see [Example](#example-synthetic-data)).
No real data is included. `.gitignore` excludes `data/`, `results/`, and spreadsheets and CSVs
outside `examples/`.

---

## Background

### Types of missingness

Rubin (1976) sorts missing data by how the probability of a value being missing relates to the data:

| Type | Definition | What it looks like in LFQ proteomics |
|---|---|---|
| **MCAR** — missing completely at random | unrelated to anything, observed or not | a peptide not picked for fragmentation, a failed retention-time alignment, a spectrum not matched. Can hit abundant proteins in good runs. |
| **MAR** — missing at random | depends only on *observed* information (e.g. a known sample property), not on the missing value itself | the missingness is explained by something we can see, like the run having detected few proteins overall |
| **MNAR** — missing not at random | depends on the *unobserved value itself* | **left-censoring**: the protein is there, but below the detection limit (low abundance, or a shallow run with a higher effective limit) |

Two practical variants of MNAR matter here:

- **Censored (below detection):** the value exists but is small. It is usually the dominant
  mechanism in LFQ data: the fraction missing falls steeply with abundance.
- **Structural / biological absence ("on/off"):** the protein is absent, or far below detection,
  in one condition and present in another. The pattern *is* the signal.

Real proteins often mix mechanisms: censored in the shallow runs, randomly missing in a deep run.
So this module classifies per protein, and for `Mixed` proteins decides per cell.

MAR and MNAR can't be told apart from the data alone. Here the distinction is made through an
explicit model (abundance + run depth → P(missing)). The classes are therefore statements about
**which hypothesis explains the NAs best**, not ground truth.

### Methods for handling missing values

| Method | What it does | Suited to | Wrong for | Used here? |
|---|---|---|---|---|
| **Drop** (presence filter) | removes proteins detected in too few samples | proteins with too little data to estimate anything | on/off proteins with low overall detection (so the group screen runs first) | ✔ `Sparse (drop)` |
| **kNN** (Troyanskaya et al. 2001) | fills a gap with the distance-weighted mean of the *k* proteins with the most similar profiles, in that sample | MCAR/MAR: the value exists at a typical level | censored values (it fills in typical values and erases the low end); on/off proteins (it borrows from the group where the protein is present) | ✔ `MCAR - random`, random-looking cells of `Mixed` |
| **MinProb** (Perseus-style; Tyanova et al. 2016) | draws from a narrow normal in each sample's left tail, `N(mean_j − 1.8·sd_j, (0.3·sd_j)²)` | MNAR/censored: "below what we could detect" | abundant proteins that dropped out at random (it creates fake outliers and fake group differences) | ✔ `MNAR - censored`, `Group-structured`, censored-looking cells of `Mixed` |
| **Per-cell mixed** (after MSnbase `impute(method="mixed")`; Lazar et al. 2016) | picks MinProb or kNN *per cell*, from the detection-model probability | proteins censored in some runs and randomly missing in others | – | ✔ `Mixed` |
| **Binary presence feature** | keeps detected/not-detected as its own 0/1 variable | on/off proteins, whose information is blurred once absences become "low numbers" | – | ✔ exported for all proteins; recommended for `Group-structured` |
| **MinDet** | fills with one fixed low value (e.g. the 1% quantile) | plots, presence summaries | anything followed by a statistical test (variance collapses, so tests become anti-conservative) | ✘ |
| **QRILC** (imputeLCMD) | draws from a truncated normal fitted to each sample's left tail | censored values, when the tail shape should be learned from the data | MCAR values | ✘ (drop-in alternative to MinProb) |
| **missForest / BPCA** | iterative random-forest / Bayesian PCA reconstruction | MCAR/MAR with non-linear or low-rank structure | censored values; very small n | ✘ (for larger datasets) |
| **No imputation** | models that accept NaN (gradient-boosted trees), or hurdle models (presence × abundance) | prediction, when a complete matrix isn't needed | – | ✘ (worth comparing downstream) |

In the synthetic example, MinProb fills sit a median 0.8 log2 below the protein's lowest observed
value. All kNN fills in the deeper half of the runs land inside the protein's observed range.

## Why one imputation method for everything is wrong

In typical LFQ data the missingness is structured:

- It depends on **abundance**: low-intensity proteins are missing more often. That is left-censoring,
  i.e. missing not at random (MNAR).
- It depends on **run depth**: a shallow run detects far fewer proteins. In the example, run
  Disease04.L detects 168 proteins against up to 726 in the deepest run.
- For some proteins it depends on **group**: detected in most patients of one group and almost none
  of another.

kNN assumes values are missing at random. Applied to censored values, it fills in *typical* values
and erases the low end. MinProb assumes censoring. Applied to an abundant protein that dropped out
at random, it creates a fake outlier. So the module decides per protein, and for "Mixed" proteins
per cell.

## The six classes

| Class | NA profile | Imputation | n (example) |
|---|---|---|---|
| `Complete` | no NAs | none | 81 |
| `MCAR - random` | NAs where the detection model says the protein should have been seen | kNN | 87 |
| `MNAR - censored` | ≥ 80% of NAs explained by low abundance and/or shallow run | MinProb | 128 |
| `Mixed` | between 20% and 80% of NAs look censored | per cell: MinProb if `P_missing ≥ 0.5`, else kNN | 287 |
| `Group-structured` | on/off between groups, beyond what run depth explains | MinProb + binary presence feature | 27 |
| `Sparse (drop)` | detected in < 70% of patients in *every* group | dropped | 170 |

In the example (780 proteins with at least one observation, 24 samples), 610 proteins are kept and
2,194 cells are imputed (1,381 MinProb, 813 kNN).

## How it decides

1. **Median polish** on observed cells gives each protein an abundance `μ_i` and each sample an
   offset `s_j`. The offsets show how much the shallow runs are shifted by median normalisation (see
   caveats).
2. **Detection model:** a logistic regression over all cells,
   `logit P(missing_ij) = a_j − b·μ_i`, with a per-sample intercept `a_j` that captures run depth.
   This gives every cell a probability `P_missing`, i.e. how strongly the censoring hypothesis
   *expected* that cell to be missing. In the example: b = 0.91 (+1 log2 → ×2.5 odds of
   detection), AUC 0.89.
3. **Per-protein profile:**
   - `frac_censored` is the share of the protein's NAs with `P_missing ≥ censor_cell_p`.
   - Patient-level presence per group, a group test on the k × 2 detected / not-detected table
     (Fisher exact for k = 2, Monte Carlo Freeman–Halton for k > 2), and a depth-adjusted group
     difference (observed minus model-expected detection).
   - For k groups, the "difference" is a **spread** between the group with the highest presence
     (`group_high`) and the one with the lowest (`group_low`). With two groups this is exactly
     |presence_A − presence_B|.
4. **Decision rules**, applied in this order:
   1. `n_missing == 0` → **Complete**
   2. `group_p < group_p` (config), **and** `presence_spread ≥ group_delta`, **and**
      `resid_spread ≥ group_resid_delta` → **Group-structured**. `resid_spread` is taken between
      the same two groups, so it must point the same way as the presence difference.
   3. `max_presence < presence_threshold` → **Sparse (drop)**
   4. `frac_censored ≥ mnar_frac` → **MNAR - censored**
   5. `frac_censored ≤ mcar_frac` → **MCAR - random**
   6. otherwise → **Mixed**

   The group screen runs *before* the sparsity filter, so on/off proteins aren't thrown away. It is
   a **raw-p screen**; with few patients per group, expect nothing to survive FDR correction.

Implementation notes: kNN is `sklearn.impute.KNNImputer` on the protein × sample matrix
(`k = 10`, distance-weighted, observed values as donors only). MinProb parameters come from each
sample's own observed values, and the draws use a fixed seed.

---

## Input format

A table with one row per protein: `.csv`, `.tsv`/`.txt` or Excel (first sheet unless `--sheet` is
given). Columns:

| Column | Meaning |
|---|---|
| `PG.ProteinAccessions` | protein accession (required) |
| `PG.Genes` | gene name (optional; missing values fall back to the accession) |
| `Protein.Names` | description (optional) |
| `Median` | optional; ignored |
| one column per sample | log2 intensity, empty = missing, named by sample ID |

Every other column is treated as a sample **if it contains at least one number**. So remove
numeric annotation columns (e.g. precursor counts) before running. Text-only columns are ignored
and empty columns dropped, both with a note.

### Sample metadata

Each sample needs a **patient** and a **group**. There are three ways to provide them, from most
to least explicit.

**1. A sample sheet (`--sample-sheet`, recommended).** A `.csv`, `.tsv` or Excel file with one row
per sample:

```csv
sample_id,patient,group,subtype,side,batch
Control01.R,Control01,Control,Control,R,B1
Disease05.L,Disease05,Disease,DiseaseA,L,B1
Disease05.R,Disease05,Disease,DiseaseA,R,B1
...
```

| Column | Required | Meaning |
|---|---|---|
| `sample_id` | yes | must match the matrix column names |
| `patient` | no | samples sharing it are one patient: presence is counted once per patient, the group test works at patient level, and labels must agree within a patient. Replaces the patient parsed from the sample ID. |
| any other | no | available as a grouping with `--group-col <column>` (e.g. `group`, `subtype`, `batch`) and listed in the report's `sample_qc` table |

With a sample sheet, sample IDs can be anything (`S01`, `run_2024_117`, …).
- The module stops with an error if a matrix sample has no row in the sheet, or if `--group-col`
  names a column that doesn't exist.
- It also stops if a patient's samples carry different labels.
- Rows for samples not in the matrix are ignored, with a note.
- `n_detected`, `na_group` and `polish_shift` are reserved column names.

```bash
python na_imputation.py --lfq matrix.csv --sample-sheet samples.csv                     # group column "group"
python na_imputation.py --lfq matrix.csv --sample-sheet samples.csv --group-col subtype
```

**2. A two-column groups file (`--groups-file`).** `(sample_id or patient, group)`; see
[More than two groups](#more-than-two-groups). It overrides `--group-col`. It can be combined
with a sample sheet, which then only provides the patients.

**3. Parsed from the sample IDs (fallback).** IDs are dot-separated:
`PatientID[.fields…][.L|R][.diagnosis…]`, e.g. `Disease05.L`.
- The patient ID is the first field.
- The default `group` is its leading letters: `Disease05` → `Disease`.
- An `L`/`R` field marks the side; the fields after it form `diagnosis`, usable with
  `--group-col diagnosis`.

A sample sheet overrides any of these parsed fields.

Non-numeric cells in sample columns are reported (the first 10, then a count) and treated as
missing. Proteins with no observed value at all are reported and removed. Duplicated column names,
fewer than two sample columns, or a matrix with no missing values stop the run with an error.

## Requirements

Python ≥ 3.10 with `numpy`, `pandas`, `openpyxl`, `scipy`, `scikit-learn`, `statsmodels` and
`matplotlib ≥ 3.7` (for reports). All are listed in `pyproject.toml`, so from the repository folder:

```bash
uv run na_imputation.py --lfq examples/synthetic/lfq_synthetic.csv --out-dir results/demo --report
```

`uv` creates the environment on first use. Alternatively, `pip install numpy pandas openpyxl scipy
scikit-learn statsmodels matplotlib` and run it with `python`.

Tests: `uv run pytest` (about 1 minute).
- **Loading:** CSV / TSV / Excel give identical results; messy columns; numeric sample names.
- **Metadata:** groups files and sample sheets, including every validation error.
- **Statistics:** the group tests are checked against SciPy, and median polish is checked on
  exact data.
- **Workflow:** observed values untouched, one method per missing cell, classes follow the
  documented rules, reproducible with a seed; k groups and minimum group size.
- **Edge cases:** nothing retained, a sample with one value, a complete matrix.
- **CLI and report.**

Scale: a 5,864-protein × 100-sample matrix runs in about 70 s with under 400 MB peak memory (the
detection model uses a sparse design matrix).

## Command line

```bash
python na_imputation.py --lfq matrix.csv                       # -> results/
python na_imputation.py --lfq matrix.xlsx --sheet 2 --report   # + results/report/
python na_imputation.py --help                                 # list every option

# sensitivity run: looser presence filter, stricter censoring call, separate output folder
python na_imputation.py --lfq matrix.csv \
    --presence-threshold 0.5 --censor-cell-p 0.6 --out-dir results/sens_p50_c60 --report
```

| Option | Default | Meaning |
|---|---|---|
| `--lfq` | required | input matrix (`.csv`, `.tsv`/`.txt`, Excel) |
| `--sheet` | first sheet | Excel sheet name or index |
| `--out-dir` | `results/` | where the four CSVs are written |
| `--report` | off | also write tables, figures and `report.html` to `<out-dir>/report/` |
| `--group-col` | `group` | metadata column holding the group: any sample-sheet column, or `group` (patient-ID prefix) / `diagnosis` parsed from the sample ID |
| `--sample-sheet` | – | per-sample metadata (`.csv`/`.tsv`/Excel) with a `sample_id` column; optional `patient`; any column usable with `--group-col` |
| `--groups-file` | – | CSV `(sample_id or patient, group)` with custom labels; overrides `--group-col` |
| `--min-group-size` | `3` | groups with fewer patients are left out of the group screen and the presence rule |
| `--presence-threshold` | `0.70` | keep a protein if detected in ≥ this fraction of patients in ≥ 1 group |
| `--group-p` | `0.05` | raw p for the group screen (Fisher if k = 2, Freeman–Halton if k > 2) |
| `--group-delta` | `0.50` | minimum presence spread (highest − lowest group) |
| `--group-resid-delta` | `0.25` | minimum depth-adjusted spread between the same two groups |
| `--n-perm` | `10000` | Monte Carlo permutations for the k > 2 test |
| `--censor-cell-p` | `0.50` | a missing cell is "censored-like" if `P_missing ≥` this |
| `--mnar-frac` | `0.80` | `frac_censored ≥` this → MNAR |
| `--mcar-frac` | `0.20` | `frac_censored ≤` this → MCAR (in between → Mixed) |
| `--knn-k` | `10` | number of neighbour proteins for kNN |
| `--minprob-shift` | `1.8` | MinProb down-shift, in sample SDs |
| `--minprob-width` | `0.3` | MinProb width, in sample SDs |
| `--seed` | `20260925` | RNG seed for the MinProb draws and the permutations |

## Python API

```python
from na_imputation import NAConfig, run_workflow      # run from the repository folder (or add it to sys.path)

res = run_workflow("matrix.csv")                                           # defaults, writes results/
res = run_workflow("matrix.csv", cfg=NAConfig(censor_cell_p=0.4), out_dir=None, verbose=False)  # no files
res = run_workflow("matrix.csv", cfg=NAConfig(group_col="diagnosis"), out_dir=None)            # other grouping
res = run_workflow("matrix.csv", sample_sheet="samples.csv", cfg=NAConfig(group_col="subtype"))  # sample sheet
res = run_workflow("matrix.csv", sample_sheet=samples_df, out_dir=None)   # DataFrame with a sample_id column
res = run_workflow("matrix.csv", groups="my_groups.csv", out_dir=None)                         # custom labels
res = run_workflow("matrix.csv", groups=my_series, out_dir=None)   # pd.Series indexed by sample_id
res = run_workflow("matrix.csv", report=True)                      # + results/report/
```

`run_workflow()` returns an `NAResult` (shapes for the example: 780 proteins, 24 samples, 610 retained):

| Attribute | Shape | Contents |
|---|---|---|
| `X` | 780 × 24 | raw log2 matrix, NaN = missing |
| `prot` | 780 × 3 | `accession`, `gene`, `protein_name` |
| `meta` | 24 × 7 | patient, group, side, diagnosis, `n_detected`, `na_group` (the labels used), `polish_shift` |
| `mu`, `sample_shift` | 780 / 24 | median-polish protein abundance and sample offset |
| `P_miss` | 780 × 24 | detection-model P(missing) per cell |
| `profile` | 780 × (18 + k) | per-protein statistics + `na_class` + `imputation` (see below) |
| `method` | 610 × 24 | per cell: `"MinProb"`, `"kNN"` or `""` (observed); retained proteins only |
| `X_imputed` | 610 × 24 | imputed matrix; retained proteins only |
| `presence` | 780 × 24 | 0/1 detection matrix |
| `groups_used` | list | groups that entered the group screen (≥ `min_group_size` patients) |
| `group_test` | str | name of the test behind `group_p` |
| `cfg` | `NAConfig` | the settings used |
| `det_model` | `LogisticRegression` | the fitted detection model |

Rows in every table share the same integer index (the row number in the input), so they line up
directly. Gene names are not used as the index because they aren't guaranteed unique.

The building blocks can also be used one at a time, for example to run the pipeline inside a
cross-validation fold:

| Function | Does |
|---|---|
| `load_lfq(path, sheet=0)` | read the matrix → `(X, prot, meta)`; reports non-numeric cells and empty proteins |
| `read_matrix(path, sheet=0)` | read `.csv` / `.tsv` / Excel into a DataFrame |
| `parse_sample_id(id)` | `Disease05.L` → patient, group, side, diagnosis |
| `load_sample_sheet(path, sheet=0)` | read a sample sheet → DataFrame indexed by `sample_id` |
| `prepare_sample_sheet(df)` | validate a sample-sheet DataFrame (`sample_id` column or index) → indexed copy |
| `apply_sample_sheet(meta, sample_sheet)` | merge its columns into the metadata (validated; `patient` overrides) |
| `load_groups(path, meta)` | read a custom groups CSV → per-sample labels |
| `resolve_groups(meta, cfg, groups=None)` | pick the labels and check they're constant within each patient |
| `median_polish(X)` | → `(mu, sample_shift)` |
| `fit_detection_model(X, mu)` | → `(P_miss, fitted LogisticRegression)` |
| `na_profile(X, prot, meta, mu, P_miss, groups, cfg)` | per-protein statistics → `(profile, groups_used, test_name)` |
| `presence_group_test(present, labels, n_perm, rng)` | Fisher (k = 2) or Freeman–Halton (k > 2) per protein |
| `classify_proteins(profile, cfg)` | → categorical `na_class` series (`classify_protein` for one row) |
| `assign_methods(M, P_miss, classes, cfg)` | → per-cell method table |
| `knn_impute(X, k)` | protein-wise kNN fill of every NaN |
| `minprob_params(X_ref, cfg)` | per-sample MinProb `loc` / `scale` |
| `impute_by_class(X, profile, P_miss, cfg, X_ref=None)` | drop Sparse proteins, fill each cell with its method → `(X_imputed, method)` |
| `save_outputs(res, out_dir)` | write the four CSVs |

## Report (`--report`)

`--report` (or `run_workflow(..., report=True)`) writes a report folder next to the CSVs:

```
<out-dir>/report/
├── report.html          # self-contained: all figures embedded, opens offline, follows dark mode
├── figures/*.png
└── tables/*.csv
```

**Example:** [`examples/synthetic/output/report/`](examples/synthetic/output/report/) is the report
for the synthetic data. Download `report.html` and open it in a browser; GitHub shows HTML as
source, not as a page. The figures display directly on GitHub.

The HTML page starts with a run summary: input, matrix size, groups (with excluded ones marked),
group test, number of retained proteins and imputed cells, and every setting. Six sections follow:

| Section | Figures | Tables |
|---|---|---|
| Overview | `classes`: proteins per class; classes in abundance × missingness space | `class_summary`: n, share and median statistics per class, plus the method used. `imputation_counts`: MinProb / kNN cells per class |
| Samples and run depth | `sample_depth`: proteins detected per sample, coloured by group | `sample_qc`: per sample: group, depth, median-polish offset, median of the complete proteins (normalisation check), MinProb mean/sd, MinProb and kNN cell counts, share of cells imputed |
| Detection model | `missing_vs_abundance`; `detection_model`: calibration plot (with AUC) and fitted curves for the deepest, median and shallowest run | `detection_model`: slope, odds ratio per log2 unit, AUC. `calibration`: predicted vs observed P(missing) in 20 bins |
| NA classes | `missingness_map`: missing cells, proteins grouped by class × samples by depth. `class_examples`: one protein per class, observed values coloured by group, NAs marked censored-like / random-like | – |
| Group-structured proteins | `group_structured`: per-group presence for each on/off protein (hollow = excluded small group) | `group_structured`: presence per group, high/low group, spreads, p, q |
| Imputation | `imputed_distributions`: observed vs kNN vs MinProb values, one panel per sample | – |

Pieces can also be built one at a time, e.g. in a notebook:

```python
from na_imputation import run_workflow
from na_report import make_report, report_tables, report_figures

res = run_workflow("examples/synthetic/lfq_synthetic.csv", out_dir=None)
tables = report_tables(res)                                 # dict of DataFrames
figs = report_figures(res, examples={"Mixed": "P0038"})     # dict of matplotlib Figures; pick example genes
figs["classes"]                                             # displays inline in Jupyter
make_report(res, "results/report", examples={"Mixed": "P0038"})
```

Figures use matplotlib's object-oriented API (no `pyplot`). They work headless and leave a
notebook's plotting backend alone. Group colours come from a fixed 8-colour categorical order; a 9th
group or later is drawn in grey.

## Outputs

| File | Rows | Contents |
|---|---|---|
| `protein_na_classification.csv` | all proteins | the profile table, sorted by class then `group_p` |
| `LFQ_imputed_by_na_class.csv` | retained proteins | `accession`, `gene`, then one imputed column per sample |
| `LFQ_imputation_method_per_cell.csv` | retained proteins | same layout; each cell is `observed`, `kNN` or `MinProb` (for sensitivity analyses or masking imputed cells) |
| `LFQ_presence_matrix.csv` | all proteins | same layout; 1 = detected, 0 = missing (use as binary features for Group-structured proteins) |

### Columns of `protein_na_classification.csv`

| Column | Meaning |
|---|---|
| `accession`, `gene`, `protein_name` | identifiers (missing gene names fall back to the accession) |
| `n_missing`, `frac_missing` | NAs across the samples |
| `mu_hat` | protein abundance from median polish (log2) |
| `frac_censored` | share of this protein's NAs with `P_missing ≥ censor_cell_p` (NaN if complete) |
| `mean_p_at_missing` | mean `P_missing` over its missing cells |
| `depth_auc` | AUC of run depth for detected vs missing samples: 1 = NAs only in the shallowest runs, 0.5 = unrelated to depth |
| `presence_<group>` | one column per group (e.g. `presence_control`, `presence_disease`): fraction of patients with ≥ 1 detection (multiple samples of a patient count once). Includes excluded small groups. |
| `max_presence` | highest presence over the groups used (all patients pooled if fewer than 2 groups qualify); drives the Sparse rule |
| `group_high`, `group_low` | groups with the highest / lowest presence |
| `presence_spread` | `presence(group_high) − presence(group_low)` |
| `resid_spread` | depth-adjusted spread: mean (observed − expected detection) in `group_high` minus in `group_low` |
| `group_p`, `group_q` | group test on patient-level presence (Fisher exact for k = 2, Monte Carlo Freeman–Halton for k > 2); BH-adjusted q |
| `na_class` | one of the six classes |
| `imputation` | the method applied to that class |

---

## Example (synthetic data)

`examples/` holds a synthetic dataset where the true mechanism behind every missing value is known,
so the method can be checked against the truth.

```bash
uv run examples/make_synthetic_data.py      # -> examples/synthetic/*.csv
uv run na_imputation.py --lfq examples/synthetic/lfq_synthetic.csv \
    --sample-sheet examples/synthetic/sample_sheet.csv --out-dir examples/synthetic/output --report
uv run examples/evaluate_synthetic.py       # -> examples/synthetic/output/evaluation_*.csv
```

**What is simulated** (`make_synthetic_data.py`, seed 7):
- 20 patients: `Control01`–`08` and `Disease01`–`12`. Four patients have two samples (L/R),
  giving 24 samples.
- 800 proteins with true abundance ~ N(22, 2.5) log2, a patient effect and technical noise.
- Detection is left-censored below a per-run limit. Three runs are shallow:
  - Disease04.L: 168 proteins detected
  - Control06.R: 250
  - Disease11.L: 325

  The other runs detect up to 726.
- On top of censoring, 2% of cells drop out at random.
- Built-in protein types:
  - 40 proteins with differential abundance (Disease +1.5 log2)
  - 20 on/off proteins, truly absent in one group: 12 in Control, 8 in Disease
  - 30 "flaky" abundant proteins with 20% random dropout
- Each sample is median-normalised on its observed values, as a typical pipeline does. This
  reproduces the down-shift of shallow runs: median-polish offsets of −2.3, −1.8 and −1.3 log2.

This gives 28.0% missing overall. Twenty proteins are never observed and are removed on loading.

**Files:** `lfq_synthetic.csv` (the input), `truth_proteins.csv` (built-in mechanism per protein),
`truth_cells.csv` (why each value is missing: `observed` / `censored` / `mcar` / `absent`),
`truth_samples.csv` (simulated detection-limit offset per run), `sample_sheet.csv` (patient,
group, subtype, side and batch per sample), `groups_rare.csv` (the minimum-group-size example
below), and `output/` (the four CSVs, the report and the evaluation).

**How well it recovers the truth** (`evaluate_synthetic.py`):

| Check | Result |
|---|---|
| on/off proteins called `Group-structured` | 20 / 20 |
| `Group-structured` calls that are true on/off | 20 / 27 (7 false positives: 3 differential-abundance, 3 ordinary, 1 flaky) |
| "flaky" (random-dropout) proteins called `MCAR - random` | 27 / 30 |
| truly random missing cells (MCAR), among retained proteins, given kNN | 95% |
| censored / absent cells, among retained proteins, given MinProb | 75% |

The miss rate for censored cells is the realistic weak point. Censored values just below the
detection limit in a *deeper* run look, to the model, like random dropouts: the protein "should"
have been seen there. Those cells get kNN, i.e. a typical value rather than a low one. Stricter
settings (`--censor-cell-p 0.4`, `--mcar-frac 0.1`) trade this against more MinProb fills for
genuinely random NAs. The per-cell tables in `output/evaluation_by_cell.csv` make that trade-off
visible.

---

## More than two groups

Only the group screen and the presence rule depend on the groups. The detection model, the
censoring score, kNN and MinProb don't see group labels at all: groups are deliberately left out of
the detection model, so it can't absorb a real group effect.

**Choosing the groups**

```bash
python na_imputation.py --lfq matrix.csv --sample-sheet samples.csv --group-col subtype --report
python na_imputation.py --lfq matrix.csv --group-col diagnosis --out-dir results/by_diagnosis --report
python na_imputation.py --lfq matrix.csv --groups-file groups.csv --out-dir results/custom --report
```

The sample sheet is described under [Sample metadata](#sample-metadata). A `groups.csv` has two
columns. The first lists every **sample ID** or every **patient ID**; the second gives the group
(see `examples/synthetic/groups_rare.csv`):

```csv
patient,group
Control01,Control
Disease06,Disease
Disease07,DiseaseRare
...
```

Labels must be the same for all samples of a patient. The module stops with an error otherwise.

**What changes with k groups**

| Step | k = 2 | k > 2 |
|---|---|---|
| group test | two-sided Fisher exact | Fisher's exact test extended to k × 2 (Freeman–Halton), p-value estimated from `n_perm` shuffles of the patient labels (both margins fixed); checked against `scipy.stats.fisher_exact` R×C to within Monte Carlo error |
| effect size | \|presence_A − presence_B\| | `presence_spread` = highest − lowest group presence |
| depth-adjusted check | residual difference, same sign | `resid_spread` between the same two groups (`group_high` − `group_low`) |
| Sparse rule | max over the 2 groups | max over all groups used |

**Minimum group size.** Groups with fewer than `min_group_size` patients (default 3) are left out of
the group screen and the presence rule. Their samples still go into the detection model and are
imputed. The reason: in a tiny group presence can only be 0, ½ or 1, so a single patient can
decide the Sparse rule and the on/off call. On the synthetic data:

| Run | Groups (patients) | Test | Group-structured | Sparse |
|---|---|---|---|---|
| default | Control 8, Disease 12 | Fisher | 27 | 170 |
| `--sample-sheet sample_sheet.csv --group-col subtype` | Control 8, DiseaseA 6, DiseaseB 6 | Freeman–Halton | 29 | 150 |
| `--groups-file groups_rare.csv` | Control 8, Disease 10, DiseaseRare 2 (excluded) | Fisher | 29 | 148 |
| `--groups-file groups_rare.csv --min-group-size 1` | all three | Freeman–Halton | 29 | **130** |

The last row shows the problem: counting a 2-patient group lets proteins seen in just those two
patients escape the Sparse filter. Merge rare groups (or rely on the minimum size) rather than
lowering it.

**Things to keep in mind**
- More groups at the same total n means fewer patients per group and less power.
- `group_high` / `group_low` tell you *which* groups differ. The test itself is global ("detection
  depends on group"), not pairwise.
- Groupings that overlap (e.g. treatment history and diagnosis) are confounded. A protein that is
  on/off by one may really be on/off by the other.

## Caveats

- **Don't impute the full dataset before cross-validation.** `run_workflow()` fits the detection
  model, the MinProb parameters and the **group screen, which uses the group labels**, on
  every sample. That is fine for exploration. For a classifier, run the steps inside each
  patient-grouped fold, so labels and held-out samples never shape the imputation. The module has
  no sklearn-style `fit`/`transform` yet.
- **Shallow runs are offset by median normalisation.** When normalisation uses only each sample's
  own observed proteins, a shallow run (which sees mostly abundant proteins) is shifted down
  compared with deep runs; check `meta["polish_shift"]` and the `sample_qc` table. MinProb works in
  each sample's own scale, but kNN fills in those runs pick up the offset. Consider re-normalising
  on a common protein set (e.g. the complete proteins) upstream.
- **Shallow runs end up mostly imputed.** In the example, 72% of Disease04.L's cells among
  retained proteins are imputed. Run a sensitivity analysis without such runs.
- **Group-structured calls are hypotheses.** They are a raw-p screen; in the example 7 of 27 are
  false positives. Check sample handling or contamination before interpreting on/off proteins
  biologically.
- **The Sparse rule is strict by design.** It removes proteins seen in few patients. Check key
  proteins with `--presence-threshold 0.5` as well.
- **MinProb is random.** For inference, repeat with several `--seed` values (multiple imputation)
  rather than relying on a single draw.

## References

- Rubin DB (1976) Inference and missing data. *Biometrika* 63:581–592.
- Lazar C, et al. (2016) Accounting for the multiple natures of missing values in label-free
  quantitative proteomics data sets to compare imputation strategies. *J Proteome Res* 15:1116–1125.
- Tyanova S, et al. (2016) The Perseus computational platform for comprehensive analysis of
  (prote)omics data. *Nat Methods* 13:731–740.
- Troyanskaya O, et al. (2001) Missing value estimation methods for DNA microarrays.
  *Bioinformatics* 17:520–525.
- MSnbase `impute(method = "mixed")`; imputeLCMD `impute.MinProb` / `impute.QRILC`.
