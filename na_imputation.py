"""
NA-profile classification and class-specific imputation for label-free (LFQ) proteomics.

Each protein is labelled by the mechanism most likely behind its missing values, and
imputed accordingly:

    Complete          no NAs                                   -> none
    MCAR - random     NAs where the protein should be seen     -> kNN (protein-wise)
    MNAR - censored   NAs explained by abundance / run depth   -> MinProb (per sample)
    Mixed             some of each                             -> per cell (MinProb if p>=cut else kNN)
    Group-structured  on/off between groups (k >= 2)           -> MinProb + binary presence feature
    Sparse (drop)     < 70% presence in every group            -> dropped

Groups default to the letter prefix of the patient ID (Control01 -> Control); any per-sample
label works (--group-col diagnosis, or --groups-file). Groups with fewer than --min-group-size patients are left out of the group
screen and the presence rule (their samples are still modelled and imputed). The group test
is Fisher's exact test for 2 groups and its k x 2 extension (Freeman-Halton, Monte Carlo over
patient-label permutations) for k > 2.

Usage as a script (writes the four CSVs, plus figures/tables/HTML with --report):
    python na_imputation.py --lfq MATRIX.{csv,tsv,xlsx} [--out-dir DIR] [--report] ...

Usage as a module:
    from na_imputation import NAConfig, run_workflow
    res = run_workflow("matrix.csv", cfg=NAConfig(censor_cell_p=0.4), out_dir=None)   # don't save
    res.profile, res.X_imputed, res.method, res.presence
"""
import argparse
import re
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.multitest import multipletests

DEFAULT_OUT = Path("results")  # relative to the working directory
ID_COLS = ("PG.ProteinAccessions", "PG.Genes", "Protein.Names", "Median")

CLASS_ORDER = ["Complete", "MCAR - random", "MNAR - censored", "Mixed", "Group-structured", "Sparse (drop)"]


@dataclass
class NAConfig:
    group_col: str = "group"          # column of meta holding the group label ("group" = patient-ID prefix)
    min_group_size: int = 3           # groups with fewer patients are ignored by the group screen/presence rule
    presence_threshold: float = 0.70  # keep if detected in >= this fraction of patients in >= 1 group
    group_p: float = 0.05             # raw p for the group screen (Fisher if k = 2, permutation if k > 2)
    group_delta: float = 0.50         # presence spread: max - min group presence
    group_resid_delta: float = 0.25   # depth-adjusted spread: resid(highest group) - resid(lowest group)
    n_perm: int = 10000               # permutations for the k > 2 group test
    censor_cell_p: float = 0.50       # a missing cell is censored-like if P(missing | model) >= this
    mnar_frac: float = 0.80           # >= this share of censored-like NAs -> MNAR
    mcar_frac: float = 0.20           # <= this share -> MCAR; in between -> Mixed
    knn_k: int = 10
    minprob_shift: float = 1.8        # Perseus-style down-shift, in sample SDs
    minprob_width: float = 0.3        # Perseus-style width, in sample SDs
    seed: int = 20260925


@dataclass
class NAResult:
    X: pd.DataFrame           # raw log2 matrix (proteins x samples), NaN = missing
    prot: pd.DataFrame        # accession / gene / protein_name per row
    meta: pd.DataFrame        # per-sample metadata (+ n_detected)
    mu: pd.Series             # protein abundance from median polish
    sample_shift: pd.Series   # sample effect from median polish (normalisation offset)
    P_miss: pd.DataFrame      # detection-model P(missing) per cell
    profile: pd.DataFrame     # per-protein NA statistics + na_class + imputation
    method: pd.DataFrame      # per-cell imputation method ("" = observed), retained proteins only
    X_imputed: pd.DataFrame   # imputed matrix, retained proteins only
    presence: pd.DataFrame    # 0/1 detection matrix, all proteins
    groups_used: list         # groups that entered the group screen (>= min_group_size patients)
    group_test: str           # name of the test used for group_p
    cfg: NAConfig = None      # configuration the result was produced with


# ------------------------------------------------------------------ loading

def parse_sample_id(sample_id: str) -> dict:
    """`PatientID[.fields...][.L|R][.Diagnosis...]` -> dict.

    patient = first dot-separated field; group = its leading letters (Control01 -> Control,
    Case7 -> Case); side = first "L"/"R" field (samples of one patient share the patient ID);
    diagnosis = fields after the side, joined with "+".
    """
    parts = sample_id.split(".")
    patient = parts[0]
    prefix = re.match(r"[A-Za-z]+", patient)
    side_idx = next((i for i, p in enumerate(parts) if p in ("L", "R")), None)
    return dict(sample_id=sample_id, patient=patient,
                group=prefix.group(0) if prefix else patient,
                side=parts[side_idx] if side_idx is not None else "",
                diagnosis="+".join(parts[side_idx + 1:]) if side_idx is not None else "")


def read_matrix(path, sheet=0) -> pd.DataFrame:
    """Read the proteins x samples table from .csv, .tsv/.txt or Excel (first sheet by default)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    return pd.read_excel(path, sheet_name=sheet)


def load_lfq(path, sheet=0, verbose=True):
    """Return (X, prot, meta). Non-numeric cells are reported and set to NaN.

    Expected columns: PG.ProteinAccessions, PG.Genes, Protein.Names (optional: Median), then one
    log2 intensity column per sample, named by sample ID (see parse_sample_id).
    """
    raw = read_matrix(path, sheet)
    sample_cols = [c for c in raw.columns if c not in ID_COLS]
    meta = pd.DataFrame([parse_sample_id(c) for c in sample_cols]).set_index("sample_id")

    X_raw = raw[sample_cols]
    X = X_raw.apply(pd.to_numeric, errors="coerce")
    if verbose:
        for r, c in zip(*np.where(X.isna() & X_raw.notna())):
            print(f"[data-quality] non-numeric cell coerced to NaN: gene={raw['PG.Genes'][r]} "
                  f"sample={sample_cols[c]} value={X_raw.iat[r, c]!r}")

    prot = raw[["PG.ProteinAccessions", "PG.Genes", "Protein.Names"]].rename(
        columns={"PG.ProteinAccessions": "accession", "PG.Genes": "gene", "Protein.Names": "protein_name"})
    prot["gene"] = prot["gene"].fillna(prot["accession"])
    X.index = prot.index

    empty = X.isna().all(axis=1)
    if empty.any():  # no observed value at all: nothing to model or impute
        if verbose:
            print(f"[data-quality] removed {int(empty.sum())} protein(s) with no observed value")
        X, prot = X[~empty], prot[~empty]
    meta["n_detected"] = X.notna().sum(axis=0)
    return X, prot, meta


def load_groups(path, meta: pd.DataFrame) -> pd.Series:
    """Read a 2-column CSV (id, group) keyed by sample_id or by patient -> per-sample labels."""
    g = pd.read_csv(path)
    if g.shape[1] < 2:
        raise ValueError(f"{path}: expected two columns (sample_id or patient, group)")
    ids, labels = g.iloc[:, 0].astype(str), g.iloc[:, 1].astype(str)
    lookup = dict(zip(ids, labels))
    if set(meta.index) <= set(ids):
        out = meta.index.to_series().map(lookup)
    elif set(meta["patient"]) <= set(ids):
        out = meta["patient"].map(lookup)
    else:
        raise ValueError(f"{path}: first column must list every sample_id or every patient")
    return out.rename("na_group")


def resolve_groups(meta: pd.DataFrame, cfg: NAConfig, groups=None) -> pd.Series:
    """Per-sample group labels (from `groups` if given, else meta[cfg.group_col]); must be constant per patient."""
    g = meta[cfg.group_col] if groups is None else pd.Series(groups).reindex(meta.index)
    if g.isna().any():
        raise ValueError(f"no group label for samples: {list(g.index[g.isna()])}")
    g = g.astype(str).rename("na_group")
    mixed = g.groupby(meta["patient"]).nunique()
    if (mixed > 1).any():
        raise ValueError(f"group label differs between samples of the same patient: {list(mixed.index[mixed > 1])}")
    return g


# ------------------------------------------------------------------ detection model

def median_polish(X: pd.DataFrame, n_iter: int = 30):
    """Additive fit X_ij = mu_i + s_j on observed cells. Returns (mu, s), median(s) = 0."""
    mu = X.median(axis=1)
    s = pd.Series(0.0, index=X.columns)
    for _ in range(n_iter):
        s = X.sub(mu, axis=0).median(axis=0)
        s -= s.median()
        mu = X.sub(s, axis=1).median(axis=1)
    return mu, s


def fit_detection_model(X: pd.DataFrame, mu: pd.Series):
    """logit P(missing_ij) = a_j - b * mu_i. Returns (P_miss DataFrame, fitted model)."""
    n_p, n_s = X.shape
    design = np.column_stack([np.repeat(mu.values, n_s), np.tile(np.eye(n_s), (n_p, 1))])
    y = X.isna().values.ravel().astype(int)
    model = LogisticRegression(C=1e3, max_iter=5000).fit(design, y)
    P_miss = pd.DataFrame(model.predict_proba(design)[:, 1].reshape(n_p, n_s), index=X.index, columns=X.columns)
    return P_miss, model


# ------------------------------------------------------------------ per-protein profile & classes

def _depth_auc(miss: np.ndarray, depth: np.ndarray) -> float:
    """1 = NAs only in the shallowest runs, 0.5 = unrelated to run depth."""
    if miss.all() or not miss.any():
        return np.nan
    u = stats.mannwhitneyu(depth[~miss], depth[miss]).statistic
    return u / ((~miss).sum() * miss.sum())


def _col(label: str) -> str:
    return "presence_" + "".join(ch if ch.isalnum() else "_" for ch in label.lower())


def presence_group_test(present: np.ndarray, labels: np.ndarray, n_perm: int, rng):
    """Test detection vs group per protein, on patient-level 0/1 presence.

    present: proteins x patients (bool); labels: group per patient.
    k = 2 -> two-sided Fisher exact test.
    k > 2 -> Monte Carlo Freeman-Halton test (Fisher's exact test for k x 2 tables): patient
             labels are shuffled (both margins fixed) and p = share of tables at most as
             probable as the observed one. Under fixed margins log P(table) =
             sum_g log C(n_g, a_g) + const, so that sum is the statistic.
    Returns (p-values, test name). Proteins with no variation get p = 1.
    """
    labels = np.asarray(labels, dtype=object).astype(str)
    groups = np.unique(labels)
    k = len(groups)
    onehot = (labels[:, None] == groups[None, :]).astype(float)   # patients x k
    n_g = onehot.sum(axis=0)
    P = present.astype(float)

    if k == 2:
        a = P @ onehot                                              # proteins x 2 detected counts
        p = np.array([stats.fisher_exact([[x, n_g[0] - x], [y, n_g[1] - y]])[1] for x, y in a])
        return p, "Fisher exact (2x2)"

    total = P.sum(axis=1)
    ok = (total > 0) & (total < P.shape[1])
    lg_n = gammaln(n_g + 1)

    def log_prob(oh):  # sum_g log C(n_g, a_g), up to a per-protein constant
        a = P[ok] @ oh
        return (lg_n - gammaln(a + 1) - gammaln(n_g - a + 1)).sum(axis=1)

    t_obs = log_prob(onehot)
    extreme = np.zeros(ok.sum())
    for _ in range(n_perm):
        extreme += log_prob(onehot[rng.permutation(len(labels))]) <= t_obs + 1e-9
    p = np.ones(len(P))
    p[ok] = (extreme + 1) / (n_perm + 1)
    return p, f"Freeman-Halton, Monte Carlo ({k}x2, {n_perm} perms)"


def na_profile(X, prot, meta, mu, P_miss, groups: pd.Series, cfg: NAConfig):
    """Per-protein NA statistics: censoring share, depth AUC, group presence and tests.

    groups: per-sample label (see resolve_groups). Returns (profile, groups_used, test_name).
    """
    M = X.isna()
    n_s = X.shape[1]
    depth = meta["n_detected"].reindex(X.columns).values

    cell_p = P_miss.where(M)
    n_miss = M.sum(axis=1)
    frac_censored = (cell_p >= cfg.censor_cell_p).sum(axis=1) / n_miss.replace(0, np.nan)
    auc_depth = pd.Series([_depth_auc(m, depth) for m in M.values], index=X.index)

    # group presence at patient level (bilateral: detected if seen on either side)
    patient = meta["patient"].reindex(X.columns)
    pat_present = (~M).T.groupby(patient).any().T
    pat_group = groups.groupby(patient).first().reindex(pat_present.columns)
    sizes = pat_group.value_counts()
    used = sorted(sizes.index[sizes >= cfg.min_group_size])

    presence = pd.DataFrame({g: pat_present.loc[:, pat_group == g].mean(axis=1) for g in sorted(sizes.index)})
    out = prot.assign(n_missing=n_miss, frac_missing=n_miss / n_s, mu_hat=mu.round(3),
                      frac_censored=frac_censored, mean_p_at_missing=cell_p.mean(axis=1), depth_auc=auc_depth)
    for g in presence.columns:
        out[_col(g)] = presence[g]
    out["max_presence"] = presence[used].max(axis=1) if used else pat_present.mean(axis=1)

    if len(used) >= 2:
        pres_used = presence[used]
        hi, lo = pres_used.values.argmax(axis=1), pres_used.values.argmin(axis=1)

        # depth-adjusted: observed minus model-expected detection, mean per group (sample level)
        resid = (~M).astype(float) - (1 - P_miss)
        resid_g = np.column_stack([resid.loc[:, (groups.reindex(X.columns) == g).values].mean(axis=1) for g in used])
        rows = np.arange(len(X))

        in_screen = pat_group.isin(used).values
        p, test_name = presence_group_test(pat_present.values[:, in_screen], pat_group.values[in_screen],
                                           cfg.n_perm, np.random.default_rng([cfg.seed, 1]))
        out["group_high"] = np.array(used)[hi]
        out["group_low"] = np.array(used)[lo]
        out["presence_spread"] = pres_used.values[rows, hi] - pres_used.values[rows, lo]
        out["resid_spread"] = resid_g[rows, hi] - resid_g[rows, lo]
        out["group_p"] = p
        out["group_q"] = multipletests(p, method="fdr_bh")[1]
    else:
        test_name = f"none (fewer than 2 groups with >= {cfg.min_group_size} patients)"
        for c in ["group_high", "group_low", "presence_spread", "resid_spread", "group_p", "group_q"]:
            out[c] = np.nan
    return out, used, test_name


def classify_protein(r, cfg: NAConfig) -> str:
    """Decision rules, applied in order (group screen before the sparsity filter)."""
    if r.n_missing == 0:
        return "Complete"
    if (r.group_p < cfg.group_p and r.presence_spread >= cfg.group_delta
            and r.resid_spread >= cfg.group_resid_delta):
        return "Group-structured"
    if r.max_presence < cfg.presence_threshold:
        return "Sparse (drop)"
    if r.frac_censored >= cfg.mnar_frac:
        return "MNAR - censored"
    if r.frac_censored <= cfg.mcar_frac:
        return "MCAR - random"
    return "Mixed"


def classify_proteins(profile: pd.DataFrame, cfg: NAConfig) -> pd.Series:
    return pd.Series(pd.Categorical(profile.apply(classify_protein, axis=1, cfg=cfg),
                                    categories=CLASS_ORDER, ordered=True), index=profile.index, name="na_class")


def imputation_label(cls: str, cfg: NAConfig) -> str:
    return {"Complete": "none", "MCAR - random": "kNN", "MNAR - censored": "MinProb",
            "Mixed": "per-cell (MinProb if p>=%.2f else kNN)" % cfg.censor_cell_p,
            "Group-structured": "MinProb + binary presence feature", "Sparse (drop)": "dropped"}[cls]


# ------------------------------------------------------------------ imputation

def assign_methods(M: pd.DataFrame, P_miss: pd.DataFrame, classes: pd.Series, cfg: NAConfig) -> pd.DataFrame:
    """Per-cell method ('MinProb' / 'kNN' / '' for observed) for the rows in M."""
    classes = classes.reindex(M.index)
    method = pd.DataFrame("", index=M.index, columns=M.columns)
    minprob_rows = classes.isin(["MNAR - censored", "Group-structured"]).values[:, None]
    knn_rows = (classes == "MCAR - random").values[:, None]
    mixed_rows = (classes == "Mixed").values[:, None]
    P = P_miss.loc[M.index, M.columns]
    method[M & minprob_rows] = "MinProb"
    method[M & knn_rows] = "kNN"
    method[M & mixed_rows & (P >= cfg.censor_cell_p)] = "MinProb"
    method[M & mixed_rows & (P < cfg.censor_cell_p)] = "kNN"
    if not ((method.values[M.values] != "").all() and (method.values[~M.values] == "").all()):
        raise ValueError("every missing cell (and only those) must be assigned a method")
    return method


def knn_impute(X: pd.DataFrame, k: int) -> pd.DataFrame:
    """Protein-wise kNN (neighbours = proteins, as impute::impute.knn); donors are observed values only."""
    filled = KNNImputer(n_neighbors=k, weights="distance").fit_transform(X.values)
    return pd.DataFrame(filled, index=X.index, columns=X.columns)


def minprob_params(X_ref: pd.DataFrame, cfg: NAConfig) -> pd.DataFrame:
    """Per-sample N(mean - shift*sd, (width*sd)^2), from that sample's observed values."""
    mean, sd = X_ref.mean(axis=0), X_ref.std(axis=0)
    return pd.DataFrame({"loc": mean - cfg.minprob_shift * sd, "scale": cfg.minprob_width * sd})


def impute_by_class(X, profile, P_miss, cfg: NAConfig, X_ref=None):
    """Drop Sparse proteins, then fill each missing cell with its assigned method.

    X_ref: matrix whose per-sample distribution sets the MinProb parameters
    (default: X, all proteins, as in the notebook). Returns (X_imputed, method).
    """
    X_ref = X if X_ref is None else X_ref
    keep = profile["na_class"] != "Sparse (drop)"
    Xk = X[keep.values]
    method = assign_methods(Xk.isna(), P_miss, profile.loc[keep, "na_class"], cfg)

    X_imp = Xk.mask(method == "kNN", knn_impute(Xk, cfg.knn_k))

    rng = np.random.default_rng(cfg.seed)
    params = minprob_params(X_ref, cfg)
    for s in Xk.columns:  # fixed column order -> reproducible draws
        cells = method.index[method[s] == "MinProb"]
        X_imp.loc[cells, s] = rng.normal(params.at[s, "loc"], params.at[s, "scale"], size=len(cells))

    if X_imp.isna().any().any():
        raise RuntimeError("imputation left NaNs")
    return X_imp, method


# ------------------------------------------------------------------ workflow

def run_workflow(lfq_path, cfg: NAConfig = None, out_dir=DEFAULT_OUT, groups=None,
                 report=False, sheet=0, verbose=True) -> NAResult:
    """Load -> detection model -> per-protein profile -> classes -> imputation -> (save).

    groups: optional per-sample labels (Series indexed by sample_id, or a path to a CSV for
    load_groups); overrides cfg.group_col.
    report: also write figures, tables and report.html to <out_dir>/report (see na_report).
    """
    cfg = cfg or NAConfig()
    X, prot, meta = load_lfq(lfq_path, sheet=sheet, verbose=verbose)
    if isinstance(groups, (str, Path)):
        groups = load_groups(groups, meta)
    meta["na_group"] = resolve_groups(meta, cfg, groups)
    mu, shift = median_polish(X)
    meta["polish_shift"] = shift
    P_miss, _ = fit_detection_model(X, mu)

    profile, used, test_name = na_profile(X, prot, meta, mu, P_miss, meta["na_group"], cfg)
    profile["na_class"] = classify_proteins(profile, cfg)
    profile["imputation"] = profile["na_class"].astype(str).map(lambda c: imputation_label(c, cfg))

    X_imp, method = impute_by_class(X, profile, P_miss, cfg)
    presence = X.notna().astype(int)
    res = NAResult(X, prot, meta, mu, shift, P_miss, profile, method, X_imp, presence, used, test_name, cfg)

    if verbose:
        M = X.isna()
        print(f"{X.shape[0]} proteins x {X.shape[1]} samples | missing = {M.values.mean():.1%}")
        sizes = meta.drop_duplicates("patient")["na_group"].value_counts().sort_index()
        print("groups (patients): " + ", ".join(f"{g}={n}{'' if g in used else ' [excluded]'}"
                                                for g, n in sizes.items()))
        print(f"group test: {test_name}")
        print(profile["na_class"].value_counts().reindex(CLASS_ORDER).to_string())
        counts = pd.Series(method.values[method.values != ""]).value_counts()
        print(f"retained {len(X_imp)} proteins | imputed cells: "
              + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if out_dir is not None:
        save_outputs(res, out_dir, verbose=verbose)
        if report:
            from na_report import make_report
            path = make_report(res, Path(out_dir) / "report", source=lfq_path)
            if verbose:
                print("report:", path)
    return res


def save_outputs(res: NAResult, out_dir=DEFAULT_OUT, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def with_ids(df):
        return res.prot.loc[df.index, ["accession", "gene"]].join(df)

    files = {
        "protein_na_classification.csv": res.profile.sort_values(["na_class", "group_p"]),
        "LFQ_imputed_by_na_class.csv": with_ids(res.X_imputed),
        "LFQ_imputation_method_per_cell.csv": with_ids(res.method.replace("", "observed")),
        "LFQ_presence_matrix.csv": with_ids(res.presence),
    }
    for name, df in files.items():
        df.to_csv(out_dir / name, index=False)
        if verbose:
            print("saved", out_dir / name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--lfq", type=Path, required=True,
                    help="proteins x samples matrix (.csv, .tsv or Excel)")
    ap.add_argument("--sheet", default="0", help="Excel sheet name or index (default: first sheet)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", action="store_true",
                    help="also write figures, tables and report.html to <out-dir>/report")
    ap.add_argument("--groups-file", type=Path, default=None,
                    help="CSV (sample_id or patient, group); overrides --group-col")
    for f in fields(NAConfig):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    args = ap.parse_args(argv)
    cfg = NAConfig(**{f.name: getattr(args, f.name) for f in fields(NAConfig)})
    sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet
    run_workflow(args.lfq, cfg, args.out_dir, groups=args.groups_file, report=args.report, sheet=sheet)


if __name__ == "__main__":
    main()
