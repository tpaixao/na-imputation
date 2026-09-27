"""
Report tables, figures and a self-contained HTML report for an NAResult (see na_imputation).

    from na_imputation import run_workflow
    from na_report import make_report, report_tables, report_figures
    res = run_workflow("matrix.csv", out_dir=None)
    make_report(res, "results/report")          # tables/*.csv, figures/*.png, report.html
    report_tables(res)["sample_qc"]             # or build individual pieces
    report_figures(res)["classes"]              # matplotlib Figure objects (display in Jupyter)

Or from the command line: python na_imputation.py --lfq ... --report

Figures are built with the object-oriented matplotlib API (no pyplot), so this works headless
and does not change the notebook backend.
"""
import base64
import html
import io
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

from na_imputation import CLASS_ORDER, NAResult, _col, fit_detection_model, minprob_params

# categorical slots (fixed order); a 9th+ group folds to gray
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY, INK, INK2, GRID, SURFACE = "#9a9994", "#0b0b0b", "#52514e", "#e6e5e1", "#f0efec"
CLASS_COLOR = dict(zip(CLASS_ORDER, SLOTS[:5] + [GRAY]))
NA_CENSORED, NA_RANDOM = INK2, "#e34948"

STYLE = {"figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
         "font.size": 9, "axes.titlelocation": "left", "axes.titlesize": 10, "axes.edgecolor": INK2,
         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK}


# ------------------------------------------------------------------ helpers

def _group_colors(res: NAResult) -> dict:
    groups = sorted(res.meta["na_group"].unique())
    return {g: (SLOTS[i] if i < len(SLOTS) else GRAY) for i, g in enumerate(groups)}


def _short(res: NAResult, s: str) -> str:
    side = res.meta.at[s, "side"]
    return f"{res.meta.at[s, 'patient']}.{side}" if isinstance(side, str) and side in ("L", "R") else str(s)


def _samples_by_depth(res: NAResult, deep_first=True):
    return res.meta.sort_values("n_detected", ascending=not deep_first).index


def _pick_example(res: NAResult, cls: str):
    p = res.profile
    d = p[p["na_class"] == cls]
    if d.empty:
        return None
    if cls == "Group-structured":
        return d["group_p"].idxmin()
    if cls == "Complete":
        return d["mu_hat"].idxmin()
    mid = d["n_missing"].between(3, 15)
    return d.loc[d.index[mid] if mid.any() else d.index, "n_missing"].idxmax()


def _legend(ax, handles, **kw):
    ax.legend(handles=handles, frameon=False, fontsize=8, **kw)


# ------------------------------------------------------------------ tables

def report_tables(res: NAResult) -> dict:
    """Dict of DataFrames: class_summary, imputation_counts, sample_qc, detection_model,
    calibration, group_structured."""
    X, M, meta, prof, cfg = res.X, res.X.isna(), res.meta, res.profile, res.cfg
    out = {}

    cs = (prof.groupby("na_class", observed=False)
          .agg(n_proteins=("gene", "size"), median_frac_missing=("frac_missing", "median"),
               median_abundance=("mu_hat", "median"), median_frac_censored=("frac_censored", "median"),
               median_depth_auc=("depth_auc", "median")))
    cs.insert(1, "share", cs["n_proteins"] / len(prof))
    cs["imputation"] = [prof.loc[prof["na_class"] == c, "imputation"].iloc[0] if n else ""
                        for c, n in cs["n_proteins"].items()]
    out["class_summary"] = cs

    cls = prof.loc[res.method.index, "na_class"]
    counts = pd.DataFrame({m: (res.method == m).sum(axis=1).groupby(cls, observed=False).sum()
                           for m in ["MinProb", "kNN"]})
    counts["total"] = counts.sum(axis=1)
    counts.index = counts.index.astype(str)
    counts.loc["All retained"] = counts.sum()
    out["imputation_counts"] = counts

    complete = ~M.any(axis=1)
    mp = minprob_params(X, cfg)
    base = ["patient", "side", "na_group"]
    if not meta["side"].astype(str).isin(["L", "R"]).any():
        base.remove("side")
    # sample-sheet (or other extra) columns, but not the ID-parsed defaults
    extra = [c for c in meta.columns if c not in {"patient", "side", "na_group", "group", "diagnosis",
                                                    "n_detected", "polish_shift"}]
    qc = meta[base + extra + ["n_detected", "polish_shift"]].copy()
    qc["median_complete_proteins"] = X[complete].median(axis=0) if complete.any() else np.nan
    qc["minprob_mean"], qc["minprob_sd"] = mp["loc"], mp["scale"]
    qc["n_MinProb"] = (res.method == "MinProb").sum()
    qc["n_kNN"] = (res.method == "kNN").sum()
    qc["frac_imputed_retained"] = (qc["n_MinProb"] + qc["n_kNN"]) / len(res.method) if len(res.method) else np.nan
    out["sample_qc"] = qc.sort_values("n_detected")

    y, p = M.values.ravel(), res.P_miss.values.ravel()
    model = res.det_model if res.det_model is not None else fit_detection_model(X, res.mu)[1]
    b = -model.coef_[0][0]
    out["detection_model"] = pd.DataFrame({
        "cells": [y.size], "frac_missing": [y.mean()], "abundance_slope_b": [b],
        "odds_ratio_per_log2_unit": [np.exp(b)], "auc": [roc_auc_score(y, p)]})
    bins = pd.qcut(p, 20, duplicates="drop")
    out["calibration"] = (pd.DataFrame({"predicted": p, "observed": y}).groupby(bins, observed=True)
                          .agg(predicted=("predicted", "mean"), observed=("observed", "mean"),
                               cells=("observed", "size")).reset_index(drop=True))

    pres_cols = [c for c in prof.columns if c.startswith("presence_") and c != "presence_spread"]
    gs = prof[prof["na_class"] == "Group-structured"].sort_values("group_p")
    out["group_structured"] = gs[["accession", "gene", "protein_name", "n_missing", *pres_cols,
                                  "group_high", "group_low", "presence_spread", "resid_spread",
                                  "group_p", "group_q", "mu_hat"]]
    return out


# ------------------------------------------------------------------ figures

def fig_sample_depth(res: NAResult) -> Figure:
    fig = Figure(figsize=(9, 3.4), layout="constrained")
    ax = fig.subplots()
    order = _samples_by_depth(res, deep_first=False)
    gcol = _group_colors(res)
    ax.bar(range(len(order)), res.meta.loc[order, "n_detected"], width=0.8,
           color=[gcol[g] for g in res.meta.loc[order, "na_group"]])
    ax.set_xticks(range(len(order)), [_short(res, s) for s in order], rotation=70)
    ax.set_ylabel("proteins detected")
    ax.set_title("Run depth per sample")
    ax.grid(axis="x", visible=False)
    _legend(ax, [Line2D([], [], ls="", marker="s", ms=8, color=c, label=g) for g, c in gcol.items()],
            loc="upper left")
    return fig


def fig_missing_vs_abundance(res: NAResult) -> Figure:
    fig = Figure(figsize=(5.5, 3.8), layout="constrained")
    ax = fig.subplots()
    m, frac = res.X.mean(axis=1), res.X.isna().mean(axis=1)
    ax.scatter(m, frac, s=8, alpha=0.45, color=SLOTS[0], edgecolor="none")
    rho = pd.Series(m).corr(pd.Series(frac), method="spearman")
    ax.set_xlabel("mean log2 intensity (observed values)")
    ax.set_ylabel("fraction missing")
    ax.set_title(f"Missingness vs abundance (Spearman ρ = {rho:.2f})")
    return fig


def fig_detection_model(res: NAResult, tables=None) -> Figure:
    t = tables or report_tables(res)
    cal, dm = t["calibration"], t["detection_model"].iloc[0]
    fig = Figure(figsize=(10, 3.6), layout="constrained")
    ax = fig.subplots(1, 2)
    ax[0].plot([0, 1], [0, 1], color=GRID, lw=1.5)
    ax[0].plot(cal["predicted"], cal["observed"], "o-", color=SLOTS[0], lw=2, ms=5)
    ax[0].set_xlabel("predicted P(missing)"); ax[0].set_ylabel("observed fraction missing")
    ax[0].set_title(f"Calibration (AUC = {dm.auc:.2f})")

    depth = res.meta["n_detected"].sort_values()
    picks = [(depth.index[-1], "deepest"), (depth.index[len(depth) // 2], "median"), (depth.index[0], "shallowest")]
    # the model is linear in mu on the logit scale: recover (slope, intercept) per sample, draw on a grid
    mu = res.mu.values
    grid = np.linspace(np.nanmin(mu), np.nanmax(mu), 300)
    for (s, lab), col in zip(picks, SLOTS[:3]):
        p = np.clip(res.P_miss[s].values, 1e-12, 1 - 1e-12)
        slope, icpt = np.polyfit(mu, np.log(p / (1 - p)), 1)
        ax[1].plot(grid, 1 / (1 + np.exp(-(icpt + slope * grid))), lw=2, color=col,
                   label=f"{lab} run ({_short(res, s)}, n={res.meta.at[s, 'n_detected']})")
    ax[1].set_xlabel("protein abundance μ (log2)"); ax[1].set_ylabel("P(missing)")
    ax[1].set_title(f"Fitted detection curves (+1 log2 → ×{dm.odds_ratio_per_log2_unit:.2f} odds of detection)")
    ax[1].legend(frameon=False, fontsize=8)
    return fig


def fig_classes(res: NAResult) -> Figure:
    prof = res.profile
    counts = prof["na_class"].value_counts().reindex(CLASS_ORDER)
    fig = Figure(figsize=(11, 3.9), layout="constrained")
    ax = fig.subplots(1, 2, width_ratios=[1, 1.4])
    ys = np.arange(len(counts))[::-1]
    ax[0].barh(ys, counts.values, color=[CLASS_COLOR[c] for c in counts.index], height=0.7)
    ax[0].set_yticks(ys, counts.index)
    for y, v in zip(ys, counts.values):
        ax[0].text(v + counts.max() * 0.01, y, str(v), va="center", color=INK2)
    ax[0].set_xlabel("proteins"); ax[0].set_title("Proteins per NA class"); ax[0].grid(axis="y", visible=False)
    for c in CLASS_ORDER:
        d = prof[prof["na_class"] == c]
        ax[1].scatter(d["mu_hat"], d["frac_missing"], s=10, alpha=0.7, color=CLASS_COLOR[c], label=c, edgecolor="none")
    ax[1].set_xlabel("protein abundance μ (log2, median polish)"); ax[1].set_ylabel("fraction missing")
    ax[1].set_title("Classes in abundance × missingness space")
    ax[1].legend(frameon=False, fontsize=8, markerscale=1.6, loc="upper right")
    return fig


def fig_missingness_map(res: NAResult) -> Figure:
    prof, M = res.profile, res.X.isna()
    ordered = prof.sort_values(["na_class", "mu_hat"], ascending=[True, False]).index
    s_order = _samples_by_depth(res)
    fig = Figure(figsize=(9, 8), layout="constrained")
    axc, axm = fig.subplots(1, 2, width_ratios=[0.04, 1])
    axc.imshow(prof.loc[ordered, "na_class"].cat.codes.values[:, None], aspect="auto",
               cmap=ListedColormap([CLASS_COLOR[c] for c in CLASS_ORDER]),
               vmin=-0.5, vmax=len(CLASS_ORDER) - 0.5, interpolation="nearest")
    axc.set_xticks([]); axc.set_yticks([]); axc.grid(False)
    axm.imshow(M.loc[ordered, s_order].values, aspect="auto", cmap=ListedColormap([SURFACE, "#256abf"]),
               interpolation="nearest")
    axm.set_yticks([]); axm.grid(False)
    axm.set_xticks(range(len(s_order)), [_short(res, s) for s in s_order], rotation=70)
    bounds = np.cumsum(prof["na_class"].value_counts().reindex(CLASS_ORDER).values)
    for b, c, start in zip(bounds, CLASS_ORDER, np.r_[0, bounds[:-1]]):
        if b > start:
            axm.axhline(b - 0.5, color="white", lw=1.5)
            axm.text(len(s_order) - 0.3, (start + b) / 2, f" {c}", va="center", fontsize=8, color=INK2)
    axm.set_title("Missing cells (blue) by protein class — samples ordered deep → shallow")
    return fig


def fig_class_examples(res: NAResult, examples=None) -> Figure:
    """One protein per class (or `examples`: dict class -> gene), observed values + NA rail."""
    X, M, P, prof, cfg = res.X, res.X.isna(), res.P_miss, res.profile, res.cfg
    gcol = _group_colors(res)
    s_order = _samples_by_depth(res)
    xs = np.arange(len(s_order))
    fig = Figure(figsize=(12, 6.4), layout="constrained")
    axes = fig.subplots(2, 3, sharex=True)
    for ax, cls in zip(axes.ravel(), CLASS_ORDER):
        if examples and cls in examples:
            hits = prof.index[(prof["gene"] == examples[cls]) | (prof["accession"] == examples[cls])]
            if not len(hits):
                raise ValueError(f"example for {cls!r}: no protein with gene or accession {examples[cls]!r}")
            i = hits[0]
        else:
            i = _pick_example(res, cls)
        if i is None:
            ax.set_title(f"{cls}: (none)", fontsize=8.5); ax.set_yticks([])
            continue
        v, m, p = X.loc[i, s_order], M.loc[i, s_order].values, P.loc[i, s_order].values
        grp = res.meta.loc[s_order, "na_group"].values
        ax.scatter(xs[~m], v.values[~m], s=24, zorder=3, color=[gcol[g] for g in grp[~m]])
        floor = np.nanmin(v.values) - 1.0
        ax.axhline(floor, color=GRID, lw=4, zorder=1)
        cens = m & (p >= cfg.censor_cell_p)
        rand = m & (p < cfg.censor_cell_p)
        ax.scatter(xs[cens], np.full(cens.sum(), floor), marker="x", color=NA_CENSORED, s=34, lw=1.8, zorder=3)
        ax.scatter(xs[rand], np.full(rand.sum(), floor), marker="x", color=NA_RANDOM, s=34, lw=1.8, zorder=3)
        r = prof.loc[i]
        fc = "–" if pd.isna(r.frac_censored) else f"{r.frac_censored:.2f}"
        ax.set_title(f"{cls}: {r.gene}\nNA={r.n_missing}, censored share={fc}", fontsize=8.5)
    for ax in axes[-1]:
        ax.set_xticks(xs, [_short(res, s) for s in s_order], rotation=80, fontsize=7)
    handles = [Line2D([], [], ls="", marker="o", color=c, label=f"observed ({g})") for g, c in gcol.items()]
    handles += [Line2D([], [], ls="", marker="x", color=NA_CENSORED, mew=1.8, label="NA, censored-like"),
                Line2D([], [], ls="", marker="x", color=NA_RANDOM, mew=1.8, label="NA, random-like")]
    fig.legend(handles=handles, loc="outside upper center", ncol=min(len(handles), 6), frameon=False, fontsize=8)
    fig.supylabel("log2 intensity", color=INK2)
    return fig


def fig_imputed_distributions(res: NAResult) -> Figure:
    s_order = _samples_by_depth(res)
    n = len(s_order)
    ncol = 7
    nrow = int(np.ceil(n / ncol))
    fig = Figure(figsize=(14, 2 * nrow + 0.4), layout="constrained")
    axes = np.atleast_2d(fig.subplots(nrow, ncol, sharex=True, sharey=True))
    Xk, meth, Xi = res.X.loc[res.method.index], res.method, res.X_imputed
    if Xi.empty:
        axes.ravel()[0].text(0.5, 0.5, "No retained proteins", ha="center", va="center", color=INK2)
        for ax in axes.ravel():
            ax.set_axis_off()
        return fig
    lo, hi = np.nanmin(Xi.values), np.nanmax(Xi.values)
    bins = np.linspace(lo, hi if hi > lo else lo + 1, 40)
    for ax, s in zip(axes.ravel(), s_order):
        ax.hist(Xk[s].dropna(), bins=bins, color="#c3c2b7")
        ax.hist(Xi.loc[meth[s] == "kNN", s], bins=bins, color=SLOTS[1], alpha=0.85)
        ax.hist(Xi.loc[meth[s] == "MinProb", s], bins=bins, color=SLOTS[2], alpha=0.85)
        ax.set_title(f"{_short(res, s)} (n={res.meta.at[s, 'n_detected']})", fontsize=8)
        ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.legend(handles=[Line2D([], [], ls="", marker="s", ms=8, color=c, label=l) for c, l in
                        [("#c3c2b7", "observed"), (SLOTS[1], "kNN"), (SLOTS[2], "MinProb")]],
               loc="outside upper right", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Observed vs imputed values per sample (retained proteins)", x=0.01, ha="left", fontsize=10)
    return fig


def fig_group_structured(res: NAResult, max_rows=40) -> Figure:
    """Dot plot: patient-level presence per group for each Group-structured protein."""
    gs = res.profile[res.profile["na_class"] == "Group-structured"].sort_values("group_p").head(max_rows)
    gcol = _group_colors(res)
    fig = Figure(figsize=(7, 0.42 * max(len(gs), 2) + 1.9), layout="constrained")
    ax = fig.subplots()
    if gs.empty:
        ax.text(0.5, 0.5, "No Group-structured proteins", ha="center", va="center", color=INK2)
        ax.set_axis_off()
        return fig
    ys = np.arange(len(gs))[::-1]
    # small vertical offset per group (first group on top, as in the legend) keeps tied dots visible
    offsets = np.linspace(0.22, -0.22, len(gcol)) if len(gcol) > 1 else [0.0]
    for (g, c), dy in zip(gcol.items(), offsets):
        col = _col(g)
        if col in gs:
            used = g in res.groups_used
            ax.scatter(gs[col].values, ys + dy, s=40, color=c if used else "white", edgecolor=c, lw=1.5,
                       zorder=4 if used else 3, label=g + ("" if used else " (excluded)"))
    ax.hlines(ys, 0, 1, color=GRID, lw=0.8, zorder=1)
    ax.set_yticks(ys, [f"{g}  (p={p:.3f})" for g, p in zip(gs["gene"], gs["group_p"])])
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.6, len(gs) - 0.4)
    ax.set_xlabel("fraction of patients with the protein detected")
    ax.grid(axis="y", visible=False)
    ax.set_title("Group-structured proteins — presence per group\n" f"{res.group_test}, raw p")
    fig.legend(frameon=False, fontsize=8, loc="outside lower center", ncol=min(len(gcol), 4))
    return fig


def report_figures(res: NAResult, examples=None, tables=None) -> dict:
    """Dict of matplotlib Figures, in report order."""
    with matplotlib.rc_context(STYLE):
        t = tables or report_tables(res)
        return {
            "sample_depth": fig_sample_depth(res),
            "missing_vs_abundance": fig_missing_vs_abundance(res),
            "detection_model": fig_detection_model(res, t),
            "classes": fig_classes(res),
            "missingness_map": fig_missingness_map(res),
            "class_examples": fig_class_examples(res, examples),
            "imputed_distributions": fig_imputed_distributions(res),
            "group_structured": fig_group_structured(res),
        }


# ------------------------------------------------------------------ HTML

CSS = """
:root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --rule:#e6e5e1; --panel:#f5f4f1; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --ink:#f4f3ef; --ink2:#c3c2b7; --rule:#383835; --panel:#232322; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 19px; margin: 40px 0 8px; padding-top: 8px; border-top: 1px solid var(--rule); }
.sub { color: var(--ink2); margin: 0 0 20px; }
figure { margin: 12px 0 20px; } figure img { max-width: 100%; height: auto; background: #fff; border-radius: 6px; }
figcaption, .note { color: var(--ink2); font-size: 13px; }
.tbl { overflow-x: auto; margin: 8px 0 16px; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 10px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; } td:first-child, th:first-child { text-align: left; }
.kv { display:grid; grid-template-columns: max-content 1fr; gap: 2px 16px; background: var(--panel); padding: 12px 16px; border-radius: 8px; font-size: 13px; }
.kv b { color: var(--ink2); font-weight: 600; }
nav a { color: inherit; margin-right: 14px; font-size: 13px; }
"""

SECTIONS = [
    ("overview", "Overview", ["class_summary", "imputation_counts"], ["classes"],
     "Each protein is labelled by the mechanism most likely behind its NAs; the label decides how it is imputed."),
    ("samples", "Samples and run depth", ["sample_qc"], ["sample_depth"],
     "polish_shift and median_complete_proteins expose the normalisation offset of shallow runs; "
     "frac_imputed_retained is the share of retained-protein cells in that sample that are imputed."),
    ("model", "Detection model (censoring hypothesis)", ["detection_model", "calibration"],
     ["missing_vs_abundance", "detection_model"],
     "logit P(missing_ij) = a_j − b·μ_i. A well-calibrated curve supports using P(missing) to call NAs censored-like (≥ cut-off) or random-like."),
    ("classes", "NA classes", [], ["missingness_map", "class_examples"],
     "Example panels: crosses on the grey rail are missing cells, coloured by the censoring model."),
    ("groups", "Group-structured proteins", ["group_structured"], ["group_structured"],
     "Raw-p screen (not FDR-controlled). Hollow markers: groups below min_group_size, not used in the test."),
    ("imputation", "Imputation", [], ["imputed_distributions"],
     "kNN fills should sit inside the observed distribution; MinProb fills in the left tail."),
]


def _img(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return base64.b64encode(buf.getvalue()).decode()


def _table_html(df: pd.DataFrame) -> str:
    fmt = df.copy()
    for c in fmt.columns:
        if pd.api.types.is_float_dtype(fmt[c]):
            fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else (f"{v:.3g}" if abs(v) < 1e-2 and v != 0 else f"{v:.3f}".rstrip("0").rstrip(".")))
    return '<div class="tbl">' + fmt.to_html(border=0, escape=True, na_rep="") + "</div>"


def make_report(res: NAResult, out_dir, examples=None, source=None, fig_format="png", sample_sheet=None) -> Path:
    """Write tables/*.csv, figures/*.<fig_format> and a self-contained report.html. Returns the HTML path."""
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    tables = report_tables(res)
    figs = report_figures(res, examples, tables)
    for name, df in tables.items():
        df.to_csv(out_dir / "tables" / f"{name}.csv")
    with matplotlib.rc_context(STYLE):
        for name, fig in figs.items():
            fig.savefig(out_dir / "figures" / f"{name}.{fig_format}", facecolor="white")

    cfg = res.cfg
    X = res.X
    sizes = res.meta.drop_duplicates("patient")["na_group"].value_counts().sort_index()
    kv = {
        "Input": html.escape(str(source)) if source else "–",
        "Sample sheet": html.escape(str(sample_sheet)) if sample_sheet else "– (metadata parsed from sample IDs)",
        "Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Matrix": f"{X.shape[0]} proteins × {X.shape[1]} samples, {X.isna().values.mean():.1%} missing",
        "Groups (patients)": ", ".join(f"{html.escape(g)} = {n}" + ("" if g in res.groups_used else " (excluded)")
                                       for g, n in sizes.items()),
        "Group test": html.escape(res.group_test),
        "Retained": f"{len(res.X_imputed)} proteins; {int((res.method != '').values.sum())} cells imputed",
        "Settings": html.escape(", ".join(f"{k}={v}" for k, v in asdict(cfg).items())),
    }
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>NA profile report</title><style>{CSS}</style></head><body><main>",
             "<h1>NA profile report</h1>",
             "<p class='sub'>Missing-value classification and class-specific imputation</p>",
             "<nav>" + "".join(f"<a href='#{sid}'>{title}</a>" for sid, title, *_ in SECTIONS) + "</nav>",
             "<div class='kv' style='margin-top:16px'>" + "".join(f"<b>{k}</b><span>{v}</span>" for k, v in kv.items()) + "</div>"]
    for sid, title, tnames, fnames, note in SECTIONS:
        parts.append(f"<h2 id='{sid}'>{title}</h2><p class='note'>{html.escape(note)}</p>")
        for f in fnames:
            parts.append(f"<figure><img alt='{html.escape(f)}' src='data:image/png;base64,{_img(figs[f])}'></figure>")
        for t in tnames:
            parts.append(f"<h3 style='font-size:14px;margin:16px 0 4px'>{t.replace('_', ' ')}</h3>{_table_html(tables[t])}")
    parts.append("</main></body></html>")
    path = out_dir / "report.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
