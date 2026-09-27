"""Tests for na_imputation and na_report.  Run:  uv run pytest"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import na_imputation as na
from na_imputation import NAConfig, run_workflow


# ------------------------------------------------------------------ fixtures

def make_matrix(n_prot=150, seed=0, groups=("Control", "Disease"), n_per_group=(5, 7), bilateral=("Disease02",),
                n_onoff=8, shallow=1):
    """Small LFQ-like matrix: censoring below a per-run limit, MCAR dropout, on/off proteins.
    Returns (DataFrame in input format, list of sample IDs)."""
    rng = np.random.default_rng(seed)
    patients = [f"{g}{i + 1:02d}" for g, n in zip(groups, n_per_group) for i in range(n)]
    samples = [f"{p}.{s}" for p in patients for s in (["L", "R"] if p in bilateral else ["R"])]
    grp = np.array([s.rstrip(".LR").rstrip("0123456789") for s in samples])
    mu = rng.normal(22, 2.5, n_prot)
    Y = mu[:, None] + rng.normal(0, 0.4, (n_prot, len(samples)))
    onoff = rng.choice(n_prot, n_onoff, replace=False)
    mu[onoff] = 24
    Y[onoff] = 24 + rng.normal(0, 0.3, (n_onoff, len(samples)))
    Y[np.ix_(onoff, grp == groups[0])] -= 8
    lod = 20 + rng.normal(0, 0.5, len(samples))
    lod[:shallow] += 3
    detected = rng.random(Y.shape) < 1 / (1 + np.exp(-(Y - lod) / 0.5))
    detected &= rng.random(Y.shape) > 0.03
    X = np.where(detected, Y, np.nan)
    X = X - np.nanmedian(X, axis=0, keepdims=True)
    df = pd.DataFrame(X, columns=samples).round(4)
    df.insert(0, "Protein.Names", [f"protein {i}" for i in range(n_prot)])
    df.insert(0, "PG.Genes", [f"G{i:03d}" for i in range(n_prot)])
    df.insert(0, "PG.ProteinAccessions", [f"ACC{i:04d}" for i in range(n_prot)])
    return df, samples


@pytest.fixture
def matrix_csv(tmp_path):
    df, _ = make_matrix()
    path = tmp_path / "m.csv"
    df.to_csv(path, index=False)
    return path


def quiet(path, **kw):
    kw.setdefault("out_dir", None)
    return run_workflow(path, verbose=False, **kw)


# ------------------------------------------------------------------ parsing & loading

def test_parse_sample_id():
    d = na.parse_sample_id("Disease05.W.34.L.ATMG.OA")
    assert (d["patient"], d["group"], d["side"], d["diagnosis"]) == ("Disease05", "Disease", "L", "ATMG+OA")
    d = na.parse_sample_id("S001")
    assert (d["patient"], d["group"], d["side"], d["diagnosis"]) == ("S001", "S", "", "")
    d = na.parse_sample_id(17)                      # numeric column names
    assert d["sample_id"] == "17" and d["group"] == "17"


def test_formats_give_identical_results(tmp_path):
    df, _ = make_matrix()
    df.to_csv(tmp_path / "m.csv", index=False)
    df.to_csv(tmp_path / "m.tsv", sep="\t", index=False)
    with pd.ExcelWriter(tmp_path / "m.xlsx") as w:
        pd.DataFrame({"x": [1]}).to_excel(w, sheet_name="other", index=False)
        df.to_excel(w, sheet_name="data", index=False)
    a = quiet(tmp_path / "m.csv")
    b = quiet(tmp_path / "m.tsv")
    c = quiet(tmp_path / "m.xlsx", sheet="data")
    for r in (b, c):
        assert (a.profile["na_class"] == r.profile["na_class"]).all()
        assert np.allclose(a.X_imputed.values, r.X_imputed.values)


def test_load_handles_messy_columns(tmp_path, capsys):
    df, samples = make_matrix()
    df["Description"] = "text"                     # annotation column: ignored
    df["Empty"] = np.nan                           # empty column: dropped
    df[samples[0]] = df[samples[0]].astype(object)
    df.loc[3, samples[0]] = "W1"                   # non-numeric cell
    df.loc[5, samples] = np.nan                    # protein with no observed value
    path = tmp_path / "m.csv"
    df.to_csv(path, index=False)
    X, prot, meta = na.load_lfq(path)
    out = capsys.readouterr().out
    assert list(X.columns) == samples
    assert "ignored 1 non-numeric column" in out and "dropped 1 empty column" in out
    n_empty = int(df[samples].apply(pd.to_numeric, errors="coerce").isna().all(axis=1).sum())
    assert "'W1'" in out and f"removed {n_empty} protein" in out
    assert 5 not in X.index and np.isnan(X.at[3, samples[0]])
    assert (meta["n_detected"] == X.notna().sum()).all()


def test_optional_id_columns_and_numeric_sample_names(tmp_path):
    df, samples = make_matrix()
    df = df.drop(columns=["PG.Genes", "Protein.Names"]).rename(columns={s: i for i, s in enumerate(samples)})
    path = tmp_path / "m.csv"
    df.to_csv(path, index=False)
    res = quiet(path, cfg=NAConfig(min_group_size=1))
    assert list(res.X.columns) == [str(i) for i in range(len(samples))]
    assert (res.prot["gene"] == res.prot["accession"]).all()


def test_load_errors(tmp_path):
    df, _ = make_matrix()
    df.drop(columns="PG.ProteinAccessions").to_csv(tmp_path / "a.csv", index=False)
    with pytest.raises(ValueError, match="PG.ProteinAccessions"):
        na.load_lfq(tmp_path / "a.csv", verbose=False)
    df.iloc[:, :4].to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError, match="at least 2 sample columns"):
        na.load_lfq(tmp_path / "b.csv", verbose=False)


# ------------------------------------------------------------------ groups & sample sheets

def test_groups_file_by_sample_and_patient(tmp_path, matrix_csv):
    X, prot, meta = na.load_lfq(matrix_csv, verbose=False)
    pd.DataFrame({"patient": meta["patient"].unique(), "g": "A"}).to_csv(tmp_path / "gp.csv", index=False)
    pd.DataFrame({"sample_id": meta.index, "g": "B"}).to_csv(tmp_path / "gs.csv", index=False)
    assert (na.load_groups(tmp_path / "gp.csv", meta) == "A").all()
    assert (na.load_groups(tmp_path / "gs.csv", meta) == "B").all()


def test_groups_file_errors(tmp_path, matrix_csv):
    X, prot, meta = na.load_lfq(matrix_csv, verbose=False)
    pats = list(meta["patient"].unique())
    pd.DataFrame({"patient": pats, "g": ["A"] * (len(pats) - 1) + [None]}).to_csv(tmp_path / "nan.csv", index=False)
    with pytest.raises(ValueError, match="empty id or group"):
        na.load_groups(tmp_path / "nan.csv", meta)
    pd.DataFrame({"patient": pats + pats[:1], "g": ["A"] * len(pats) + ["B"]}).to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(ValueError, match="conflicting groups"):
        na.load_groups(tmp_path / "c.csv", meta)
    pd.DataFrame({"patient": pats[:-1], "g": "A"}).to_csv(tmp_path / "short.csv", index=False)
    with pytest.raises(ValueError, match="every sample_id or every patient"):
        na.load_groups(tmp_path / "short.csv", meta)


def test_group_label_must_be_constant_within_patient(matrix_csv):
    X, prot, meta = na.load_lfq(matrix_csv, verbose=False)
    g = meta["group"].copy()
    g["Disease02.L"] = "Other"
    with pytest.raises(ValueError, match="differs between samples of the same patient"):
        quiet(matrix_csv, groups=g)


def test_sample_sheet_patient_override_equals_parsed(tmp_path):
    """Anonymous IDs + patients from the sheet give the same result as informative IDs."""
    df, samples = make_matrix()
    df.to_csv(tmp_path / "m.csv", index=False)
    ref = quiet(tmp_path / "m.csv")
    anon = {s: f"S{i:02d}" for i, s in enumerate(samples)}
    df.rename(columns=anon).to_csv(tmp_path / "anon.csv", index=False)
    sheet = pd.DataFrame({"sample_id": [anon[s] for s in samples],
                          "patient": [f"P{s.split('.')[0]}" for s in samples],
                          "condition": [s.rstrip(".LR").rstrip("0123456789") for s in samples]})
    res = quiet(tmp_path / "anon.csv", sample_sheet=sheet, cfg=NAConfig(group_col="condition"))
    assert (res.profile["na_class"].values == ref.profile["na_class"].values).all()
    assert np.allclose(res.profile["group_p"], ref.profile["group_p"])
    assert np.allclose(res.X_imputed.values, ref.X_imputed.values)


def test_sample_sheet_validation(tmp_path, matrix_csv, capsys):
    X, prot, meta = na.load_lfq(matrix_csv, verbose=False)
    sheet = pd.DataFrame({"sample_id": meta.index, "subtype": "x", "batch": 1})
    before = sheet.copy()
    quiet(matrix_csv, sample_sheet=sheet)
    pd.testing.assert_frame_equal(sheet, before)                  # caller's DataFrame untouched
    with pytest.raises(ValueError, match="no row in the sample sheet"):
        quiet(matrix_csv, sample_sheet=sheet.iloc[1:])
    with pytest.raises(ValueError, match="duplicated sample_id"):
        quiet(matrix_csv, sample_sheet=pd.concat([sheet, sheet.iloc[:1]]))
    with pytest.raises(ValueError, match="reserved"):
        quiet(matrix_csv, sample_sheet=sheet.assign(n_detected=1))
    with pytest.raises(ValueError, match="group column 'nope' not found"):
        quiet(matrix_csv, sample_sheet=sheet, cfg=NAConfig(group_col="nope"))
    with pytest.raises(ValueError, match="sample_id"):
        quiet(matrix_csv, sample_sheet=sheet.rename(columns={"sample_id": "id"}))
    extra = pd.concat([sheet, pd.DataFrame({"sample_id": ["ghost"], "subtype": ["x"], "batch": [1]})])
    run_workflow(matrix_csv, sample_sheet=extra, out_dir=None, verbose=True)
    assert "ignoring 1 row" in capsys.readouterr().out


def test_sample_sheet_excel_float_ids(tmp_path):
    df, samples = make_matrix()
    ids = {s: i + 1 for i, s in enumerate(samples)}
    df.rename(columns=ids).to_csv(tmp_path / "m.csv", index=False)
    sheet = pd.DataFrame({"sample_id": [float(ids[s]) for s in samples],     # Excel-style 1.0, 2.0
                          "patient": [s.split(".")[0] for s in samples],
                          "group": [s.rstrip(".LR").rstrip("0123456789") for s in samples]})
    sheet.to_excel(tmp_path / "sheet.xlsx", index=False)
    res = quiet(tmp_path / "m.csv", sample_sheet=tmp_path / "sheet.xlsx")
    assert sorted(res.meta["na_group"].unique()) == ["Control", "Disease"]


def test_group_name_collision(tmp_path):
    df, samples = make_matrix()
    df.to_csv(tmp_path / "m.csv", index=False)
    g = pd.Series({s: ("Disease A" if s.startswith("D") else "Disease_A") for s in samples})
    with pytest.raises(ValueError, match="same/ reserved column"):
        quiet(tmp_path / "m.csv", groups=g)


# ------------------------------------------------------------------ statistics

def test_fisher_matches_scipy():
    rng = np.random.default_rng(1)
    present = rng.random((50, 12)) < 0.5
    labels = np.array(["A"] * 5 + ["B"] * 7)
    p, name = na.presence_group_test(present, labels, 0, rng)
    ref = [stats.fisher_exact([[r[:5].sum(), 5 - r[:5].sum()], [r[5:].sum(), 7 - r[5:].sum()]])[1] for r in present]
    assert name.startswith("Fisher") and np.allclose(p, ref)


def test_freeman_halton_matches_scipy_and_no_variation():
    rng = np.random.default_rng(2)
    labels = np.array(["A"] * 5 + ["B"] * 6 + ["C"] * 4)
    present = rng.random((12, 15)) < rng.random((12, 1))
    present[0] = True                                             # no variation -> p = 1
    p, name = na.presence_group_test(present, labels, 20000, np.random.default_rng(3))
    assert name.startswith("Freeman-Halton") and p[0] == 1
    for i in range(1, 12):
        t = np.array([[present[i, labels == g].sum(), (labels == g).sum() - present[i, labels == g].sum()]
                      for g in "ABC"])
        if t[:, 0].sum() in (0, 15):
            continue
        ref = stats.fisher_exact(t, method=stats.MonteCarloMethod(n_resamples=20000, rng=np.random.default_rng(4))).pvalue
        assert abs(p[i] - ref) < 0.02


def test_median_polish_recovers_additive_effects():
    rng = np.random.default_rng(5)
    mu, s = rng.normal(0, 3, 40), rng.normal(0, 1, 8)
    s -= np.median(s)
    X = pd.DataFrame(mu[:, None] + s[None, :])
    mu_hat, s_hat = na.median_polish(X)
    assert np.allclose(s_hat.values, s, atol=1e-8) and np.allclose(mu_hat.values, mu, atol=1e-8)


def test_detection_model_needs_missing_values():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 4)))
    with pytest.raises(ValueError, match="no missing values"):
        na.fit_detection_model(X, X.median(axis=1))


# ------------------------------------------------------------------ workflow invariants

def test_workflow_invariants(matrix_csv):
    res = quiet(matrix_csv)
    X, M, prof = res.X, res.X.isna(), res.profile
    kept = prof.index[prof["na_class"] != "Sparse (drop)"]
    assert list(res.X_imputed.index) == list(kept) and not res.X_imputed.isna().any().any()
    obs = ~M.loc[kept]
    assert np.allclose(res.X_imputed.values[obs.values], X.loc[kept].values[obs.values])  # observed untouched
    assert ((res.method != "").values == M.loc[kept].values).all()                        # a method per NA
    # classes follow the documented rules
    assert (prof.loc[prof["n_missing"] == 0, "na_class"] == "Complete").all()
    assert (prof.loc[prof["na_class"] == "MNAR - censored", "frac_censored"] >= 0.8).all()
    assert (prof.loc[prof["na_class"] == "MCAR - random", "frac_censored"] <= 0.2).all()
    gs = prof[prof["na_class"] == "Group-structured"]
    assert ((gs["group_p"] < 0.05) & (gs["presence_spread"] >= 0.5) & (gs["resid_spread"] >= 0.25)).all()
    # per-class methods
    cls = prof.loc[kept, "na_class"]
    minprob_only = cls.isin(["MNAR - censored", "Group-structured"]).values
    assert not (res.method.values[minprob_only] == "kNN").any()
    assert not (res.method.values[(cls == "MCAR - random").values] == "MinProb").any()
    # the planted on/off proteins are found
    assert len(gs) >= 4


def test_workflow_is_reproducible_and_seeded(matrix_csv):
    a, b = quiet(matrix_csv), quiet(matrix_csv)
    c = quiet(matrix_csv, cfg=NAConfig(seed=1))
    assert a.X_imputed.equals(b.X_imputed)
    mp = (a.method == "MinProb").values
    assert not np.allclose(a.X_imputed.values[mp], c.X_imputed.values[mp])


def test_three_groups_and_min_group_size(tmp_path):
    df, _ = make_matrix(groups=("A", "B", "C"), n_per_group=(4, 4, 2), bilateral=())
    df.to_csv(tmp_path / "m.csv", index=False)
    r3 = quiet(tmp_path / "m.csv", cfg=NAConfig(min_group_size=2, n_perm=500))
    assert r3.groups_used == ["A", "B", "C"] and r3.group_test.startswith("Freeman-Halton")
    r2 = quiet(tmp_path / "m.csv", cfg=NAConfig(min_group_size=3))
    assert r2.groups_used == ["A", "B"] and r2.group_test.startswith("Fisher")
    assert "presence_c" in r2.profile                              # excluded group still reported
    r0 = quiet(tmp_path / "m.csv", cfg=NAConfig(min_group_size=10))
    assert r0.groups_used == [] and r0.group_test.startswith("none")
    assert not (r0.profile["na_class"] == "Group-structured").any()


def test_edge_cases(tmp_path, matrix_csv):
    # everything sparse -> empty retained matrix, still a valid result and report
    # (complete proteins are always kept, so give every protein one NA first)
    df, samples = make_matrix()
    for i in df.index:
        df.loc[i, samples[i % len(samples)]] = np.nan
    df.to_csv(tmp_path / "nocomplete.csv", index=False)
    res = run_workflow(tmp_path / "nocomplete.csv", cfg=NAConfig(presence_threshold=1.01, group_p=0),
                       out_dir=tmp_path / "o", report=True, verbose=False)
    assert res.X_imputed.empty and (tmp_path / "o" / "report" / "report.html").exists()
    # a sample with a single observed value (no sd for MinProb)
    df, samples = make_matrix()
    df.loc[1:, samples[-1]] = np.nan
    df.to_csv(tmp_path / "one.csv", index=False)
    r = quiet(tmp_path / "one.csv")
    assert not r.X_imputed.isna().any().any()
    # a complete matrix is rejected with a clear message
    full = make_matrix()[0]
    full[samples] = full[samples].fillna(0.0)
    full.to_csv(tmp_path / "full.csv", index=False)
    with pytest.raises(ValueError, match="no missing values"):
        quiet(tmp_path / "full.csv")
    with pytest.raises(ValueError, match="needs an out_dir"):
        run_workflow(matrix_csv, report=True, out_dir=None, verbose=False)


# ------------------------------------------------------------------ CLI & report

def test_cli_writes_outputs(tmp_path, matrix_csv):
    out = tmp_path / "cli"
    na.main(["--lfq", str(matrix_csv), "--out-dir", str(out), "--report", "--censor-cell-p", "0.4",
             "--knn-k", "5"])
    for f in ["protein_na_classification.csv", "LFQ_imputed_by_na_class.csv",
              "LFQ_imputation_method_per_cell.csv", "LFQ_presence_matrix.csv", "report/report.html"]:
        assert (out / f).exists(), f
    prof = pd.read_csv(out / "protein_na_classification.csv")
    assert prof["imputation"].str.contains("0.40").any() or not (prof["na_class"] == "Mixed").any()


def test_report_contents(tmp_path):
    from na_report import make_report, report_figures, report_tables
    df, samples = make_matrix(groups=("Ctl", "<b>Dis</b>"), n_per_group=(5, 7), bilateral=())
    df.to_csv(tmp_path / "m.csv", index=False)
    res = quiet(tmp_path / "m.csv", groups=pd.Series({s: ("Ctl" if s.startswith("Ctl") else "<b>Dis</b>")
                                                     for s in samples}))
    t = report_tables(res)
    assert set(t) == {"class_summary", "imputation_counts", "sample_qc", "detection_model", "calibration",
                      "group_structured"}
    assert t["class_summary"]["n_proteins"].sum() == len(res.profile)
    assert t["imputation_counts"].loc["All retained", "total"] == (res.method != "").values.sum()
    figs = report_figures(res)
    assert len(figs) == 8
    gene = res.profile["gene"].iloc[0]
    path = make_report(res, tmp_path / "rep", examples={"Complete": gene})
    text = path.read_text(encoding="utf-8")
    assert "<b>Dis</b>" not in text.split("<main>")[1].split("<h2")[0]     # group names escaped in summary
    assert text.count("<img") == 8
    with pytest.raises(ValueError, match="no protein with gene or accession"):
        make_report(res, tmp_path / "rep2", examples={"Mixed": "NOT_A_GENE"})


def test_report_many_groups_and_no_side(tmp_path):
    from na_report import make_report
    groups = tuple(f"G{i}" for i in range(10))                     # > 8 groups: colours fold to grey
    df, samples = make_matrix(groups=groups, n_per_group=(2,) * 10, bilateral=())
    anon = {s: f"S{i}" for i, s in enumerate(samples)}
    df.rename(columns=anon).to_csv(tmp_path / "m.csv", index=False)
    sheet = pd.DataFrame({"sample_id": [anon[s] for s in samples], "patient": [s.split(".")[0] for s in samples],
                          "group": [s.split(".")[0].rstrip("0123456789") for s in samples]})
    res = quiet(tmp_path / "m.csv", sample_sheet=sheet, cfg=NAConfig(min_group_size=2, n_perm=200))
    path = make_report(res, tmp_path / "rep")
    assert path.exists() and "side" not in pd.read_csv(tmp_path / "rep" / "tables" / "sample_qc.csv").columns
