"""Which genera differ between cases and controls, and which of those
differences survive being computed two different ways.

The wider profiling step already showed that a sample's originating study is
the single largest axis of between-sample variation -- far bigger than
case/control status -- and that a linear model recovers the study from
composition at ~0.76 accuracy against a 0.04 baseline. So any pooled
case/control signal has to be read against how easily a model can latch
onto the lab instead of the biology. Everything here is built so that can
be seen.

Five pieces, all driven from this module:

**Within-project differential abundance.** Per project, per disease
category, test every prevalence-filtered taxon for a case/control CLR
difference. The primary estimate is an OLS contrast (`clr ~ case + age +
sex`, covariates included per project only where coverage allows), which
yields the effect size + SE the meta-analysis needs; a Wilcoxon rank-sum +
rank-biserial correlation on the raw CLR is carried alongside as a
non-parametric cross-check. Benjamini-Hochberg FDR within each project.

**Random-effects meta-analysis.** Combine the per-project effect sizes with
inverse-variance weighting and a DerSimonian-Laird between-study variance
component. This never pools raw samples across labs, so batch effects
cannot enter the pooled estimate. Reports pooled effect, 95% CI, I^2
heterogeneity and the per-taxon project count.

**Naive pooled + batch correction.** Pool the same samples across projects
and test, running two correction strategies: parametric ComBat on the CLR
matrix (`combat_adjust`) followed by OLS, and a project random-intercept
mixed model. Included as the contrast case for the concordance step.

**Concordance.** Compare the ranked taxon lists from the meta-analysis and
the pooled test: overlap at a fixed FDR, effect-size rank correlation, sign
agreement, and a scatter of the two effect sizes. Every taxon is classified
*replicated* (both), *pooling_only* (pooled test only -- likely a batch
artifact) or *within_only* (meta only -- likely real but under-powered).

**Batch-leakage quantification.** Train a disease classifier (L2-regularised
logistic regression and gradient boosting) under two CV schemes --
leave-one-project-out and random stratified k-fold -- and report both AUCs.
The gap is a direct measurement of how much apparent predictive performance
is study recognition rather than disease signal.

Plain functions, dataframes / paths in and results out, same shape as the
rest of `src/`. CLR rows are pulled from Parquet a batch at a time via
`src.variance`; nothing here loads a full wide matrix densely.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

SAMPLE_COL = "sample"          # "{project}_{srr}", matching the taxa Parquet / npz files
CASE, CONTROL = 1, 0

# categories that are not a single disease -- never run as their own contrast
NON_DISEASE_CATEGORIES = ("other", "healthy_only")


# ---------------------------------------------------------------------------
# Cohort -> per-category contrast frame
# ---------------------------------------------------------------------------

def sample_keys(df):
    """Build the "{project}_{srr}" join key the taxa Parquet / npz files use."""
    return df["project"].astype(str) + "_" + df["srr"].astype(str)


def build_analysis_frame(cohort_df, category_groups=None):
    """One row per (sample, disease_category) contrast membership.

    `disease_category` in the harmonised data is only ever set on *case*
    rows (`src/harmonize.py` assigns it when `disease_label == "case"`), so
    a project's healthy samples carry no category. Each control is therefore
    expanded to one row per case category present in its own project -- a
    control in a project that studied both UC and CD becomes a control row
    for each. Within any single category's analysis a sample still appears
    exactly once.

    `category_groups` optionally remaps raw categories onto coarser ones
    (e.g. the IBD subtypes -> "IBD", since CD / UC / unspecified-IBD are one
    disease family).

    Returns columns: sample, project, srr, disease_category, y (1 case /
    0 control), and whatever of age_years / sex / the technical covariates /
    depth the cohort carries.
    """
    category_groups = category_groups or {}
    df = cohort_df.copy()
    df[SAMPLE_COL] = sample_keys(df)

    remap = lambda c: category_groups.get(c, c)

    cases = df[df["disease_label"] == "case"].copy()
    cases["disease_category"] = cases["disease_category"].map(remap)
    case_rows = cases.assign(y=CASE)

    proj_cats = (case_rows.groupby("project")["disease_category"]
                 .agg(lambda s: sorted(set(s.dropna()))))

    ctrl = df[df["disease_label"] == "healthy"].copy()
    ctrl_parts = []
    for proj, grp in ctrl.groupby("project"):
        for cat in proj_cats.get(proj, []):
            ctrl_parts.append(grp.assign(disease_category=cat, y=CONTROL))
    ctrl_rows = (pd.concat(ctrl_parts, ignore_index=True) if ctrl_parts
                 else ctrl.assign(disease_category=pd.NA, y=CONTROL).iloc[:0])

    keep = [SAMPLE_COL, "project", "srr", "disease_category", "y", "age_years",
            "sex", "amplicon", "kit_family", "instrument", "region", "depth"]
    out = pd.concat([case_rows, ctrl_rows], ignore_index=True)
    return out[[c for c in keep if c in out.columns]].reset_index(drop=True)


def category_counts(frame, min_cases_per_project, min_controls_per_project):
    """Per (disease_category, project): case/control counts and whether the
    project clears the per-project floor for that category."""
    g = (frame.groupby(["disease_category", "project", "y"]).size()
         .unstack("y", fill_value=0))
    g = g.rename(columns={CASE: "n_case", CONTROL: "n_control"})
    for c in ("n_case", "n_control"):
        if c not in g.columns:
            g[c] = 0
    g["ok"] = (g["n_case"] >= min_cases_per_project) & (g["n_control"] >= min_controls_per_project)
    return g.reset_index()


def eligible_categories(cat_counts, min_projects, min_cases_total,
                        exclude=NON_DISEASE_CATEGORIES):
    """Disease categories with >= `min_projects` projects that each clear the
    per-project floor, and >= `min_cases_total` cases pooled across them.
    `exclude` drops the non-disease buckets ("other", "healthy_only")."""
    ok = cat_counts[cat_counts["ok"]]
    agg = ok.groupby("disease_category").agg(
        n_projects=("project", "nunique"),
        n_cases=("n_case", "sum"),
        n_controls=("n_control", "sum"),
    )
    agg = agg[(~agg.index.isin(exclude))
              & (agg["n_projects"] >= min_projects)
              & (agg["n_cases"] >= min_cases_total)]
    return agg.sort_values("n_cases", ascending=False)


# ---------------------------------------------------------------------------
# Streamed CLR + sparse presence loaders
# ---------------------------------------------------------------------------

def load_clr(clr_path, samples):
    """CLR rows for `samples`, as (clr_df indexed by `sample`, taxa list).
    Reuses `src.variance.load_abundance_subset` so the wide file is never
    fully resident."""
    from src import variance as var

    ids, mat, taxa = var.load_abundance_subset(clr_path, set(samples))
    clr = pd.DataFrame(mat, columns=list(taxa))
    clr.insert(0, SAMPLE_COL, ids)
    return clr.set_index(SAMPLE_COL), list(taxa)


def presence_matrix(prev01_path):
    """Boolean presence (raw count > 0) for the >=1%-prevalence taxa, as a
    DataFrame indexed by `sample`, columns = taxonomy strings. Used to
    prevalence-filter *within* each small contrast -- a taxon common
    compendium-wide can be near-absent inside one project's samples, and
    testing it there is just noise."""
    from src import io as taxa_io

    mat, sample_ids, taxa = taxa_io.load_taxa_prev01(prev01_path)
    present = (mat > 0).toarray()
    return pd.DataFrame(present, index=np.asarray(sample_ids), columns=list(taxa))


def genus_labels(taxa):
    """Readable genus label per taxonomy string, falling back up the ranks
    when the genus itself is unresolved (`NA`)."""
    out = []
    for t in taxa:
        parts = str(t).split(".", maxsplit=5)
        label = parts[5] if len(parts) == 6 else str(t)
        if label in ("NA", "", "nan"):
            label = next((parts[i] for i in (4, 3, 2)
                          if len(parts) > i and parts[i] not in ("NA", "")), str(t))
        out.append(label)
    return out


# ---------------------------------------------------------------------------
# Shared statistics: vectorised OLS contrast, Wilcoxon, BH-FDR
# ---------------------------------------------------------------------------

def _bh_fdr(pvals):
    """Benjamini-Hochberg q-values, NaN-safe (NaN p stays NaN, is not
    counted in the correction)."""
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    m = np.isfinite(p)
    if m.any():
        q[m] = multipletests(p[m], method="fdr_bh")[1]
    return q


def _ols_contrast(Y, X, coef_idx):
    """Vectorised OLS of every column of `Y` (n x T) on the shared design
    `X` (n x p). Returns (beta, se, t, p) for design column `coef_idx` --
    the case/control indicator. NaN where the fit is degenerate."""
    n = X.shape[0]
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta_all = XtX_inv @ (X.T @ Y)                     # p x T
    resid = Y - X @ beta_all                            # n x T
    df_resid = n - np.linalg.matrix_rank(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma2 = (resid ** 2).sum(axis=0) / df_resid
        se = np.sqrt(XtX_inv[coef_idx, coef_idx] * sigma2)
        beta = beta_all[coef_idx]
        t = beta / se
        p = 2 * stats.t.sf(np.abs(t), df_resid)
    bad = (df_resid <= 0) | ~np.isfinite(se) | (se <= 0)
    beta = np.where(bad, np.nan, beta)
    se = np.where(bad, np.nan, se)
    t = np.where(bad, np.nan, t)
    p = np.where(bad, np.nan, p)
    return beta, se, t, p


def _wilcoxon(Y, y):
    """Per-taxon Mann-Whitney rank-sum p and rank-biserial correlation
    (positive => higher CLR in cases). Loops the ~few hundred columns --
    `mannwhitneyu` has no vectorised form and this is cheap at this width."""
    case, ctrl = Y[y == CASE], Y[y == CONTROL]
    n1, n2 = len(case), len(ctrl)
    p = np.full(Y.shape[1], np.nan)
    rbc = np.full(Y.shape[1], np.nan)
    if n1 < 3 or n2 < 3:
        return p, rbc
    for j in range(Y.shape[1]):
        a, b = case[:, j], ctrl[:, j]
        if np.ptp(np.concatenate([a, b])) == 0:
            continue
        try:
            U, pj = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        p[j] = pj
        rbc[j] = 2.0 * U / (n1 * n2) - 1.0
    return p, rbc


def _build_design(meta_sub, keys, covariates, add_indicator=True):
    """Design matrix [const, (y), covars...] aligned to `keys`, plus the
    row mask of samples with a usable value for every chosen covariate.
    `sex` -> male indicator; numeric covariates -> standardised."""
    gi = meta_sub.set_index(SAMPLE_COL).loc[keys]
    cols, names = [np.ones(len(keys))], ["const"]
    if add_indicator:
        cols.append(gi["y"].to_numpy(float))
        names.append("y")
    ok = np.ones(len(keys), dtype=bool)
    for c in covariates:
        if c == "sex":
            s = gi["sex"].astype(str).str.lower().to_numpy()   # NaN/NA -> "nan"
            d = np.full(len(s), np.nan)
            d[s == "male"] = 1.0
            d[s == "female"] = 0.0
        else:
            v = pd.to_numeric(gi[c], errors="coerce").to_numpy(float)
            sd = np.nanstd(v)
            d = (v - np.nanmean(v)) / (sd if sd else 1.0)
        cols.append(d)
        names.append(c)
        ok &= np.isfinite(d)
    return np.column_stack(cols), names, ok


def _pick_covariates(frame_sub, candidates=("age_years", "sex"), min_coverage=0.8):
    """Keep a covariate only if both arms have >= `min_coverage` non-null
    for it and it actually varies."""
    use = []
    for c in candidates:
        if c not in frame_sub.columns:
            continue
        cov = frame_sub.groupby("y")[c].apply(lambda s: s.notna().mean())
        if (cov >= min_coverage).all() and frame_sub[c].nunique(dropna=True) > 1:
            use.append(c)
    return use


# ---------------------------------------------------------------------------
# Within-project differential abundance
# ---------------------------------------------------------------------------

def differential_within_project(clr_df, presence_df, frame, category, taxa, *,
                                within_prevalence=0.10, covariate_min_coverage=0.8,
                                min_cases=10, min_controls=10):
    """Per project carrying `category` cases: OLS case/control contrast on
    each within-project prevalence-filtered taxon, covariate-adjusted where
    coverage allows, with a Wilcoxon rank-sum carried alongside. BH-FDR
    within each project. One row per (project, taxon)."""
    rows = []
    sub_cat = frame[frame["disease_category"] == category]
    for proj, g in sub_cat.groupby("project"):
        g = g.drop_duplicates(SAMPLE_COL)
        g = g[g[SAMPLE_COL].isin(clr_df.index)]
        n_case = int((g["y"] == CASE).sum())
        n_ctrl = int((g["y"] == CONTROL).sum())
        if n_case < min_cases or n_ctrl < min_controls:
            continue

        keys = g[SAMPLE_COL].to_numpy()
        prev = presence_df.reindex(keys)[taxa].mean(axis=0)
        keep = [t for t in taxa if prev[t] >= within_prevalence]
        if not keep:
            continue

        covs = _pick_covariates(g, min_coverage=covariate_min_coverage)
        X, names, ok = _build_design(g, keys, covs)
        keys_ok = keys[ok]
        if (g.set_index(SAMPLE_COL).loc[keys_ok, "y"] == CASE).sum() < min_cases or \
           (g.set_index(SAMPLE_COL).loc[keys_ok, "y"] == CONTROL).sum() < min_controls:
            continue

        Y = clr_df.loc[keys_ok, keep].to_numpy(float)
        yv = g.set_index(SAMPLE_COL).loc[keys_ok, "y"].to_numpy(float)
        Xok = X[ok]

        beta, se, t, p = _ols_contrast(Y, Xok, names.index("y"))
        w_p, rbc = _wilcoxon(Y, yv)
        q = _bh_fdr(p)
        gen = genus_labels(keep)

        for j, tax in enumerate(keep):
            rows.append(dict(
                analysis="within_project", disease_category=category, project=proj,
                taxon=tax, genus=gen[j],
                n_case=int((yv == CASE).sum()), n_control=int((yv == CONTROL).sum()),
                effect=beta[j], se=se[j], stat=t[j], p=p[j], q=q[j],
                wilcoxon_p=w_p[j], rank_biserial=rbc[j],
                covariates=",".join(covs), n_taxa_tested=len(keep),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DerSimonian-Laird random-effects meta-analysis
# ---------------------------------------------------------------------------

def meta_analyze(per_project_df, *, min_projects_per_taxon=2):
    """Pool the per-project (effect, SE) across projects, per (category,
    taxon), with inverse-variance weights and a DerSimonian-Laird
    random-effects variance component. BH-FDR within each category."""
    out = []
    for (cat, taxon), g in per_project_df.groupby(["disease_category", "taxon"]):
        g = g[np.isfinite(g["effect"]) & np.isfinite(g["se"]) & (g["se"] > 0)]
        k = len(g)
        if k < min_projects_per_taxon:
            continue
        yi = g["effect"].to_numpy(float)
        vi = g["se"].to_numpy(float) ** 2
        wi = 1.0 / vi
        mu_fixed = (wi * yi).sum() / wi.sum()
        Q = float((wi * (yi - mu_fixed) ** 2).sum())
        dfree = k - 1
        C = wi.sum() - (wi ** 2).sum() / wi.sum()
        tau2 = max(0.0, (Q - dfree) / C) if C > 0 else 0.0
        wr = 1.0 / (vi + tau2)
        mu = (wr * yi).sum() / wr.sum()
        se = float(np.sqrt(1.0 / wr.sum()))
        z = mu / se
        out.append(dict(
            analysis="meta", disease_category=cat, taxon=taxon,
            genus=g["genus"].iloc[0], k_projects=k,
            pooled_effect=float(mu), se=se,
            ci_low=float(mu - 1.96 * se), ci_high=float(mu + 1.96 * se),
            z=float(z), p=float(2 * stats.norm.sf(abs(z))),
            Q=Q, I2=float(max(0.0, (Q - dfree) / Q) * 100) if Q > 0 else 0.0,
            tau2=tau2,
            n_case=int(g["n_case"].sum()), n_control=int(g["n_control"].sum()),
            sign_consistency=float(np.mean(np.sign(yi) == np.sign(mu))),
        ))
    df = pd.DataFrame(out)
    if len(df):
        df["q"] = df.groupby("disease_category")["p"].transform(_bh_fdr)
    return df


# ---------------------------------------------------------------------------
# Naive pooled + batch correction
# ---------------------------------------------------------------------------

def combat_adjust(Y, batch, mod=None, *, eb=True, max_iter=200, tol=1e-4):
    """Parametric ComBat (Johnson, Li & Rabinovic 2007) for continuous data
    -- location/scale batch adjustment with an empirical-Bayes shrinkage of
    the per-batch parameters. `Y` is n x T (samples x taxa; CLR is a good
    fit -- roughly Gaussian, already log-ratio). `mod` is an n x q design of
    biological covariates to PROTECT from the adjustment (pass the
    case/control indicator here, plus any age/sex terms). Returns the
    batch-adjusted `Y`."""
    Y = np.asarray(Y, dtype=float)
    n, T = Y.shape
    batch = np.asarray(batch)
    levels, bidx = np.unique(batch, return_inverse=True)
    nb = len(levels)
    B = np.zeros((n, nb))
    B[np.arange(n), bidx] = 1.0
    ni = B.sum(axis=0)

    design = B if mod is None or np.size(mod) == 0 else np.column_stack([B, np.asarray(mod, float)])
    coef = np.linalg.pinv(design.T @ design) @ (design.T @ Y)      # (nb+q) x T

    grand_mean = (ni / n) @ coef[:nb]                               # T
    stand_mean = np.tile(grand_mean, (n, 1))
    if mod is not None and np.size(mod):
        stand_mean = stand_mean + np.asarray(mod, float) @ coef[nb:]

    resid = Y - design @ coef
    var_pooled = np.maximum((resid ** 2).sum(axis=0) / n, 1e-8)     # T (MLE)
    Z = (Y - stand_mean) / np.sqrt(var_pooled)

    gamma_hat = np.vstack([Z[bidx == i].mean(axis=0) for i in range(nb)])
    delta_hat = np.vstack([np.maximum(Z[bidx == i].var(axis=0, ddof=1), 1e-8)
                           for i in range(nb)])

    if eb:
        gamma_star = np.empty_like(gamma_hat)
        delta_star = np.empty_like(delta_hat)
        for i in range(nb):
            gbar = gamma_hat[i].mean()
            t2 = gamma_hat[i].var(ddof=1)
            dm, dv = delta_hat[i].mean(), delta_hat[i].var(ddof=1)
            lam = (dm ** 2 + 2 * dv) / dv
            theta = (dm ** 3 + dm * dv) / dv
            g, d = gamma_hat[i].copy(), delta_hat[i].copy()
            Zi = Z[bidx == i]
            for _ in range(max_iter):
                g_new = (ni[i] * t2 * gamma_hat[i] + d * gbar) / (ni[i] * t2 + d)
                s = ((Zi - g_new) ** 2).sum(axis=0)
                d_new = (0.5 * s + theta) / (0.5 * ni[i] + lam - 1)
                if (np.max(np.abs(g_new - g) / (np.abs(g) + 1e-8)) < tol and
                        np.max(np.abs(d_new - d) / (np.abs(d) + 1e-8)) < tol):
                    g, d = g_new, d_new
                    break
                g, d = g_new, d_new
            gamma_star[i], delta_star[i] = g, d
    else:
        gamma_star, delta_star = gamma_hat, delta_hat

    Zc = Z.copy()
    for i in range(nb):
        m = bidx == i
        Zc[m] = (Z[m] - gamma_star[i]) / np.sqrt(delta_star[i])
    return Zc * np.sqrt(var_pooled) + stand_mean


def _pooled_samples(frame, category, clr_index):
    """The (keys, y, project, meta_sub) tuple for one category's pooled
    case+control set, restricted to samples present in the CLR matrix."""
    sub = frame[frame["disease_category"] == category].drop_duplicates(SAMPLE_COL)
    sub = sub[sub[SAMPLE_COL].isin(clr_index)]
    keys = sub[SAMPLE_COL].to_numpy()
    gi = sub.set_index(SAMPLE_COL).loc[keys]
    return keys, gi["y"].to_numpy(float), gi["project"].to_numpy(), sub


def differential_pooled(clr_df, presence_df, frame, category, taxa, *,
                        methods=("combat_ols", "mixedlm"), within_prevalence=0.10,
                        covariate_min_coverage=0.8, progress=False):
    """Pool a category's case+control samples across projects and test each
    prevalence-filtered taxon, correcting for project two ways:

    - **combat_ols** -- `combat_adjust` the CLR matrix by project (protecting
      the case indicator + any covariates), then a vectorised OLS contrast.
    - **mixedlm** -- `statsmodels` MixedLM with a project random intercept,
      per taxon (iteration-capped; non-convergence is recorded, not silently
      dropped).

    BH-FDR within category, per method. One row per (method, taxon)."""
    keys, y, proj, sub = _pooled_samples(frame, category, clr_df.index)
    prev = presence_df.reindex(keys)[taxa].mean(axis=0)
    keep = [t for t in taxa if prev[t] >= within_prevalence]
    if not keep:
        return pd.DataFrame()

    covs = _pick_covariates(sub, min_coverage=covariate_min_coverage)
    X, names, ok = _build_design(sub, keys, covs)
    keys, y, proj, X = keys[ok], y[ok], proj[ok], X[ok]
    Y = clr_df.loc[keys, keep].to_numpy(float)
    gen = genus_labels(keep)
    mod = X[:, [names.index("y")] + [names.index(c) for c in covs]]
    n_case, n_ctrl, k_proj = int((y == CASE).sum()), int((y == CONTROL).sum()), len(np.unique(proj))
    frames = []

    if "combat_ols" in methods:
        Yc = combat_adjust(Y, proj, mod=mod)
        beta, se, t, p = _ols_contrast(Yc, X, names.index("y"))
        frames.append(pd.DataFrame(dict(
            analysis="pooled", method="combat_ols", disease_category=category,
            taxon=keep, genus=gen, effect=beta, se=se, stat=t, p=p, q=_bh_fdr(p),
            ci_low=beta - 1.96 * se, ci_high=beta + 1.96 * se,
            n_case=n_case, n_control=n_ctrl, k_projects=k_proj, n_taxa_tested=len(keep),
        )))

    if "mixedlm" in methods:
        import statsmodels.formula.api as smf

        base = pd.DataFrame({"y": y, "project": proj})
        for c in covs:
            base[c] = X[:, names.index(c)]
        formula = "clr ~ y" + "".join(f" + {c}" for c in covs)
        it = range(len(keep))
        if progress:
            try:
                from tqdm.auto import tqdm
                it = tqdm(it, desc=f"mixedlm {category}", leave=False)
            except ImportError:
                pass
        recs = []
        for j in it:
            dd = base.assign(clr=Y[:, j])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # capped iterations -- lbfgs on a near-degenerate taxon
                    # can otherwise loop for minutes; a non-converged fit is
                    # recorded as such, not chased.
                    m = smf.mixedlm(formula, dd, groups=dd["project"]).fit(
                        method="lbfgs", maxiter=200)
                recs.append((m.params.get("y", np.nan), m.bse.get("y", np.nan),
                             m.pvalues.get("y", np.nan), bool(m.converged)))
            except Exception:
                recs.append((np.nan, np.nan, np.nan, False))
        md = pd.DataFrame(recs, columns=["effect", "se", "p", "converged"])
        md.insert(0, "taxon", keep)
        md["genus"] = gen
        md["q"] = _bh_fdr(md["p"])
        md = md.assign(
            analysis="pooled", method="mixedlm", disease_category=category,
            stat=md["effect"] / md["se"],
            ci_low=md["effect"] - 1.96 * md["se"], ci_high=md["effect"] + 1.96 * md["se"],
            n_case=n_case, n_control=n_ctrl, k_projects=k_proj, n_taxa_tested=len(keep),
        )
        frames.append(md)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Concordance
# ---------------------------------------------------------------------------

def concordance(meta_df, pooled_df, *, pooled_method="combat_ols", fdr_alpha=0.05):
    """Join the meta-analysis and one pooled-test method per (category,
    taxon), classify each taxon, and summarise per category. Returns
    (per_taxon_df, summary_df)."""
    m = meta_df.rename(columns={"pooled_effect": "meta_effect", "q": "meta_q", "p": "meta_p"})
    p = (pooled_df[pooled_df["method"] == pooled_method]
         .rename(columns={"effect": "pooled_effect", "q": "pooled_q", "p": "pooled_p"}))
    j = m[["disease_category", "taxon", "genus", "meta_effect", "meta_p", "meta_q", "k_projects", "I2"]].merge(
        p[["disease_category", "taxon", "pooled_effect", "pooled_p", "pooled_q"]],
        on=["disease_category", "taxon"], how="outer",
    )

    def klass(r):
        msig = np.isfinite(r.meta_q) and r.meta_q < fdr_alpha
        psig = np.isfinite(r.pooled_q) and r.pooled_q < fdr_alpha
        if msig and psig:
            return "replicated"
        if psig:
            return "pooling_only"
        if msig:
            return "within_only"
        return "ns"

    j["concordance_class"] = j.apply(klass, axis=1)
    j["sign_agree"] = np.sign(j["meta_effect"]) == np.sign(j["pooled_effect"])
    j["pooled_method"] = pooled_method

    rows = []
    for cat, g in j.groupby("disease_category"):
        both = g.dropna(subset=["meta_effect", "pooled_effect"])
        rho = (both["meta_effect"].corr(both["pooled_effect"], method="spearman")
               if len(both) > 2 else np.nan)
        rows.append(dict(
            disease_category=cat, pooled_method=pooled_method,
            n_taxa=len(g), n_tested_both=len(both),
            n_replicated=int((g["concordance_class"] == "replicated").sum()),
            n_pooling_only=int((g["concordance_class"] == "pooling_only").sum()),
            n_within_only=int((g["concordance_class"] == "within_only").sum()),
            spearman_effect=float(rho) if pd.notna(rho) else np.nan,
            sign_agreement=float(both["sign_agree"].mean()) if len(both) else np.nan,
        ))
    return j, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batch-leakage quantification
# ---------------------------------------------------------------------------

def batch_leakage_cv(clr_df, frame, category, seed, *, n_splits=5,
                     models=("logreg", "gboost"), logreg_C=1.0,
                     gboost_max_iter=300, gboost_learning_rate=0.05, gboost_max_depth=3):
    """Disease classifier (case vs control) under two CV schemes:
    `leave_project_out` -- every test sample comes from a project the model
    never saw -- and `random_kfold` -- test samples share projects with
    training. Per fold: a `StandardScaler` fit on training rows only (for
    the linear model), then AUC + balanced accuracy on the held-out fold.
    The gap between the two schemes is the batch-leakage measurement."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    sub = frame[frame["disease_category"] == category].drop_duplicates(SAMPLE_COL)
    sub = sub[sub[SAMPLE_COL].isin(clr_df.index)]
    keys = sub[SAMPLE_COL].to_numpy()
    X = clr_df.loc[keys].to_numpy(float)
    y = sub.set_index(SAMPLE_COL).loc[keys, "y"].to_numpy(int)
    groups = sub.set_index(SAMPLE_COL).loc[keys, "project"].to_numpy()

    k = int(min(n_splits, np.bincount(y).min()))
    schemes = {
        "leave_project_out": list(LeaveOneGroupOut().split(X, y, groups)),
    }
    if k >= 2:
        schemes["random_kfold"] = list(
            StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y))

    def make(name):
        if name == "logreg":
            # liblinear defaults to L2; passing penalty= explicitly is
            # deprecated in recent sklearn, so rely on the default.
            return LogisticRegression(C=logreg_C, max_iter=5000,
                                      class_weight="balanced", solver="liblinear")
        return HistGradientBoostingClassifier(
            random_state=seed, max_iter=gboost_max_iter,
            learning_rate=gboost_learning_rate, max_depth=gboost_max_depth)

    rows = []
    for scheme, folds in schemes.items():
        for fold, (tr, te) in enumerate(folds):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            scaler = StandardScaler().fit(X[tr])
            Xtr_s, Xte_s = scaler.transform(X[tr]), scaler.transform(X[te])
            for name in models:
                clf = make(name)
                if name == "logreg":
                    clf.fit(Xtr_s, y[tr])
                    prob = clf.predict_proba(Xte_s)[:, 1]
                else:
                    clf.fit(X[tr], y[tr])
                    prob = clf.predict_proba(X[te])[:, 1]
                rows.append(dict(
                    disease_category=category, model=name, cv_scheme=scheme, fold=fold,
                    auc=float(roc_auc_score(y[te], prob)),
                    balanced_accuracy=float(balanced_accuracy_score(y[te], (prob >= 0.5).astype(int))),
                    n_train=int(len(tr)), n_test=int(len(te)), n_test_pos=int(y[te].sum()),
                    test_project=str(groups[te][0]) if scheme == "leave_project_out" else "",
                ))
    return pd.DataFrame(rows)


def batch_leakage_summary(cv_df):
    """(long per-(category, model, scheme) means, wide table with the
    random - LOPO AUC gap)."""
    g = (cv_df.groupby(["disease_category", "model", "cv_scheme"])
         .agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
              bal_acc_mean=("balanced_accuracy", "mean"), n_folds=("fold", "size"))
         .reset_index())
    wide = g.pivot_table(index=["disease_category", "model"], columns="cv_scheme",
                         values="auc_mean")
    if {"random_kfold", "leave_project_out"}.issubset(wide.columns):
        wide["auc_gap"] = wide["random_kfold"] - wide["leave_project_out"]
    return g, wide.reset_index()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def differential_report_md(elig_df, meta_df, concordance_summary, leakage_wide,
                           *, fdr_alpha=0.05, pooled_method="combat_ols"):
    """Human-readable companion to the differential-abundance CSVs."""
    lines = ["# Disease association", ""]

    lines += ["## Disease categories analysed", "",
              "| category | projects | cases | controls |", "|---|---|---|---|"]
    for r in elig_df.reset_index().itertuples(index=False):
        lines.append(f"| {r.disease_category} | {r.n_projects} | {r.n_cases:,} | {r.n_controls:,} |")

    lines += ["", f"## Within-study meta-analysis hits, q < {fdr_alpha}", "",
              "| category | genus | pooled effect | 95% CI | I² | k | q |",
              "|---|---|---|---|---|---|---|"]
    hits = meta_df[meta_df["q"] < fdr_alpha].sort_values(["disease_category", "q"])
    if len(hits):
        for r in hits.itertuples(index=False):
            lines.append(
                f"| {r.disease_category} | {r.genus} | {r.pooled_effect:+.3f} | "
                f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.I2:.0f}% | {r.k_projects} | {r.q:.2e} |")
    else:
        lines.append("| — | none | | | | | |")

    lines += ["", f"## Concordance: meta-analysis vs pooled {pooled_method}", "",
              "| category | replicated | pooling-only | within-only | effect ρ | sign agree |",
              "|---|---|---|---|---|---|"]
    for r in concordance_summary.itertuples(index=False):
        rho = "—" if pd.isna(r.spearman_effect) else f"{r.spearman_effect:.2f}"
        sa = "—" if pd.isna(r.sign_agreement) else f"{r.sign_agreement:.0%}"
        lines.append(f"| {r.disease_category} | {r.n_replicated} | {r.n_pooling_only} | "
                     f"{r.n_within_only} | {rho} | {sa} |")

    lines += ["", "## Batch leakage: disease-classifier AUC by CV scheme", "",
              "| category | model | leave-project-out | random k-fold | gap |",
              "|---|---|---|---|---|"]
    for r in leakage_wide.itertuples(index=False):
        lpo = getattr(r, "leave_project_out", np.nan)
        rk = getattr(r, "random_kfold", np.nan)
        gap = getattr(r, "auc_gap", np.nan)
        lines.append(
            f"| {r.disease_category} | {r.model} | "
            f"{lpo:.3f} | {'—' if pd.isna(rk) else f'{rk:.3f}'} | "
            f"{'—' if pd.isna(gap) else f'{gap:+.3f}'} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    })


_CLASS_COLOR = {
    "replicated": "#55A868", "pooling_only": "#C44E52",
    "within_only": "#4C72B0", "ns": "#C8C8C8",
}


def plot_concordance(per_taxon_df, *, fdr_alpha=0.05):
    """Scatter grid, one panel per category: meta-analysis effect vs pooled
    effect, colour by concordance class. Off-diagonal spread and
    pooling-only points are the batch-artifact story."""
    import matplotlib.pyplot as plt

    _style()
    cats = sorted(per_taxon_df["disease_category"].dropna().unique())
    ncol = min(3, len(cats)) or 1
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, cat in zip(axes, cats):
        g = per_taxon_df[per_taxon_df["disease_category"] == cat]
        both = g.dropna(subset=["meta_effect", "pooled_effect"])
        for klass, sub in both.groupby("concordance_class"):
            ax.scatter(sub["meta_effect"], sub["pooled_effect"], s=18, alpha=0.75,
                       color=_CLASS_COLOR.get(klass, "#888"), label=klass, linewidths=0)
        if len(both):
            lim = np.nanmax(np.abs(np.r_[both["meta_effect"], both["pooled_effect"]])) * 1.1
            ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1, color="k", alpha=0.5)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            rho = both["meta_effect"].corr(both["pooled_effect"], method="spearman")
            ax.set_title(f"{cat}  (ρ={rho:.2f}, n={len(both)})")
        else:
            ax.set_title(cat)
        ax.axhline(0, color="k", lw=0.6, alpha=0.3)
        ax.axvline(0, color="k", lw=0.6, alpha=0.3)
        ax.set_xlabel("within-study meta-analysis effect")
        ax.set_ylabel("naive pooled effect")
        ax.legend(fontsize=7, framealpha=0.6)

    for ax in axes[len(cats):]:
        ax.set_visible(False)
    fig.suptitle("Concordance: within-study meta-analysis vs naive pooled", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_batch_leakage(summary_long):
    """Grouped bars: leave-project-out vs random k-fold AUC per category and
    model. The height difference is how much apparent accuracy is study
    recognition."""
    import matplotlib.pyplot as plt

    _style()
    d = summary_long.copy()
    d["key"] = d["disease_category"] + "\n" + d["model"]
    keys = list(dict.fromkeys(d["key"]))
    schemes = ["leave_project_out", "random_kfold"]
    colours = {"leave_project_out": "#4C72B0", "random_kfold": "#DD8452"}
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(keys)), 4.6))
    for i, scheme in enumerate(schemes):
        sub = d[d["cv_scheme"] == scheme].set_index("key").reindex(keys)
        ax.bar(np.arange(len(keys)) + i * width, sub["auc_mean"].to_numpy(), width,
               yerr=sub["auc_std"].to_numpy(), capsize=3,
               label=scheme.replace("_", " "), color=colours[scheme])
    ax.axhline(0.5, color="firebrick", ls="--", lw=1, label="chance (AUC 0.5)")
    ax.set_xticks(np.arange(len(keys)) + width / 2)
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylabel("ROC AUC (mean ± SD across folds)")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Disease-classifier AUC: leave-project-out vs random CV")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_meta_volcano(meta_df, *, fdr_alpha=0.05):
    """Volcano per category: pooled effect vs -log10(q), hits labelled with
    their genus."""
    import matplotlib.pyplot as plt

    _style()
    cats = sorted(meta_df["disease_category"].dropna().unique())
    ncol = min(3, len(cats)) or 1
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, cat in zip(axes, cats):
        g = meta_df[meta_df["disease_category"] == cat].copy()
        g["nlq"] = -np.log10(g["q"].clip(lower=1e-300))
        sig = g["q"] < fdr_alpha
        ax.scatter(g.loc[~sig, "pooled_effect"], g.loc[~sig, "nlq"], s=14,
                   color="#C8C8C8", linewidths=0)
        ax.scatter(g.loc[sig, "pooled_effect"], g.loc[sig, "nlq"], s=22,
                   color="#C44E52", linewidths=0)
        ax.axhline(-np.log10(fdr_alpha), ls="--", lw=1, color="k", alpha=0.5)
        ax.axvline(0, color="k", lw=0.6, alpha=0.3)
        for r in g[sig].sort_values("q").head(6).itertuples(index=False):
            ax.annotate(r.genus, (r.pooled_effect, -np.log10(max(r.q, 1e-300))),
                        fontsize=7, ha="center", va="bottom")
        ax.set_title(f"{cat}  ({int(sig.sum())} hits)")
        ax.set_xlabel("pooled effect (CLR, case − control)")
        ax.set_ylabel("−log10(q)")

    for ax in axes[len(cats):]:
        ax.set_visible(False)
    fig.suptitle("Within-study meta-analysis differential abundance", fontweight="bold")
    fig.tight_layout()
    return fig
