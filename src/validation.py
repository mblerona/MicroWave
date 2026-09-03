"""Does the disease-association result hold up when the analysis choices are
varied, and does it collapse when it should?

Six checks, all consuming the persisted differential-abundance artifacts and
the same cohort inputs the earlier steps used -- nothing here re-reads the
raw counts CSV:

1. **Label-shuffling null.** Permute case/control within each (project,
   category) and re-run the meta-analysis and the naive pooled test. A
   genuine pipeline collapses to ~zero FDR hits; anything that survives is a
   leak to find before trusting the real numbers.
2. **Depth-threshold sensitivity.** Rebuild the within-study cohort at
   5k / 10k / 20k minimum reads and re-run. Report how the *replicated*
   taxon set moves; a hit that only appears at one threshold is reported as
   such.
3. **Transform sensitivity.** Repeat the differential-abundance tests with
   the taxon response as log-relative abundance instead of CLR; report
   effect-size agreement and hit-set overlap.
4. **Prevalence-filter sensitivity.** Repeat restricting the tested taxa to
   those present in >=1% vs >=10% of the whole compendium.
5. **Literature cross-check.** Compare the replicated hits against a small
   curated set of established genus-level directions (orientation only --
   not extracted from these studies' own papers), and surface the source
   DOIs for manual follow-up.
6. **Final report.** Stitch the harmonisation, cohort-flow, variance and
   differential-abundance artifacts plus these sensitivity tables into one
   document, limitations first.

Plain functions, dataframes / paths in and out. The heavy lifting is
delegated to `src.differential`; this module orchestrates and compares.
"""
import numpy as np
import pandas as pd

from src import differential as dfa

SAMPLE_COL = dfa.SAMPLE_COL
CASE, CONTROL = dfa.CASE, dfa.CONTROL


# ---------------------------------------------------------------------------
# Shared: run the differential-abundance chain once and extract replicates
# ---------------------------------------------------------------------------

def run_da(clr_df, presence_df, frame, categories, dcfg, taxa, *,
           pooled_methods=("combat_ols",)):
    """One pass of within-project tests -> meta-analysis -> pooled test, over
    `categories`. `taxa` is the candidate universe (each contrast still
    applies its own within-prevalence filter). Returns (per_project, meta,
    pooled)."""
    per_project = pd.concat(
        [dfa.differential_within_project(
            clr_df, presence_df, frame, cat, taxa,
            within_prevalence=dcfg["within_prevalence"],
            covariate_min_coverage=dcfg["covariate_min_coverage"],
            min_cases=dcfg["min_cases_per_project"],
            min_controls=dcfg["min_controls_per_project"])
         for cat in categories],
        ignore_index=True,
    )
    if per_project.empty:
        return per_project, pd.DataFrame(), pd.DataFrame()
    meta = dfa.meta_analyze(per_project,
                            min_projects_per_taxon=dcfg["meta_min_projects_per_taxon"])
    pooled = pd.concat(
        [dfa.differential_pooled(
            clr_df, presence_df, frame, cat, taxa, methods=pooled_methods,
            within_prevalence=dcfg["within_prevalence"],
            covariate_min_coverage=dcfg["covariate_min_coverage"])
         for cat in categories],
        ignore_index=True,
    )
    return per_project, meta, pooled


def replicated_sets(meta, pooled, *, pooled_method="combat_ols", fdr_alpha=0.05):
    """{category: set(taxon)} significant in BOTH the meta-analysis and the
    pooled test, plus the full per-taxon concordance frame."""
    if meta.empty or pooled.empty:
        return {}, pd.DataFrame()
    per_taxon, _ = dfa.concordance(meta, pooled, pooled_method=pooled_method,
                                   fdr_alpha=fdr_alpha)
    rep = per_taxon[per_taxon["concordance_class"] == "replicated"]
    return {c: set(g["taxon"]) for c, g in rep.groupby("disease_category")}, per_taxon


# ---------------------------------------------------------------------------
# 1. Label-shuffling null
# ---------------------------------------------------------------------------

def shuffle_labels_within(frame, seed):
    """Permute `y` within each (project, disease_category) block -- keeps
    every arm's size and the covariate distribution intact, breaks only the
    label<->composition link."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    y = out["y"].to_numpy().copy()
    for _, idx in out.groupby(["project", "disease_category"], sort=False).indices.items():
        y[idx] = rng.permutation(y[idx])
    out["y"] = y
    return out


def label_shuffle_null(clr_df, presence_df, frame, categories, dcfg, taxa, *,
                       n_shuffles=10, seed=42, fdr_alpha=None, progress=True):
    """Re-run the meta-analysis and the pooled test on `n_shuffles` within-block
    label permutations. Returns (null_df, real_hits) where `null_df` has one
    row per (shuffle, category) with the FDR-hit counts and `real_hits` is
    the same count on the true labels for reference."""
    fdr_alpha = dcfg["fdr_alpha"] if fdr_alpha is None else fdr_alpha

    def hit_counts(meta, pooled, tag):
        rows = []
        for cat in categories:
            m = meta[meta["disease_category"] == cat] if not meta.empty else meta
            p = (pooled[(pooled["disease_category"] == cat) & (pooled["method"] == "combat_ols")]
                 if not pooled.empty else pooled)
            rows.append(dict(
                shuffle=tag, disease_category=cat,
                meta_hits=int((m["q"] < fdr_alpha).sum()) if len(m) else 0,
                pooled_hits=int((p["q"] < fdr_alpha).sum()) if len(p) else 0,
                n_meta_taxa=int(len(m)), n_pooled_taxa=int(len(p)),
            ))
        return pd.DataFrame(rows)

    _, meta0, pooled0 = run_da(clr_df, presence_df, frame, categories, dcfg, taxa)
    real_hits = hit_counts(meta0, pooled0, "real")

    it = range(n_shuffles)
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="label-shuffle null")
        except ImportError:
            pass

    parts = []
    for i in it:
        sh = shuffle_labels_within(frame, seed + 1 + i)
        _, meta_i, pooled_i = run_da(clr_df, presence_df, sh, categories, dcfg, taxa)
        parts.append(hit_counts(meta_i, pooled_i, i))
    null_df = pd.concat(parts, ignore_index=True)
    return null_df, real_hits


# ---------------------------------------------------------------------------
# 2. Depth-threshold sensitivity
# ---------------------------------------------------------------------------

def cohort_at_depth(harmonized, sample_depth, review_df, threshold, dcfg):
    """Rebuild the within-study cohort with a different minimum-reads filter,
    reusing the same QC + cohort logic the earlier step used."""
    from src import cohorts as coh

    qc_df, _ = coh.apply_qc_filters(harmonized, sample_depth, threshold)
    cohort, _ = coh.build_within_project_cohort(
        qc_df, review_df, dcfg["min_cases_per_project"], dcfg["min_controls_per_project"])
    return cohort


def depth_threshold_sweep(harmonized, sample_depth, review_df, clr_df, presence_df,
                          category_groups, categories_ref, dcfg, taxa, thresholds):
    """For each depth threshold: rebuild the cohort, re-run, and record the
    replicated set per category. Returns (long_df, stability_df) where
    `stability_df` marks each replicated taxon as stable across all
    thresholds or threshold-specific."""
    per_thr = {}
    long_rows = []
    for thr in thresholds:
        cohort = cohort_at_depth(harmonized, sample_depth, review_df, thr, dcfg)
        frame = dfa.build_analysis_frame(cohort, category_groups=category_groups)
        cc = dfa.category_counts(frame, dcfg["min_cases_per_project"],
                                 dcfg["min_controls_per_project"])
        elig = dfa.eligible_categories(cc, dcfg["min_projects_per_category"],
                                       dcfg["min_cases_per_category"])
        cats = list(elig.index)
        keys_present = set(frame[SAMPLE_COL]) & set(clr_df.index)
        frame = frame[frame[SAMPLE_COL].isin(keys_present)]
        _, meta, pooled = run_da(clr_df, presence_df, frame, cats, dcfg, taxa)
        rep, _ = replicated_sets(meta, pooled, fdr_alpha=dcfg["fdr_alpha"])
        per_thr[thr] = rep
        for cat in cats:
            long_rows.append(dict(
                threshold=thr, disease_category=cat,
                n_projects=int(elig.loc[cat, "n_projects"]) if cat in elig.index else 0,
                n_cases=int(elig.loc[cat, "n_cases"]) if cat in elig.index else 0,
                n_replicated=len(rep.get(cat, set())),
            ))
    long_df = pd.DataFrame(long_rows)

    all_cats = sorted({c for r in per_thr.values() for c in r})
    stab_rows = []
    for cat in all_cats:
        sets = {thr: per_thr[thr].get(cat, set()) for thr in thresholds}
        union = set().union(*sets.values())
        for taxon in sorted(union):
            present = [thr for thr in thresholds if taxon in sets[thr]]
            stab_rows.append(dict(
                disease_category=cat, taxon=taxon, genus=dfa.genus_labels([taxon])[0],
                thresholds=",".join(str(t) for t in present),
                stable=len(present) == len(thresholds),
            ))
    return long_df, pd.DataFrame(stab_rows)


# ---------------------------------------------------------------------------
# 3. Transform sensitivity  (CLR vs log-relative)
# ---------------------------------------------------------------------------

def load_log_relative(prev01_path, keys, pseudocount=0.5):
    """log10 relative abundance on the >=1%-prevalence closed sub-composition
    (same taxa CLR uses), indexed by `sample`. The additive pseudocount
    matches `transforms.compute_clr` so this is exactly CLR without the
    centring step -- the cleanest transform contrast."""
    from src import io as taxa_io

    mat, ids, taxa = taxa_io.load_taxa_prev01(prev01_path)
    mask = pd.Series(ids).isin(set(keys)).to_numpy()
    counts = np.asarray(mat[mask].todense(), dtype=np.float64)
    kept = np.asarray(ids)[mask]
    has_mass = counts.sum(axis=1) > 0
    counts, kept = counts[has_mass], kept[has_mass]
    adj = np.where(counts == 0, pseudocount, counts)
    p = adj / adj.sum(axis=1, keepdims=True)
    out = pd.DataFrame(np.log10(p), columns=list(taxa))
    out.insert(0, SAMPLE_COL, kept)
    return out.set_index(SAMPLE_COL)


def transform_sensitivity(clr_meta, clr_rep, alt_df, presence_df, frame, categories,
                          dcfg, taxa, *, label="log_relative"):
    """Re-run with `alt_df` as the taxon response and compare to the CLR run.
    Returns per-category overlap of meta hits + effect-size rank correlation."""
    _, meta_alt, pooled_alt = run_da(alt_df, presence_df, frame, categories, dcfg, taxa)
    rep_alt, _ = replicated_sets(meta_alt, pooled_alt, fdr_alpha=dcfg["fdr_alpha"])

    rows = []
    for cat in categories:
        m0 = clr_meta[clr_meta["disease_category"] == cat].set_index("taxon")
        m1 = meta_alt[meta_alt["disease_category"] == cat].set_index("taxon") if not meta_alt.empty else m0.iloc[:0]
        common = m0.index.intersection(m1.index)
        rho = (m0.loc[common, "pooled_effect"].corr(m1.loc[common, "pooled_effect"], method="spearman")
               if len(common) > 2 else np.nan)
        h0 = set(m0.index[m0["q"] < dcfg["fdr_alpha"]])
        h1 = set(m1.index[m1["q"] < dcfg["fdr_alpha"]])
        r0, r1 = clr_rep.get(cat, set()), rep_alt.get(cat, set())
        rows.append(dict(
            transform=label, disease_category=cat,
            meta_hits_clr=len(h0), meta_hits_alt=len(h1),
            meta_hits_shared=len(h0 & h1),
            replicated_clr=len(r0), replicated_alt=len(r1),
            replicated_shared=len(r0 & r1),
            effect_spearman=float(rho) if pd.notna(rho) else np.nan,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Prevalence-filter sensitivity
# ---------------------------------------------------------------------------

def prevalence_sensitivity(clr_df, presence_df, frame, categories, dcfg, taxa, tiers):
    """Re-run restricting the tested taxa to those present in >= tier of the
    whole compendium. Returns (long_df, per_taxon_df) -- per-category
    replicated counts per tier, and each tier's replicated taxa."""
    compendium_prev = presence_df[taxa].mean(axis=0)
    long_rows, taxon_rows = [], []
    for tier in tiers:
        tier_taxa = [t for t in taxa if compendium_prev[t] >= tier]
        _, meta, pooled = run_da(clr_df, presence_df, frame, categories, dcfg, tier_taxa)
        rep, _ = replicated_sets(meta, pooled, fdr_alpha=dcfg["fdr_alpha"])
        for cat in categories:
            long_rows.append(dict(
                prevalence_tier=tier, n_taxa_universe=len(tier_taxa),
                disease_category=cat, n_replicated=len(rep.get(cat, set())),
            ))
            for taxon in sorted(rep.get(cat, set())):
                taxon_rows.append(dict(
                    prevalence_tier=tier, disease_category=cat, taxon=taxon,
                    genus=dfa.genus_labels([taxon])[0]))
    return pd.DataFrame(long_rows), pd.DataFrame(taxon_rows)


# ---------------------------------------------------------------------------
# 5. Literature cross-check
# ---------------------------------------------------------------------------

def literature_check(concordance_primary, priors, projects_df, per_project_df=None, *,
                     top_n=15, fdr_alpha=0.05):
    """Compare each category's replicated hits (top `top_n` by meta q) against
    the curated genus-direction priors. `expected` up/down comes from a
    case-insensitive substring match of the genus against the prior lists;
    `observed` from the sign of the meta-analysis effect. Returns
    (per_hit_df, rollup_df); `rollup_df` also carries the source DOIs."""
    rep = concordance_primary[concordance_primary["concordance_class"] == "replicated"].copy()

    def expected_dir(genus, cat):
        pr = priors.get(cat, {})
        g = str(genus).lower()
        for d in ("up", "down"):
            for name in pr.get(d, []) or []:
                if str(name).lower() in g:
                    return d
        return "no_prior"

    hit_rows = []
    for cat, g in rep.groupby("disease_category"):
        g = g.sort_values("meta_q").head(top_n)
        for r in g.itertuples(index=False):
            exp = expected_dir(r.genus, cat)
            obs = "up" if r.meta_effect > 0 else "down"
            verdict = ("no_prior" if exp == "no_prior"
                       else "concordant" if exp == obs else "discordant")
            hit_rows.append(dict(
                disease_category=cat, genus=r.genus, taxon=r.taxon,
                meta_effect=r.meta_effect, meta_q=r.meta_q,
                expected=exp, observed=obs, verdict=verdict))
    per_hit = pd.DataFrame(hit_rows)

    rep_with_proj = (rep.merge(per_project_df[["disease_category", "taxon", "project"]].drop_duplicates(),
                               on=["disease_category", "taxon"], how="left")
                     if per_project_df is not None else rep)
    doi_by_cat = _dois_per_category(rep_with_proj, projects_df)
    roll_rows = []
    for cat in sorted(rep["disease_category"].unique()):
        h = per_hit[per_hit["disease_category"] == cat] if len(per_hit) else per_hit
        roll_rows.append(dict(
            disease_category=cat, n_checked=len(h),
            concordant=int((h["verdict"] == "concordant").sum()) if len(h) else 0,
            discordant=int((h["verdict"] == "discordant").sum()) if len(h) else 0,
            no_prior=int((h["verdict"] == "no_prior").sum()) if len(h) else 0,
            source_dois="; ".join(doi_by_cat.get(cat, [])),
        ))
    return per_hit, pd.DataFrame(roll_rows)


def _dois_per_category(rep_df, projects_df):
    """Map each replicated category to the DOIs of the projects that carried
    its cases -- the pointer for a manual read of the source papers. Needs a
    `project` column on `rep_df`; falls back to empty when unavailable."""
    if "project" not in rep_df.columns:
        return {}
    link = projects_df.set_index("project")["link"].to_dict()
    out = {}
    for cat, g in rep_df.groupby("disease_category"):
        dois = sorted({link.get(p) for p in g["project"].dropna().unique() if link.get(p)})
        out[cat] = dois
    return out


# ---------------------------------------------------------------------------
# 6. Final report
# ---------------------------------------------------------------------------

_LIMITATIONS = [
    "**The sample is not the world.** ~61% of samples are from Europe and North "
    "America, 36% from the United States alone; ~17% have no usable country. Any "
    "claim about \"the human gut microbiome\" is really a claim about a mostly "
    "Western one.",
    "**Genus-level resolution only.** Broad bacterial groups, not species or "
    "strains. Strain-level effects are invisible here.",
    "**Observational, so no causal claims.** An elevated genus in a disease may be "
    "a consequence of the disease, its treatment, or an associated diet change.",
    "**Most samples have no health label.** ~81.5% are unlabelled; the disease work "
    "rests on the labelled minority, and the within-study comparison on 41 projects.",
    "**Health labels are self-defined by each study.** One study's \"healthy\" is "
    "another's \"mild symptoms\" -- the words are harmonised, the underlying "
    "clinical assessments are not.",
    "**Some labels are participant-reported, not clinician-assigned.** Only "
    "clinician diagnoses are counted as cases; weaker levels are retained as a "
    "graded covariate for sensitivity use only.",
    "**Diet / smoking / antibiotic history come from essentially one cohort** and "
    "describe that cohort, not the compendium.",
]


def final_report_md(reports_dir, elig_df, null_df, real_hits, depth_long, depth_stab,
                    transform_df, prevalence_long, lit_rollup, *, fdr_alpha=0.05):
    """Stitch the persisted artifacts and the sensitivity tables into one
    document -- limitations first, then the headline result and every
    robustness check behind it."""
    import re
    from pathlib import Path

    reports_dir = Path(reports_dir)

    def _grab(name, start=None, end=None):
        """Pull a slice out of a sibling report, drop its own H1 title, and
        demote its section headings one level so they nest under this
        report's `##` sections."""
        p = reports_dir / name
        if not p.exists():
            return f"_({name} not found)_"
        text = p.read_text(encoding="utf-8")
        if start is not None:
            s = text.find(start)
            if s >= 0:
                e = text.find(end, s) if end else -1
                text = text[s:(e if e > 0 else len(text))]
        text = re.sub(r"^# .*$", "", text, count=1, flags=re.M)   # drop the H1
        text = re.sub(r"^(#{2,5}) ", r"#\1 ", text, flags=re.M)   # ## -> ###
        return text.strip()

    L = ["# Human Microbiome Compendium — final report", "",
         "## Limitations, stated up front", ""]
    L += [f"- {x}" for x in _LIMITATIONS]

    L += ["", "## What was built", "",
          "A cleaned per-sample table, an auditable filtering flow, a "
          "variance-partitioning answer for the technical vs biological split, a "
          "ranked list of disease-associated genera with a replication verdict, "
          "and the robustness checks below. The numbered notebooks run in order "
          "from the raw files; this report consumes their outputs.", ""]

    L += ["## Metadata harmonisation — coverage", "",
          _grab("harmonization_coverage.md"), ""]

    L += ["## Cohort flow", "", _grab("cohort_flow.md"), ""]

    L += ["## Goal 2 — how much variation is the lab, not the person", "",
          _grab("variance_report.md", "## Marginal PERMANOVA", "## Project-predictability"),
          "", _grab("variance_report.md", "## Project-predictability", "## Depth vs alpha"), ""]

    L += ["## Goal 1 — disease-associated genera", "",
          _grab("differential_report.md", "## Disease categories analysed", "## Within-study"),
          "", _grab("differential_report.md", "## Concordance", "## Batch leakage"),
          "", _grab("differential_report.md", "## Batch leakage"), ""]

    # ---- Phase-6 robustness ----
    L += ["## Robustness checks", "", "### Label-shuffling null", "",
          "Case/control permuted within every (project, category); the "
          "meta-analysis and pooled test re-run. Under a working pipeline the "
          "FDR hit count collapses to near zero.", "",
          "| | meta hits (Σ over categories) | pooled hits |", "|---|---|---|"]
    real_m = int(real_hits["meta_hits"].sum())
    real_p = int(real_hits["pooled_hits"].sum())
    nm = null_df.groupby("shuffle")["meta_hits"].sum()
    npd = null_df.groupby("shuffle")["pooled_hits"].sum()
    L += [f"| real labels | {real_m} | {real_p} |",
          f"| shuffled (mean ± max over {nm.size} permutations) | "
          f"{nm.mean():.1f} ± {int(nm.max())} | {npd.mean():.1f} ± {int(npd.max())} |", ""]

    L += ["### Depth-threshold sensitivity", "",
          "Within-study cohort rebuilt at each minimum-reads cut; replicated "
          "genera per category:", "",
          "| category | " + " | ".join(f"{t:,}" for t in sorted(depth_long['threshold'].unique())) + " |",
          "|---|" + "---|" * depth_long["threshold"].nunique()]
    piv = depth_long.pivot_table(index="disease_category", columns="threshold",
                                 values="n_replicated", fill_value=0)
    for cat, row in piv.iterrows():
        L.append(f"| {cat} | " + " | ".join(str(int(row[t])) for t in sorted(piv.columns)) + " |")
    n_stable = int(depth_stab["stable"].sum()) if len(depth_stab) else 0
    L += ["", f"{n_stable} of {len(depth_stab)} replicated genera hold at every "
          f"threshold (`validation_depth_sensitivity.csv` lists the rest).", ""]

    L += ["### Transform sensitivity (CLR vs log-relative abundance)", "",
          "| category | meta hits CLR | meta hits log-rel | shared | effect ρ |",
          "|---|---|---|---|---|"]
    for r in transform_df.itertuples(index=False):
        rho = "—" if pd.isna(r.effect_spearman) else f"{r.effect_spearman:.2f}"
        L.append(f"| {r.disease_category} | {r.meta_hits_clr} | {r.meta_hits_alt} | "
                 f"{r.meta_hits_shared} | {rho} |")

    L += ["", "### Prevalence-filter sensitivity", "",
          "| category | " + " | ".join(f"≥{int(t*100)}%" for t in sorted(prevalence_long['prevalence_tier'].unique())) + " |",
          "|---|" + "---|" * prevalence_long["prevalence_tier"].nunique()]
    pv = prevalence_long.pivot_table(index="disease_category", columns="prevalence_tier",
                                     values="n_replicated", fill_value=0)
    for cat, row in pv.iterrows():
        L.append(f"| {cat} | " + " | ".join(str(int(row[t])) for t in sorted(pv.columns)) + " |")

    L += ["", "### Literature cross-check", "",
          "Replicated hits vs curated genus-direction expectations (orientation "
          "only — not extracted from these studies' papers; verify against the "
          "DOIs in `literature_crosscheck.csv`).", "",
          "| category | checked | concordant | discordant | no prior |",
          "|---|---|---|---|---|"]
    for r in lit_rollup.itertuples(index=False):
        L.append(f"| {r.disease_category} | {r.n_checked} | {r.concordant} | "
                 f"{r.discordant} | {r.no_prior} |")

    L += ["", "## Reading the result", "",
          "The centrepiece is the *replicated* set: genera significant in the "
          "within-study meta-analysis **and** the naive pooled test. Genera "
          "significant only when samples are pooled across labs are flagged as "
          "likely batch artifacts, not findings. The batch-leakage gap (random "
          "CV minus leave-one-project-out CV) quantifies, per disease, how much "
          "of a classifier's apparent accuracy is study recognition rather than "
          "biology.", ""]
    return "\n".join(L) + "\n"


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


def plot_null(null_df, real_hits):
    """Real FDR hits per category vs the shuffled-label distribution."""
    import matplotlib.pyplot as plt

    _style()
    cats = list(real_hits["disease_category"])
    x = np.arange(len(cats))
    real = real_hits.set_index("disease_category")["meta_hits"].reindex(cats).to_numpy()
    g = null_df.groupby("disease_category")["meta_hits"]
    nmean = g.mean().reindex(cats).to_numpy()
    nmax = g.max().reindex(cats).to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - 0.2, real, 0.4, label="real labels", color="#55A868")
    ax.bar(x + 0.2, nmean, 0.4, yerr=[np.zeros_like(nmean), nmax - nmean], capsize=3,
           label="shuffled (mean, whisker=max)", color="#C8C8C8")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.set_ylabel("meta-analysis FDR hits")
    ax.set_title("Label-shuffling null — the signal collapses when labels are broken")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_depth_sensitivity(depth_long):
    """Replicated genera per category at each depth threshold."""
    import matplotlib.pyplot as plt

    _style()
    piv = depth_long.pivot_table(index="disease_category", columns="threshold",
                                 values="n_replicated", fill_value=0)
    thr = sorted(piv.columns)
    x = np.arange(len(piv.index))
    width = 0.8 / len(thr)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, t in enumerate(thr):
        ax.bar(x + i * width, piv[t].to_numpy(), width, label=f"{t:,} reads")
    ax.set_xticks(x + width * (len(thr) - 1) / 2)
    ax.set_xticklabels(piv.index, rotation=25, ha="right")
    ax.set_ylabel("replicated genera")
    ax.set_title("Depth-threshold sensitivity")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_robustness(transform_df, prevalence_long):
    """Transform and prevalence-filter sensitivity side by side."""
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    d = transform_df
    x = np.arange(len(d))
    axes[0].bar(x - 0.2, d["meta_hits_clr"], 0.4, label="CLR", color="#4C72B0")
    axes[0].bar(x + 0.2, d["meta_hits_alt"], 0.4, label="log-relative", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(d["disease_category"], rotation=25, ha="right")
    axes[0].set_ylabel("meta-analysis FDR hits")
    axes[0].set_title("Transform sensitivity")
    axes[0].legend()

    pv = prevalence_long.pivot_table(index="disease_category", columns="prevalence_tier",
                                     values="n_replicated", fill_value=0)
    tiers = sorted(pv.columns)
    xx = np.arange(len(pv.index))
    w = 0.8 / len(tiers)
    for i, t in enumerate(tiers):
        axes[1].bar(xx + i * w, pv[t].to_numpy(), w, label=f"≥{int(t * 100)}%")
    axes[1].set_xticks(xx + w * (len(tiers) - 1) / 2)
    axes[1].set_xticklabels(pv.index, rotation=25, ha="right")
    axes[1].set_ylabel("replicated genera")
    axes[1].set_title("Prevalence-filter sensitivity")
    axes[1].legend()

    fig.tight_layout()
    return fig
