"""Phase 2 -- QC filtering and cohort construction. Turns
`data/interim/samples_harmonized.parquet` into an auditable, filtered
working set and the three named cohorts IMPLEMENTATION.md's Phase 5 needs.

Every filter here is logged with a before/after count (the flow table is
itself a deliverable), and no cohort's membership is ever read off
`projects.csv.condition` -- ANALYSIS_PLAN.md Section 4 already showed that
field misdescribes its own contents in both directions. Cohort membership
comes only from `disease_label`, which Phase 1 built from per-sample
values curated in `config/within_study_review.csv`.
"""
import numpy as np
import pandas as pd

SAMPLE_COL = "sample"  # sample_depth.parquet's join key: "{project}_{srr}"


# ---------------------------------------------------------------------------
# Step 1: QC exclusions, in the fixed order IMPLEMENTATION.md specifies
# ---------------------------------------------------------------------------

def apply_qc_filters(harmonized_df, sample_depth_df, depth_threshold):
    """Applies the three exclusions in order, returning (qc_df, flow) where
    `flow` is a list of {step, n_in, n_removed, n_out, note} dicts -- the
    raw material for the CONSORT-style flow table.

    1. Non-human `host_species`. Only samples POSITIVELY identified as
       non-human are dropped (`host_species_human is False`, 503 samples).
       The `<NA>` bucket (46,005 samples: no host tag at all, or free text
       naming neither) is kept, not dropped -- absence of evidence isn't
       evidence of non-human, and treating it as an automatic pass would
       hide the ambiguity rather than surface it. It is reported as its
       own flow-table line so the ambiguity stays visible.
    2. Non-stool `sample_type` (from projects.csv, project-level). Kept
       only when the value is exactly "stool" -- every combination
       ("stool and milk", "stool, saliva", ...) and every missing value is
       dropped, since a mixed-body-site project can't be split into
       stool/non-stool at the sample level from this field alone.
    3. Depth below `depth_threshold` reads, joined from `sample_depth.parquet`
       on the `{project}_{srr}` key Phase 0 used.
    """
    df = harmonized_df.copy()
    flow = []
    n0 = len(df)

    is_confirmed_non_human = df["host_species_human"] == False  # noqa: E712 (pd.NA-safe; `is False` fails)
    n_removed = int(is_confirmed_non_human.sum())
    n_unknown = int(df["host_species_human"].isna().sum())
    df = df[~is_confirmed_non_human]
    flow.append({
        "step": "exclude confirmed non-human host_species",
        "n_in": n0, "n_removed": n_removed, "n_out": len(df),
        "note": f"{n_unknown:,} samples with unresolved host_species (<NA>) kept, not dropped",
    })

    n_in = len(df)
    is_stool = df["sample_type"].astype(str).str.strip().str.lower() == "stool"
    df = df[is_stool]
    flow.append({
        "step": "exclude non-stool / missing sample_type",
        "n_in": n_in, "n_removed": n_in - len(df), "n_out": len(df),
        "note": "kept only sample_type == \"stool\" exactly; mixed body sites and missing values dropped",
    })

    n_in = len(df)
    depth_key = df["project"] + "_" + df["srr"]
    depth_by_key = sample_depth_df.set_index(SAMPLE_COL)["depth"]
    df = df.assign(depth=depth_key.map(depth_by_key).to_numpy())
    assert df["depth"].notna().all(), "every QC-surviving sample must have a depth from Phase 0"
    passes_depth = df["depth"] >= depth_threshold
    df = df[passes_depth]
    flow.append({
        "step": f"exclude depth < {depth_threshold:,} reads",
        "n_in": n_in, "n_removed": n_in - len(df), "n_out": len(df),
        "note": "depth = row sum from sample_depth.parquet (Phase 0), not sample_metadata.total_bases",
    })

    total_removed = sum(f["n_removed"] for f in flow)
    assert n0 - total_removed == len(df), "flow table arithmetic must reconcile exactly"
    return df.reset_index(drop=True), flow


# ---------------------------------------------------------------------------
# Step 4: the three cohorts
# ---------------------------------------------------------------------------

def build_within_project_cohort(qc_df, review_df, min_cases, min_controls):
    """Way A: QC-passed samples in CONFIRMED projects, each project then
    required to still have >= min_cases and >= min_controls -- QC can push
    a project below the curated threshold even though it passed at
    curation time. Failing projects are dropped and listed, not silently
    kept under-powered. Returns (cohort_df, excluded_projects: DataFrame).
    """
    confirmed_projects = set(review_df.loc[review_df["bucket"] == "CONFIRMED", "project"])
    pool = qc_df[qc_df["project"].isin(confirmed_projects) & qc_df["disease_label"].notna()]

    counts = pool.groupby("project")["disease_label"].value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=["healthy", "case"], fill_value=0)
    # reindex to the FULL 53, not just those with >=1 surviving sample -- a
    # project that lost every sample to QC must still show up as excluded
    # (with 0/0), not silently vanish from the accounting.
    counts = counts.reindex(sorted(confirmed_projects), fill_value=0)
    keep_projects = counts[(counts["healthy"] >= min_controls) & (counts["case"] >= min_cases)].index

    excluded = counts.loc[counts.index.difference(keep_projects)].reset_index()
    excluded = excluded.rename(columns={"healthy": "n_controls", "case": "n_cases"})
    cohort = pool[pool["project"].isin(keep_projects)].reset_index(drop=True)
    return cohort, excluded


def build_labeled_all_cohort(qc_df):
    """Way B, as far as current label coverage reaches: every QC-passed
    sample with a non-null disease_label -- CONFIRMED + HEALTHY_ONLY
    projects (62), not yet the full 116-project catalogue. See
    notebooks/01_harmonize.ipynb's closing notes for the scope decision.
    """
    return qc_df[qc_df["disease_label"].notna()].reset_index(drop=True)


def build_healthy_baseline_cohort(qc_df):
    """QC-passed samples labelled `healthy`, in projects with zero
    `case`-labelled samples -- computed from disease_label directly (never
    from projects.csv.condition, which misdescribes its own contents:
    ANALYSIS_PLAN.md Section 4). With current label coverage this reduces
    to the 9 HEALTHY_ONLY projects, since every CONFIRMED project has both
    labels by construction -- but the computation itself doesn't hardcode
    that, so it keeps working once labeled_all is extended.
    """
    labelled = qc_df.dropna(subset=["disease_label"])
    counts = labelled.groupby("project")["disease_label"].value_counts().unstack(fill_value=0)
    no_case_projects = counts.index[counts.get("case", 0) == 0]
    baseline = labelled[
        labelled["project"].isin(no_case_projects) & (labelled["disease_label"] == "healthy")
    ]
    return baseline.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5: per-project summary
# ---------------------------------------------------------------------------

def _mode_or_null(s):
    m = s.mode(dropna=True)
    return m.iloc[0] if len(m) else None


def project_summary(qc_df):
    """n samples, n cases, n controls, amplicon, kit_family, and the
    dominant instrument/region per project (a project's samples aren't
    always all on one instrument or region, so the mode is reported, not
    assumed uniform) -- the table IMPLEMENTATION.md's Phase 5 uses to
    decide which projects enter the meta-analysis.
    """
    label_counts = (
        qc_df.groupby("project")["disease_label"].value_counts().unstack(fill_value=0)
        .reindex(columns=["healthy", "case"], fill_value=0)
        .rename(columns={"healthy": "n_controls", "case": "n_cases"})
    )
    agg = qc_df.groupby("project").agg(
        n_samples=("srr", "size"),
        amplicon=("amplicon", "first"),
        kit_family=("kit_family", "first"),
        instrument=("instrument", _mode_or_null),
        region=("region", _mode_or_null),
    )
    out = agg.join(label_counts, how="left").fillna({"n_controls": 0, "n_cases": 0})
    out[["n_controls", "n_cases"]] = out[["n_controls", "n_cases"]].astype(int)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Depth-threshold justification
# ---------------------------------------------------------------------------

def depth_richness_curve(sample_depth_df, n_bins=40):
    """Empirical depth-vs-richness relationship (mean n_nonzero taxa per
    log-spaced depth bin) from Phase 0's already-computed per-sample
    depth/richness -- a proxy for a true rarefaction curve (which would
    need per-sample subsampling simulation, not built yet), used only to
    show roughly where richness plateaus and justify the QC depth
    threshold. Not a substitute for Phase 6's proper 5k/10k/20k sensitivity
    analysis.
    """
    df = sample_depth_df.copy()
    df["depth_bin"] = pd.cut(np.log10(df["depth"].clip(lower=1)), bins=n_bins)
    curve = df.groupby("depth_bin", observed=True).agg(
        depth_mid=("depth", "median"),
        richness_mean=("n_nonzero", "mean"),
        n_samples=("depth", "size"),
    ).reset_index(drop=True)
    return curve


# ---------------------------------------------------------------------------
# Flow report
# ---------------------------------------------------------------------------

def flow_report_md(flow, cohort_sizes):
    lines = ["# Phase 2 cohort flow", "", "## QC exclusions (fixed order)", "",
             "| step | n in | n removed | n out | note |", "|---|---|---|---|---|"]
    for f in flow:
        lines.append(f"| {f['step']} | {f['n_in']:,} | {f['n_removed']:,} | {f['n_out']:,} | {f['note']} |")
    lines += ["", "## Cohorts (built from the QC-passed population)", "",
              "| cohort | n samples | n projects |", "|---|---|---|"]
    for name, (n_samples, n_projects) in cohort_sizes.items():
        lines.append(f"| {name} | {n_samples:,} | {n_projects} |")
    return "\n".join(lines)
