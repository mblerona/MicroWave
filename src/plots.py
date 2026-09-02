"""Plotting helpers for the exploratory-analysis notebook.

Each ``plot_*`` function takes already-loaded data (a DataFrame, a Series, or
a path for the large count matrices) and returns a Matplotlib ``Figure``. The
notebook loads the data, calls these, saves the figures under
``reports/figures/`` and shows them inline.

A few ``*_data`` style helpers (``taxon_prevalence``, ``mean_relative_abundance``)
do the heavier work of scanning the 100k+ row count matrices one batch of rows
at a time, so nothing here ever loads a full matrix into memory.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SAMPLE_COL = "sample"

# muted categorical palette, reused across every figure for a consistent look
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
ACCENT = "#C44E52"


def apply_style():
    """Set a clean, consistent look for every figure. Call once in the notebook."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    })


def _commas(ax, axis="y"):
    """Format an axis with thousands separators."""
    fmt = FuncFormatter(lambda v, _: f"{int(v):,}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _abs_commas(ax, axis="x"):
    """Format an axis as |value| with thousands separators (for diverging bars)."""
    fmt = FuncFormatter(lambda v, _: f"{abs(int(v)):,}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------
# The count matrix
# ---------------------------------------------------------------------------

def plot_taxa_per_sample(depth_df):
    """Histogram of how many distinct genera each sample contains. A stool
    sample carries only a few dozen of the 4,680 genera in the table, so the
    matrix is overwhelmingly zeros."""
    n = depth_df["n_nonzero"].to_numpy()
    med = float(np.median(n))
    hi = np.percentile(n, 99.5)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(n[n <= hi], bins=np.arange(0, hi + 8, 8), color=PALETTE[0])
    ax.axvline(med, color=ACCENT, ls="--", lw=1.5, label=f"median = {med:.0f}")
    ax.set_xlabel("distinct genera present in a sample")
    ax.set_ylabel("samples")
    ax.set_title("Each sample carries only a few dozen genera")
    ax.legend()
    _commas(ax)
    fig.tight_layout()
    return fig


def taxon_prevalence(taxa_full_path, n_rows, batch_size=2048):
    """Fraction of samples in which each genus column is non-zero, scanned
    from the full count matrix one batch of rows at a time."""
    pf = pq.ParquetFile(taxa_full_path)
    names = [c for c in pf.schema.names if c != SAMPLE_COL]
    counts = np.zeros(len(names), dtype=np.int64)
    for batch in pf.iter_batches(batch_size=batch_size, columns=names):
        block = np.empty((batch.num_rows, len(names)), dtype=np.int32)
        for i in range(len(names)):
            block[:, i] = batch.column(i).to_numpy(zero_copy_only=False)
        counts += (block != 0).sum(axis=0)
    return pd.Series(counts / n_rows, index=names, name="prevalence")


def plot_prevalence_curve(prevalence, thresholds=(0.5, 0.1, 0.01)):
    """Genera ranked from most to least prevalent, on a log scale. A tiny core
    of genera is in almost every sample; the large majority are rare. The
    dotted lines are prevalence cut-offs the pipeline can use to trim the
    matrix before modelling."""
    p = np.sort(prevalence.to_numpy())[::-1]
    rank = np.arange(1, len(p) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rank, p, color=PALETTE[0], lw=1.6)
    ax.set_yscale("log")
    for t in thresholds:
        k = int((prevalence >= t).sum())
        ax.axhline(t, color=ACCENT, ls=":", lw=1)
        ax.text(1, t, f" ≥{t:.0%} of samples: {k} genera",
                va="bottom", ha="left", fontsize=8, color=ACCENT)
    ax.set_xlabel("genus rank (most prevalent first)")
    ax.set_ylabel("fraction of samples containing the genus")
    ax.set_title("A small core of genera, a long tail of rare ones")
    fig.tight_layout()
    return fig


def plot_kingdom_split(taxon_df):
    """How the 4,680 table columns break down by kingdom (log scale). The
    table is essentially a bacterial census with a small set of Archaea."""
    vc = (taxon_df["kingdom"].replace("NA", "unresolved")
          .value_counts().sort_values())
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(vc.index, vc.to_numpy(), color=PALETTE[2])
    for i, v in enumerate(vc.to_numpy()):
        ax.text(v, i, f" {v:,}", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("genus columns (log scale)")
    ax.set_title("Table composition by kingdom")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sequencing depth
# ---------------------------------------------------------------------------

def plot_depth_distribution(depth_df, threshold):
    """Reads per sample on a log scale, with the quality-control floor marked.
    Depth spans about seven orders of magnitude; samples left of the line are
    too shallow to trust and get dropped."""
    d = depth_df["depth"].to_numpy()
    d = d[d > 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d, bins=np.logspace(0, np.log10(d.max()), 60), color=PALETTE[0])
    ax.set_xscale("log")
    ax.axvline(threshold, color=ACCENT, ls="--", lw=1.5,
               label=f"floor = {threshold:,} reads")
    n_below = int((depth_df["depth"] < threshold).sum())
    ax.set_xlabel("reads per sample (log scale)")
    ax.set_ylabel("samples")
    ax.set_title(f"Sequencing depth — {n_below:,} samples below the floor")
    ax.legend()
    _commas(ax)
    fig.tight_layout()
    return fig


def plot_depth_vs_richness(depth_df, threshold):
    """Observed genus count against read depth (2-D density). Richness climbs
    with depth and then flattens, so depth is a confounder for any richness
    comparison and motivates a minimum-depth cut-off."""
    d = depth_df["depth"].to_numpy()
    r = depth_df["n_nonzero"].to_numpy()
    m = d > 0
    fig, ax = plt.subplots(figsize=(7, 4.2))
    hb = ax.hexbin(d[m], r[m], xscale="log", gridsize=45, cmap="viridis", mincnt=1)
    ax.axvline(threshold, color=ACCENT, ls="--", lw=1.5, label=f"floor = {threshold:,}")
    fig.colorbar(hb, ax=ax, label="samples")
    ax.set_xlabel("reads per sample (log scale)")
    ax.set_ylabel("distinct genera observed")
    ax.set_title("Deeper sequencing finds more genera, then plateaus")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metadata landscape
# ---------------------------------------------------------------------------

def plot_field_coverage(tag_summary_df, top=30):
    """The metadata fields that cover the most samples. Thousands of field
    names exist, but only a few dozen are populated for a large share of
    samples — those are the ones worth harmonising."""
    d = tag_summary_df.nlargest(top, "n_samples").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, top * 0.26)))
    ax.barh(d["tag"], d["n_samples"], color=PALETTE[0])
    ax.set_xlabel("samples with the field populated")
    ax.set_title(f"Top {top} metadata fields by coverage")
    _commas(ax, "x")
    fig.tight_layout()
    return fig


def plot_samples_per_project(project_sizes):
    """How big the contributing studies are (log scale). A few large cohorts,
    many small ones — so a pooled analysis risks being dominated by a
    handful of studies unless that is controlled for."""
    s = np.sort(project_sizes.to_numpy())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s, bins=np.logspace(np.log10(s.min()), np.log10(s.max()), 40),
            color=PALETTE[4])
    ax.set_xscale("log")
    ax.set_xlabel("samples in a study (log scale)")
    ax.set_ylabel("studies")
    ax.set_title(f"{len(s)} studies — median {np.median(s):.0f}, max {s.max():,}")
    fig.tight_layout()
    return fig


def plot_geography(sample_meta_df):
    """Where the samples come from. The compendium leans heavily toward Europe
    and North America, with a large 'unknown' slice — a limit on any claim
    about 'the human gut' in general."""
    reg = (sample_meta_df["region"].fillna("unknown")
           .replace("", "unknown").value_counts().sort_values())
    total = len(sample_meta_df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(reg.index, reg.to_numpy(), color=PALETTE[0])
    for i, v in enumerate(reg.to_numpy()):
        ax.text(v, i, f" {v / total:.0%}", va="center", fontsize=8)
    ax.set_xlabel("samples")
    ax.set_title("Samples by world region")
    _commas(ax, "x")
    fig.tight_layout()
    return fig


def plot_technical_factors(sample_meta_df, projects_df):
    """Two sources of lab-to-lab variation: which 16S region was sequenced
    (decided per study) and which instrument produced the reads (per sample).
    Both are uneven, which is why they are modelled as covariates later."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    amp = (projects_df["amplicon"].replace("", "unknown").fillna("unknown")
           .value_counts().nlargest(8).iloc[::-1])
    axes[0].barh(amp.index, amp.to_numpy(), color=PALETTE[2])
    axes[0].set_title("amplicon region (per study)")
    axes[0].set_xlabel("studies")
    ins = (sample_meta_df["instrument"].fillna("unknown")
           .value_counts().nlargest(8).iloc[::-1])
    axes[1].barh(ins.index, ins.to_numpy(), color=PALETTE[1])
    axes[1].set_title("sequencing instrument (per sample)")
    axes[1].set_xlabel("samples")
    _commas(axes[1], "x")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Harmonised variables
# ---------------------------------------------------------------------------

def plot_harmonized_coverage(harmonized_df, concepts):
    """How complete each cleaned variable is. Technical fields are effectively
    total; the biological ones (age, sex, BMI, disease label) exist for only a
    minority of samples, which bounds what the analysis can do."""
    cov = [(c, harmonized_df[c].notna().mean()) for c in concepts if c in harmonized_df]
    cov.sort(key=lambda kv: kv[1])
    labels, vals = zip(*cov)
    fig, ax = plt.subplots(figsize=(7, max(4, len(labels) * 0.34)))
    bars = ax.barh(list(labels), [v * 100 for v in vals], color=PALETTE[0])
    for b, v in zip(bars, vals):
        ax.text(v * 100, b.get_y() + b.get_height() / 2, f" {v:.1%}",
                va="center", fontsize=8)
    ax.set_xlim(0, 108)
    ax.set_xlabel("% of samples with a value")
    ax.set_title("Coverage of the harmonised variables")
    fig.tight_layout()
    return fig


def plot_comissingness(harmonized_df, concepts):
    """Do variables tend to be missing together? Each cell is the correlation
    between 'this column is missing' and 'that column is missing'. Blocks of
    high correlation mean the gaps are structural — whole studies skipped a
    group of fields — rather than random."""
    cols = [c for c in concepts if c in harmonized_df]
    miss = harmonized_df[cols].isna().astype(int)
    keep = [c for c in cols if 0 < int(miss[c].sum()) < len(miss)]
    corr = miss[keep].corr()
    fig, ax = plt.subplots(figsize=(1.05 * len(keep) + 2, 0.95 * len(keep) + 1.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels(keep, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(keep, fontsize=8)
    for i in range(len(keep)):
        for j in range(len(keep)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > 0.6 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="missingness correlation", fraction=0.046, pad=0.04)
    ax.set_title("Which variables go missing together")
    fig.tight_layout()
    return fig


def plot_age_distribution(harmonized_df):
    """Parsed age, stacked by how the value was obtained. 'inferred' means the
    unit (years vs months) was deduced from the study population rather than
    stated outright — a caveat for any age-based analysis."""
    d = harmonized_df.dropna(subset=["age_years"])
    seen = set(d["age_confidence"])
    order = [k for k in ["exact", "range", "bound", "inferred"] if k in seen]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([d.loc[d["age_confidence"] == k, "age_years"] for k in order],
            bins=np.arange(0, 101, 3), stacked=True, label=order,
            color=PALETTE[:len(order)])
    ax.set_xlabel("age (years)")
    ax.set_ylabel("samples")
    ax.set_title("Parsed age, by how it was derived")
    ax.legend()
    _commas(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# QC funnel and cohorts
# ---------------------------------------------------------------------------

def plot_qc_funnel(flow, cohort_sizes):
    """Left: the route from every sample in the compendium down to the working
    set — each bar is what survives the next exclusion. Right: the labelled
    cohorts carved out of the survivors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3),
                                   gridspec_kw={"width_ratios": [3, 2]})

    def short(step):
        return (step.replace("exclude ", "drop ")
                    .replace(" host_species", "").replace(" reads", "")
                    .replace(" / missing sample_type", ""))

    labels = ["all samples"] + [short(f["step"]) for f in flow]
    values = [flow[0]["n_in"]] + [f["n_out"] for f in flow]
    y = list(range(len(values)))[::-1]
    ax1.barh(y, values, color=PALETTE[0])
    for yi, v in zip(y, values):
        ax1.text(v, yi, f" {v:,}", va="center", fontsize=8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("samples remaining")
    ax1.set_title("QC exclusions, in order")
    _commas(ax1, "x")

    items = list(cohort_sizes.items())[::-1]
    ax2.barh(range(len(items)), [v[0] for _, v in items], color=PALETTE[3])
    for i, (_, v) in enumerate(items):
        ax2.text(v[0], i, f" {v[0]:,}  ({v[1]} studies)", va="center", fontsize=8)
    ax2.set_yticks(range(len(items)))
    ax2.set_yticklabels([k for k, _ in items], fontsize=9)
    ax2.set_xlabel("samples")
    ax2.set_title("Labelled cohorts")
    _commas(ax2, "x")

    fig.suptitle("From the whole compendium to the analysis cohorts",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def plot_project_case_control(project_summary_df, min_per_side=10):
    """For every study that has both patients and healthy controls: controls
    to the left, cases to the right, studies sorted by size. The dashed lines
    are the minimum each side needs for a within-study comparison — studies
    that fall inside them are under-powered."""
    d = project_summary_df[(project_summary_df["n_controls"] > 0)
                           & (project_summary_df["n_cases"] > 0)].copy()
    d["total"] = d["n_controls"] + d["n_cases"]
    d = d.sort_values("total")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, max(4, len(d) * 0.26)))
    ax.barh(y, -d["n_controls"].to_numpy(), color=PALETTE[0], label="controls")
    ax.barh(y, d["n_cases"].to_numpy(), color=ACCENT, label="cases")
    ax.axvline(-min_per_side, color="black", ls="--", lw=0.8)
    ax.axvline(min_per_side, color="black", ls="--", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["project"], fontsize=7)
    ax.set_xlabel("← controls          samples          cases →")
    ax.set_title("Case / control balance per study")
    ax.legend(loc="lower right")
    _abs_commas(ax, "x")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Abundance transforms
# ---------------------------------------------------------------------------

def mean_relative_abundance(abund_relative_path, batch_size=2048):
    """Mean relative abundance of every genus across the samples in
    abund_relative.parquet, accumulated one batch of rows at a time."""
    pf = pq.ParquetFile(abund_relative_path)
    names = [c for c in pf.schema.names if c != SAMPLE_COL]
    total = np.zeros(len(names), dtype=np.float64)
    n = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=names):
        block = np.empty((batch.num_rows, len(names)), dtype=np.float64)
        for i in range(len(names)):
            block[:, i] = batch.column(i).to_numpy(zero_copy_only=False)
        total += block.sum(axis=0)
        n += batch.num_rows
    return pd.Series(total / n, index=names, name="mean_relative_abundance")


def _genus_label(full_string):
    """Last non-'NA' rank of a period-separated taxonomy string."""
    parts = full_string.split(".")
    for part in reversed(parts):
        if part != "NA":
            return part
    return full_string


def plot_top_genera(mean_rel, top=15):
    """The genera that make up most of the average gut community. A dozen or so
    genera account for the bulk of the reads; everything else is a thin tail."""
    s = mean_rel.sort_values(ascending=False).head(top).iloc[::-1]
    labels = [_genus_label(name) for name in s.index]
    fig, ax = plt.subplots(figsize=(7, max(4, top * 0.32)))
    ax.barh(labels, s.to_numpy() * 100, color=PALETTE[2])
    for i, v in enumerate(s.to_numpy() * 100):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("mean relative abundance (%)")
    ax.set_title(f"The {top} most abundant genera on average")
    fig.tight_layout()
    return fig


def plot_clr_vs_relative(rel_path, clr_path, genera_full_strings):
    """The same few genera before and after the centred-log-ratio transform.
    Relative abundance (left) is bounded, spiky and zero-inflated; CLR (right)
    spreads it into something roughly bell-shaped that linear models can use."""
    rel = pq.read_table(rel_path, columns=list(genera_full_strings)).to_pandas()
    clr_cols = set(pq.ParquetFile(clr_path).schema.names)
    have_clr = [g for g in genera_full_strings if g in clr_cols]
    clr = pq.read_table(clr_path, columns=have_clr).to_pandas() if have_clr else pd.DataFrame()

    nrows = len(genera_full_strings)
    fig, axes = plt.subplots(nrows, 2, figsize=(9, 2.3 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, 2)
    for row, g in enumerate(genera_full_strings):
        name = _genus_label(g)
        a0, a1 = axes[row]
        v = rel[g].to_numpy()
        v = v[v > 0]
        a0.hist(np.log10(v), bins=40, color=PALETTE[0])
        a0.set_title(f"{name} — log₁₀ relative abundance (non-zero)", fontsize=9)
        a0.set_ylabel("samples")
        if g in clr.columns:
            a1.hist(clr[g].to_numpy(), bins=40, color=PALETTE[2])
        else:
            a1.text(0.5, 0.5, "not in CLR matrix\n(below prevalence filter)",
                    ha="center", va="center", transform=a1.transAxes, fontsize=8)
        a1.set_title(f"{name} — CLR", fontsize=9)
    fig.tight_layout()
    return fig
