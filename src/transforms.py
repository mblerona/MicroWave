"""Phase 3 -- puts the taxa counts on a comparable scale, three ways, without
picking a winner. Operates on the QC-passed population from Phase 2
(`cohorts.apply_qc_filters`), not the raw 168,464-sample matrix -- these
transforms feed Phase 4/5 directly, and there is no reason to spend compute
normalising samples QC already excluded.

The wide matrices (relative abundance, rarefied counts: 4,680 taxa) are
streamed batch-by-batch from `taxa_full.parquet` and written incrementally,
the same pattern `src/io.py`'s `stream_ingest` uses -- holding the full
QC-passed population densely in memory at once (~116,657 x 4,680 floats,
~4.3 GB) is avoidable and shouldn't be risked. The narrow CLR matrix
(~420 prevalence-filtered taxa) is small enough (~168,464 x 420, well under
1 GB) to load and filter directly from `taxa_prev01.npz`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from skbio.stats import subsample_counts

from src import io as taxa_io

SAMPLE_COL = "sample"  # "{project}_{srr}", matching the taxa Parquet/npz files


def sample_keys(df):
    """Build the "{project}_{srr}" join key used by the taxa Parquet/npz
    files, from a dataframe carrying separate `project`/`srr` columns."""
    return df["project"] + "_" + df["srr"]


# ---------------------------------------------------------------------------
# Step 1: relative abundance
# ---------------------------------------------------------------------------

def write_relative_abundance(taxa_full_path, keep_keys, out_path, batch_size=5000):
    """Streams `taxa_full.parquet`, keeps only samples in `keep_keys` (a
    set of "{project}_{srr}" strings), divides each row by its own total,
    writes incrementally. Depth (row sum) is recomputed from the same
    counts being written, not read from `sample_depth.parquet`, so
    numerator and denominator can never drift apart.

    Returns {"n_rows", "depth_min", "depth_max"}.
    """
    pf = pq.ParquetFile(taxa_full_path)
    writer = None
    n_rows = 0
    depth_min, depth_max = None, None

    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        mask = df[SAMPLE_COL].isin(keep_keys)
        if not mask.any():
            continue
        df = df.loc[mask]
        taxon_cols = [c for c in df.columns if c != SAMPLE_COL]

        counts = df[taxon_cols].to_numpy(dtype=np.float64)
        row_sums = counts.sum(axis=1)
        assert (row_sums > 0).all(), \
            "a QC-passed sample has zero total count -- should be impossible after the Phase 2 depth filter"
        rel = counts / row_sums[:, None]

        out_df = pd.DataFrame(rel, columns=taxon_cols)
        out_df.insert(0, SAMPLE_COL, df[SAMPLE_COL].to_numpy())
        table = pa.Table.from_pandas(out_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)

        n_rows += len(out_df)
        batch_min, batch_max = float(row_sums.min()), float(row_sums.max())
        depth_min = batch_min if depth_min is None else min(depth_min, batch_min)
        depth_max = batch_max if depth_max is None else max(depth_max, batch_max)

    if writer is not None:
        writer.close()
    return {"n_rows": n_rows, "depth_min": depth_min, "depth_max": depth_max}


# ---------------------------------------------------------------------------
# Step 2: CLR (centred log-ratio)
# ---------------------------------------------------------------------------

def compute_clr(taxa_prev01_path, keep_keys, pseudocount=0.5, method="additive"):
    """Loads the >=1%-prevalence sparse matrix, restricts to `keep_keys`,
    and applies the centred log-ratio transform.

    The prevalence-filtered ~420 taxa are treated as their own closed
    sub-composition: each row's proportions are computed from ONLY those
    taxa's counts (not the full 4,680-taxon depth), matching
    IMPLEMENTATION.md's framing that CLR is applied "only to the
    prevalence-filtered matrix" -- the taxa outside it simply aren't part
    of this composition, not a source of unaccounted mass within it.

    method="additive" (default, `config/params.yaml`'s documented default):
    zero counts are replaced with `pseudocount` (a count, e.g. 0.5 reads)
    before computing proportions -- simple, but changes the ratios among
    the nonzero parts of a row slightly, since the row total grows by
    `pseudocount * n_zeros`.

    method="multiplicative": the documented alternative (Martin-Fernandez
    et al. 2003 style). Zero proportions are replaced with a small
    detection-limit-like value `delta = pseudocount / row_total`, and the
    nonzero proportions are scaled down by `(1 - n_zeros * delta)` so the
    row still closes to 1 -- this preserves the *ratios* among the nonzero
    parts exactly, at the cost of a slightly more involved formula. Not
    used for the default `abund_clr.parquet` output; available for Phase 6
    sensitivity analysis.

    Returns (clr_df, excluded) where `clr_df` is `sample` + one column per
    prevalence-filtered taxon, and `excluded` is a DataFrame of samples that
    had to be dropped because ALL of their reads fall outside the
    prevalence-filtered taxon set -- CLR is mathematically undefined for a
    zero-mass composition (log(0)), so these can't be silently zero-filled
    or skipped without a record. Expected to be rare (a real compendium
    example: a sample that is ~100% a single genus absent from the >=1%
    set) -- if this list is large, that is a signal worth investigating,
    not something to paper over.
    """
    mat, sample_ids, taxon_names = taxa_io.load_taxa_prev01(taxa_prev01_path)
    keep_mask = pd.Series(sample_ids).isin(keep_keys).to_numpy()

    counts = np.asarray(mat[keep_mask].todense(), dtype=np.float64)
    kept_ids = sample_ids[keep_mask]
    row_sums = counts.sum(axis=1)

    has_mass = row_sums > 0
    excluded = pd.DataFrame({SAMPLE_COL: kept_ids[~has_mass], "reason": "zero mass in prevalence-filtered taxa"})
    counts = counts[has_mass]
    kept_ids = kept_ids[has_mass]
    row_sums = row_sums[has_mass][:, None]

    if method == "additive":
        counts_adj = np.where(counts == 0, pseudocount, counts)
        proportions = counts_adj / counts_adj.sum(axis=1, keepdims=True)
    elif method == "multiplicative":
        proportions0 = counts / row_sums
        n_zero = (counts == 0).sum(axis=1, keepdims=True)
        delta = pseudocount / row_sums
        proportions = np.where(counts == 0, delta, proportions0 * (1 - n_zero * delta))
        # multiplicative replacement isn't safe for every row: if a row has few
        # reads in the prevalence-filtered set AND most of its taxa are zero,
        # n_zero * delta can exceed 1, making the scaled-down nonzero share
        # negative -- mathematically invalid, not just numerically noisy. Drop
        # those rows with a record rather than let them silently produce NaN.
        invalid = (n_zero.ravel() * delta.ravel()) >= 1
        if invalid.any():
            excluded = pd.concat([excluded, pd.DataFrame({
                SAMPLE_COL: kept_ids[invalid],
                "reason": "multiplicative replacement invalid (too few reads / too many zeros in prevalence-filtered taxa)",
            })], ignore_index=True)
            proportions = proportions[~invalid]
            kept_ids = kept_ids[~invalid]
    else:
        raise ValueError(f"unknown zero-replacement method: {method!r}")

    assert np.isfinite(proportions).all() and (proportions > 0).all(), \
        "non-positive or non-finite proportion survived zero-replacement -- investigate before taking the log"

    log_p = np.log(proportions)
    clr = log_p - log_p.mean(axis=1, keepdims=True)

    out = pd.DataFrame(clr, columns=list(taxon_names))
    out.insert(0, SAMPLE_COL, kept_ids)
    return out, excluded


# ---------------------------------------------------------------------------
# Step 3: rarefaction
# ---------------------------------------------------------------------------

def write_rarefied(taxa_full_path, keep_keys, depth, seed, out_path, batch_size=2000):
    """Streams `taxa_full.parquet`, subsamples every kept row to exactly
    `depth` reads without replacement (`skbio.stats.subsample_counts`),
    writes incrementally. A single `numpy.random.Generator` seeded once
    from `config/params.yaml`'s global seed is reused across every row (not
    re-seeded per row), so the whole run is reproducible end to end from
    one seed rather than from a derived-per-row scheme that would need its
    own justification.

    Every kept row must already have >= depth reads -- true by construction
    for the QC-passed population (Phase 2's depth filter uses the same
    `depth` value as its threshold), asserted per row rather than assumed.

    Returns {"n_rows"}.
    """
    pf = pq.ParquetFile(taxa_full_path)
    writer = None
    n_rows = 0
    rng = np.random.default_rng(seed)

    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        mask = df[SAMPLE_COL].isin(keep_keys)
        if not mask.any():
            continue
        df = df.loc[mask]
        taxon_cols = [c for c in df.columns if c != SAMPLE_COL]

        counts = df[taxon_cols].to_numpy(dtype=np.int64)
        row_totals = counts.sum(axis=1)
        assert (row_totals >= depth).all(), \
            f"a kept sample has depth < {depth} -- rarefaction depth must not exceed the QC floor"

        rarefied = np.empty_like(counts)
        for i in range(counts.shape[0]):
            rarefied[i] = subsample_counts(counts[i], depth, seed=rng)

        out_df = pd.DataFrame(rarefied, columns=taxon_cols)
        out_df.insert(0, SAMPLE_COL, df[SAMPLE_COL].to_numpy())
        table = pa.Table.from_pandas(out_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
        n_rows += len(out_df)

    if writer is not None:
        writer.close()
    return {"n_rows": n_rows}


# ---------------------------------------------------------------------------
# Verification helpers (used by the notebook, not the transforms themselves)
# ---------------------------------------------------------------------------

def parquet_row_sums(path, sample_size=None, seed=None):
    """Row sums of every taxon column in a written transform output, read
    back from disk (not from in-memory arrays) -- used to verify the
    acceptance criteria against what was actually persisted, not against
    what was computed right before writing."""
    pf = pq.ParquetFile(path)
    sums = []
    for batch in pf.iter_batches(batch_size=5000):
        df = batch.to_pandas()
        taxon_cols = [c for c in df.columns if c != SAMPLE_COL]
        sums.append(df[taxon_cols].sum(axis=1).to_numpy())
    sums = np.concatenate(sums)
    if sample_size is not None and sample_size < len(sums):
        rng = np.random.default_rng(seed)
        sums = rng.choice(sums, size=sample_size, replace=False)
    return sums
