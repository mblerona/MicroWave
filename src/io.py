"""Converts the raw taxa count CSV into Parquet.

Streams `Data/raw_taxa_110.csv` (1.6 GB, 168,464 x 4,680) into Parquet
without ever holding the full matrix in memory, and derives the
all-zero-dropped and prevalence-filtered artifacts from the resulting
Parquet file rather than re-reading the CSV.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

TAXON_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
SAMPLE_COL = "sample"


def parse_taxon_header(taxon_cols):
    """Split period-separated taxonomy strings into a taxon table.

    Column headers are exactly 6 ranks (kingdom..genus); some rank
    names themselves contain spaces, brackets or "Incertae Sedis",
    so the split is bounded to exactly 6 fields rather than splitting
    on every '.'.
    """
    rows = []
    for i, full in enumerate(taxon_cols):
        parts = full.split(".", maxsplit=5)
        if len(parts) != 6:
            raise ValueError(
                f"taxon header col {i} does not split into 6 ranks "
                f"(got {len(parts)}): {full!r}"
            )
        rows.append((i, full, *parts))
    return pd.DataFrame(rows, columns=["col_index", "full_string", *TAXON_RANKS])


def read_taxa_header(csv_path):
    """Read only the header row and return the 4,680 taxon column names."""
    header = pd.read_csv(csv_path, nrows=0, index_col=0)
    cols = list(header.columns)
    if not cols or cols[0] != SAMPLE_COL:
        raise ValueError(f"expected first column {SAMPLE_COL!r}, got {cols[:1]!r}")
    taxon_cols = cols[1:]
    return taxon_cols


def stream_ingest(csv_path, out_path, chunksize=2000, row_group_size=5000,
                   compression="snappy"):
    """Stream the raw taxa CSV into a single Parquet file, one pass.

    Reads in `chunksize`-row pandas chunks (bounding peak memory),
    buffers them up to `row_group_size` rows before each Parquet
    write, and accumulates per-column non-zero counts / totals and
    per-row depth along the way so a second full scan is never needed.

    Returns a dict with taxon_cols, depth_df, col_nonzero, col_total,
    n_rows and checksum (sum of all counts, for verifying against the
    written Parquet file).
    """
    taxon_cols = read_taxa_header(csv_path)
    n_taxa = len(taxon_cols)
    dtype = {c: np.int32 for c in taxon_cols}
    dtype[SAMPLE_COL] = str

    col_nonzero = np.zeros(n_taxa, dtype=np.int64)
    col_total = np.zeros(n_taxa, dtype=np.int64)
    depth_records = []
    checksum = 0
    n_rows = 0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    buffer = []
    buffered_rows = 0

    def flush(schema):
        nonlocal writer, buffer, buffered_rows
        if not buffer:
            return
        block = pd.concat(buffer, ignore_index=True)
        table = pa.Table.from_pandas(block, schema=schema, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression=compression)
        writer.write_table(table)
        buffer = []
        buffered_rows = 0

    schema = None
    reader = pd.read_csv(csv_path, index_col=0, dtype=dtype, chunksize=chunksize)
    for chunk in reader:
        chunk = chunk[[SAMPLE_COL, *taxon_cols]].reset_index(drop=True)
        counts = chunk[taxon_cols].to_numpy(dtype=np.int32)

        row_sums = counts.sum(axis=1, dtype=np.int64)
        n_nonzero = (counts != 0).sum(axis=1)
        col_nonzero += (counts != 0).sum(axis=0)
        col_total += counts.sum(axis=0, dtype=np.int64)
        checksum += int(counts.sum(dtype=np.int64))
        n_rows += len(chunk)

        depth_records.append(pd.DataFrame({
            SAMPLE_COL: chunk[SAMPLE_COL].to_numpy(),
            "depth": row_sums,
            "n_nonzero": n_nonzero,
        }))

        if schema is None:
            schema = pa.Table.from_pandas(chunk, preserve_index=False).schema
        buffer.append(chunk)
        buffered_rows += len(chunk)
        if buffered_rows >= row_group_size:
            flush(schema)

    flush(schema)
    if writer is not None:
        writer.close()

    depth_df = pd.concat(depth_records, ignore_index=True)
    return {
        "taxon_cols": taxon_cols,
        "depth_df": depth_df,
        "col_nonzero": col_nonzero,
        "col_total": col_total,
        "n_rows": n_rows,
        "checksum": checksum,
    }


def parquet_checksum(parquet_path, taxon_cols):
    """Sum every count in a taxa Parquet file, column by column, without
    loading the whole matrix at once. Used to verify against the
    streaming checksum from `stream_ingest`."""
    pf = pq.ParquetFile(parquet_path)
    total = 0
    for col in taxon_cols:
        total += int(pf.read(columns=[col]).column(col).to_numpy(zero_copy_only=False).sum(dtype=np.int64))
    return total


def iter_row_sums(parquet_path, taxon_cols, batch_size=5000):
    """Per-row sum across `taxon_cols` in a taxa Parquet file, computed in
    batches so the file is never fully materialized in memory. Used to
    check that dropping all-zero columns doesn't change any row's depth."""
    pf = pq.ParquetFile(parquet_path)
    samples, sums = [], []
    for batch in pf.iter_batches(batch_size=batch_size, columns=[SAMPLE_COL, *taxon_cols]):
        df = batch.to_pandas()
        samples.append(df[SAMPLE_COL].to_numpy())
        sums.append(df[taxon_cols].to_numpy(dtype=np.int64).sum(axis=1))
    return pd.DataFrame({SAMPLE_COL: np.concatenate(samples), "depth": np.concatenate(sums)})


def write_taxon_table(taxon_cols, out_path):
    taxon_df = parse_taxon_header(taxon_cols)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    taxon_df.to_parquet(out_path, index=False)
    return taxon_df


def write_sample_depth(depth_df, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    depth_df.to_parquet(out_path, index=False)


def write_derived_artifacts(taxa_full_path, taxon_cols, col_nonzero, col_total,
                             n_rows, out_dir, prevalence_filter=0.01,
                             batch_size=2048, row_group_size=5000):
    """Derive taxa_nonzero.parquet and taxa_prev01.npz from the already-written
    taxa_full.parquet (columnar, so selecting a column subset is cheap) —
    the CSV itself is never read a second time.

    Both outputs are built by scanning taxa_full.parquet a slice of rows at a
    time, so the full matrix is never held in memory at once. taxa_nonzero is
    written in `row_group_size`-row groups (matching taxa_full, so it stays
    well compressed); the prevalence scan uses the smaller `batch_size`
    because it densifies each slice to pull out the non-zero cells.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    taxon_cols = np.asarray(taxon_cols)
    nonzero_mask = col_total > 0
    prevalence_mask = (col_nonzero / n_rows) >= prevalence_filter

    keep_nonzero = taxon_cols[nonzero_mask].tolist()
    keep_prev = taxon_cols[prevalence_mask].tolist()
    keep_prev_col_index = np.nonzero(prevalence_mask)[0]

    pf = pq.ParquetFile(taxa_full_path)

    # taxa_nonzero.parquet — all-zero columns dropped, streamed one row group
    # at a time so the wide matrix is never fully resident.
    nz_path = out_dir / "taxa_nonzero.parquet"
    nz_writer = None
    for batch in pf.iter_batches(batch_size=row_group_size, columns=[SAMPLE_COL, *keep_nonzero]):
        table = pa.Table.from_batches([batch])
        if nz_writer is None:
            nz_writer = pq.ParquetWriter(nz_path, table.schema, compression="snappy")
        nz_writer.write_table(table)
    if nz_writer is not None:
        nz_writer.close()

    # taxa_prev01.npz — sparse CSR of the >=prevalence_filter taxa, accumulated
    # as COO triplets over the same batched scan, plus the sample ids and the
    # original taxon column indices bundled in the same file so the matrix
    # stays self-describing.
    n_prev = len(keep_prev)
    sample_chunks, row_idx, col_idx, values = [], [], [], []
    offset = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=[SAMPLE_COL, *keep_prev]):
        sample_chunks.append(batch.column(0).to_numpy(zero_copy_only=False))
        block = np.empty((batch.num_rows, n_prev), dtype=np.int32)
        for j in range(n_prev):
            block[:, j] = batch.column(j + 1).to_numpy(zero_copy_only=False)
        r, c = np.nonzero(block)
        row_idx.append((r + offset).astype(np.int32, copy=False))
        col_idx.append(c.astype(np.int32, copy=False))
        values.append(block[r, c])
        offset += batch.num_rows

    sample_ids = (np.concatenate(sample_chunks) if sample_chunks
                  else np.array([], dtype=object))
    mat = sparse.csr_matrix(
        (
            np.concatenate(values) if values else np.zeros(0, dtype=np.int32),
            (
                np.concatenate(row_idx) if row_idx else np.zeros(0, dtype=np.int64),
                np.concatenate(col_idx) if col_idx else np.zeros(0, dtype=np.int64),
            ),
        ),
        shape=(n_rows, n_prev),
    )

    npz_path = out_dir / "taxa_prev01.npz"
    np.savez(
        npz_path,
        data=mat.data, indices=mat.indices, indptr=mat.indptr,
        shape=np.array(mat.shape),
        sample=sample_ids,
        taxon=np.array(keep_prev, dtype=object),
        taxon_col_index=keep_prev_col_index,
    )

    return {
        "keep_nonzero": keep_nonzero,
        "keep_prev": keep_prev,
        "n_nonzero_cols": len(keep_nonzero),
        "n_prev_cols": len(keep_prev),
        "n_dropped_all_zero": int((~nonzero_mask).sum()),
    }


def load_taxa_prev01(npz_path):
    """Load the sparse CSR matrix + aligned sample/taxon index written by
    `write_derived_artifacts`."""
    npz = np.load(npz_path, allow_pickle=True)
    mat = sparse.csr_matrix(
        (npz["data"], npz["indices"], npz["indptr"]),
        shape=tuple(npz["shape"]),
    )
    return mat, npz["sample"], npz["taxon"]
