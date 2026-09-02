"""Phase 4 -- Goal 2: how much of the between-sample variation is the lab
rather than the person.

IMPLEMENTATION.md Phase 4 runs *before* Phase 5 on purpose: its answer
calibrates how much weight the disease findings can bear. Five pieces, all
driven from this module:

1. **Distances** -- Bray-Curtis on relative abundance, Aitchison (Euclidean
   on CLR) on the prevalence-filtered matrix. A full 116k x 116k dense
   distance matrix is ~100 GB, so every distance-based step runs on a
   project-stratified subsample (`config/params.yaml` -> `variance.distance_subsample_n`),
   stated wherever a number is reported.
2. **Ordination** -- PCoA (via `skbio`) and UMAP, each coloured by every
   technical factor and by `disease_label`. The `project`-coloured PCoA is
   the headline figure.
3. **Variance partitioning** -- marginal PERMANOVA per factor
   (`skbio.stats.distance.permanova`, >=999 permutations) plus an
   adonis-style **sequential** decomposition run both technical-factors-first
   and disease-first, because sequential R^2 depends on term order and
   hiding that would mislead.
4. **Negative control -- project predictability.** A PyTorch classifier
   (multinomial logistic regression and an MLP) trained to recover a
   sample's `project` from its composition under stratified CV. High
   accuracy is the *expected* result: it shows the lab signature is
   low-dimensional and trivially learnable -- exactly what a naive disease
   model would latch onto.
5. **Depth vs alpha diversity** -- observed richness rises with sequencing
   depth; quantified here (raw vs rarefied) so Phase 5 controls for it
   rather than assuming it away.

Plain functions, dataframes / paths in and results out, same shape as the
rest of `src/`. The wide count matrices are streamed from Parquet a batch of
rows at a time; nothing here loads a full 116k x 4,680 matrix densely.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from scipy.spatial.distance import pdist, squareform

SAMPLE_COL = "sample"  # "{project}_{srr}", matching the taxa Parquet / npz files

# Technical factors whose share of between-sample variation Phase 4 quantifies.
# `project` is the batch itself; the others are the knobs a lab turns.
TECHNICAL_FACTORS = [
    "project", "amplicon", "kit_family", "instrument", "region",
    "depth_bin", "library_source", "bead_beating",
]
BIOLOGICAL_FACTOR = "disease_label"


# ---------------------------------------------------------------------------
# Sample keys, covariate prep, project-stratified subsampling
# ---------------------------------------------------------------------------

def sample_keys(df):
    """Build the "{project}_{srr}" join key the taxa Parquet / npz files use,
    from a dataframe carrying separate `project` / `srr` columns."""
    return df["project"].astype(str) + "_" + df["srr"].astype(str)


def prepare_covariates(df, n_depth_bins=5):
    """Return a copy of `df` with `sample` set and the derived covariates
    Phase 4 partitions on: `depth_bin` (quantile bins of read depth, as an
    ordered category) and `bead_beating` / `library_source` with their blanks
    made explicit rather than left as NaN (a blank *is* a level here -- "the
    protocol wasn't reported" -- not missing-at-random)."""
    out = df.copy()
    out[SAMPLE_COL] = sample_keys(out)

    if "depth" in out.columns:
        # labelled quantile bins; drop_duplicates guards against ties at the
        # edges collapsing a bin (won't happen on this depth spread, asserted
        # implicitly by qcut succeeding).
        out["depth_bin"] = pd.qcut(
            out["depth"], q=n_depth_bins,
            labels=[f"depth_q{i+1}" for i in range(n_depth_bins)],
        ).astype(str)

    for col in ("bead_beating", "library_source", "amplicon", "kit_family",
                "instrument", "region"):
        if col in out.columns:
            out[col] = (
                out[col].astype("string")
                .str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
                .fillna("unknown")
                .astype(str)
            )
    return out


def project_stratified_subsample(df, n, seed, project_col="project"):
    """Down-sample to ~`n` rows while keeping every project's *share* of the
    data roughly intact (and every project that had >=1 row still represented).
    Returns the full frame unchanged when `n` >= len(df).

    A single distance matrix over the whole QC-passed population is far too
    large to hold densely; IMPLEMENTATION.md Phase 4 step 1 calls for exactly
    this ("subsample (project-stratified, e.g. 10-15k)").
    """
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in df.groupby(project_col, sort=True):
        k = max(1, int(round(len(grp) * frac)))
        k = min(k, len(grp))
        parts.append(grp.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1))))
    out = pd.concat(parts)
    out = out.sample(frac=1.0, random_state=seed)  # shuffle so row order carries no project structure
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Streamed loading of an abundance subset, aligned to a key order
# ---------------------------------------------------------------------------

def load_abundance_subset(parquet_path, keys, batch_size=5000, dtype=np.float32):
    """Read only the rows whose `sample` is in `keys` from an abundance
    Parquet file, streamed a batch at a time so the wide matrix is never
    fully resident. Returns (ids, matrix, taxon_cols) with `ids` a plain
    object array and `matrix` `dtype` (n_found x n_taxa), in file order.
    float32 by default -- `pdist` upcasts per pair internally, so keeping the
    resident copy small costs no distance precision.

    `keys` that aren't present in the file (e.g. the one CLR-excluded
    zero-mass sample) simply don't come back -- the caller re-aligns.
    """
    pf = pq.ParquetFile(parquet_path)
    taxon_cols = [c for c in pf.schema.names if c != SAMPLE_COL]
    want = set(keys)
    ids, blocks = [], []
    for batch in pf.iter_batches(batch_size=batch_size):
        d = batch.to_pandas()
        mask = d[SAMPLE_COL].isin(want)
        if not mask.any():
            continue
        d = d.loc[mask]
        ids.append(d[SAMPLE_COL].to_numpy())
        blocks.append(d[taxon_cols].to_numpy(dtype=dtype))
    if not blocks:
        return np.array([], dtype=object), np.empty((0, len(taxon_cols)), dtype=dtype), taxon_cols
    return np.concatenate(ids), np.vstack(blocks), taxon_cols


def align(ids, matrix, order_keys):
    """Reorder (ids, matrix) to match `order_keys`, dropping any key not
    present in `ids`. Returns (kept_keys, reordered_matrix)."""
    pos = {k: i for i, k in enumerate(ids)}
    idx = [pos[k] for k in order_keys if k in pos]
    kept = [k for k in order_keys if k in pos]
    return np.asarray(kept, dtype=object), matrix[idx]


# ---------------------------------------------------------------------------
# Step 1: distances
# ---------------------------------------------------------------------------

def bray_curtis_matrix(relative_matrix):
    """Dense Bray-Curtis dissimilarity (n x n) from a relative-abundance
    matrix. `pdist` is C-level and upcasts internally; the squareform result
    is float64."""
    return squareform(pdist(relative_matrix, metric="braycurtis"))


def aitchison_matrix(clr_matrix):
    """Aitchison distance == Euclidean distance on CLR coordinates
    (IMPLEMENTATION.md Phase 4 step 1)."""
    return squareform(pdist(clr_matrix, metric="euclidean"))


def gower_trace(dist):
    """Total sum of squares of a Gower-centred distance matrix,
    tr(G) = sum(D^2) / (2n) -- the denominator of every R^2 below, without
    forming the n x n Gower matrix."""
    d2 = np.asarray(dist, dtype=np.float64) ** 2
    n = d2.shape[0]
    return d2.sum() / (2.0 * n)


# ---------------------------------------------------------------------------
# Step 2: ordination
# ---------------------------------------------------------------------------

def pcoa_coords(dist, ids, n_dims=3, seed=42):
    """PCoA via `skbio` (truncated SVD -- fast and memory-frugal at this
    sample count). Returns (coords_df, proportion_explained) where
    `coords_df` is `sample` + PCo1..PCo{n_dims}."""
    from skbio import DistanceMatrix
    from skbio.stats.ordination import pcoa

    dm = DistanceMatrix(np.asarray(dist, dtype=np.float64),
                        ids=[str(s) for s in ids])
    res = pcoa(dm, method="fsvd", dimensions=n_dims, seed=seed)
    coords = res.samples.iloc[:, :n_dims].copy()
    coords.columns = [f"PCo{i+1}" for i in range(n_dims)]
    coords.insert(0, SAMPLE_COL, list(ids))
    prop = np.asarray(res.proportion_explained)[:n_dims]
    return coords.reset_index(drop=True), prop


def umap_coords(matrix, ids, seed, n_neighbors=30, min_dist=0.1, metric="euclidean"):
    """2-D UMAP embedding of an abundance / CLR matrix. `metric='braycurtis'`
    on a relative-abundance matrix mirrors the Bray-Curtis PCoA; the default
    Euclidean on CLR mirrors the Aitchison PCoA."""
    import umap

    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, random_state=seed,
    )
    emb = reducer.fit_transform(matrix)
    return pd.DataFrame({SAMPLE_COL: list(ids), "UMAP1": emb[:, 0], "UMAP2": emb[:, 1]})


# ---------------------------------------------------------------------------
# Step 3a: marginal PERMANOVA, one factor at a time
# ---------------------------------------------------------------------------

def _r2_from_pseudo_f(f_stat, n, n_groups):
    """PERMANOVA reports pseudo-F, not R^2. Invert the definition:
    F = (SSb/(g-1)) / (SSw/(n-g)) and R^2 = SSb/(SSb+SSw), so with
    x = F*(g-1)/(n-g), R^2 = x / (1+x)."""
    x = f_stat * (n_groups - 1) / (n - n_groups)
    return x / (1.0 + x)


def permanova_marginal(dist, ids, metadata, factors, permutations, seed,
                        distance_name):
    """Run `skbio` PERMANOVA once per factor (each factor tested on its own,
    NaN levels dropped for that factor only). Returns a tidy frame with
    pseudo-F, the recovered R^2, the permutation p-value and the permutation
    count -- the core of `reports/variance_explained.csv`.

    Memory: the n x n `dist` is wrapped in a `DistanceMatrix` **once** and
    reused across every complete factor (all technical factors are complete
    after `prepare_covariates`, and `disease_label` is complete on the
    labelled cohort). Only a factor with genuine NaNs pays for a `.filter()`
    copy. The earlier per-factor `dist[np.ix_(keep, keep)]` copy is what blew
    the memory budget with two ~0.5 GB matrices already resident.
    """
    from skbio import DistanceMatrix
    from skbio.stats.distance import permanova

    meta = metadata.set_index(SAMPLE_COL).loc[list(ids)]
    dist = np.asarray(dist, dtype=np.float64)
    str_ids = [str(s) for s in ids]
    dm_full = DistanceMatrix(dist, ids=str_ids)  # one wrap, reused below
    rows = []
    for factor in factors:
        if factor not in meta.columns:
            continue
        col = meta[factor].astype("object")
        keep = col.notna().to_numpy()
        levels = pd.unique(col[keep])
        if keep.sum() < 10 or len(levels) < 2:
            rows.append(dict(distance=distance_name, factor=factor,
                             n=int(keep.sum()), n_groups=len(levels), df=np.nan,
                             R2=np.nan, pseudo_F=np.nan, p=np.nan,
                             n_permutations=permutations, analysis="marginal",
                             note="skipped: <2 levels or <10 samples"))
            continue
        if keep.all():
            dm, grouping = dm_full, col.astype(str).to_numpy()
        else:
            sub_ids = [s for s, k in zip(str_ids, keep) if k]
            dm = dm_full.filter(sub_ids)
            grouping = col[keep].astype(str).to_numpy()
        res = permanova(dm, grouping, permutations=permutations, seed=seed)
        n, g = int(res["sample size"]), int(res["number of groups"])
        f_stat = float(res["test statistic"])
        rows.append(dict(
            distance=distance_name, factor=factor, n=n, n_groups=g, df=g - 1,
            R2=_r2_from_pseudo_f(f_stat, n, g), pseudo_F=f_stat,
            p=float(res["p-value"]), n_permutations=int(res["number of permutations"]),
            analysis="marginal", note="",
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3b: adonis-style SEQUENTIAL decomposition (distance-based RDA)
# ---------------------------------------------------------------------------

def _design(meta, terms):
    """Cumulative dummy design: intercept column, then one dummy block per
    term (first level dropped). Returns (X, spans) with spans a list of
    (term, start_col, end_col) into X, end_col being the cumulative width."""
    blocks = [np.ones((len(meta), 1))]
    spans, start = [], 1
    for term in terms:
        dummies = pd.get_dummies(meta[term].astype(str), drop_first=True,
                                 prefix=term, dtype=float).to_numpy()
        if dummies.shape[1] == 0:  # single-level term contributes nothing
            spans.append((term, start, start))
            continue
        blocks.append(dummies)
        spans.append((term, start, start + dummies.shape[1]))
        start += dummies.shape[1]
    return np.hstack(blocks), spans


def _sequential_ss(coords, X, spans):
    """Type-I (sequential) sum-of-squares partition in a real coordinate
    space `coords` (n x d) where sum(coords**2) is the total inertia. For a
    cumulative design prefix X[:, :e], the fitted inertia is
    tr((X'X)^-1 (X'Z)(X'Z)') -- each term's SS is the increment over the
    previous prefix. Returns (ss_terms, df_terms, ss_resid, df_resid, total).
    """
    total = float((coords ** 2).sum())
    ss_cum_prev, ss_terms, df_terms = 0.0, [], []
    last_e = 1
    for (_term, s, e) in spans:
        if e == s:  # empty term
            ss_terms.append(0.0)
            df_terms.append(0)
            continue
        Xc = X[:, :e]
        XtX = Xc.T @ Xc
        XtZ = Xc.T @ coords
        # pinv, not solve: a design prefix that includes a many-level factor
        # (project, 300+ dummies) on a subsample can be rank-deficient, which
        # `solve` would reject outright. tr((X'X)^+ (X'Z)(X'Z)') is the fitted
        # inertia either way.
        ss_cum = float(np.trace(np.linalg.pinv(XtX) @ (XtZ @ XtZ.T)))
        ss_terms.append(ss_cum - ss_cum_prev)
        df_terms.append(e - s)
        ss_cum_prev = ss_cum
        last_e = e
    ss_resid = total - ss_cum_prev
    df_resid = coords.shape[0] - last_e
    return np.array(ss_terms), np.array(df_terms), ss_resid, df_resid, total


def dbrda_sequential(coords, metadata, ids, terms, permutations, seed,
                      distance_name, order_name):
    """Distance-based RDA / adonis Type-I decomposition of `coords`
    (an inertia-preserving embedding of a distance) against `terms` in the
    given order. Permutation p-values come from permuting the rows of
    `coords` and recomputing each term's pseudo-F against the *full*-model
    residual, the standard adonis scheme.

    For Aitchison this is exact: `coords` = column-centred CLR, whose total
    inertia is exactly tr(G). For Bray-Curtis `coords` is the top PCoA axes,
    so `captured` (returned) reports what fraction of tr(G) the embedding
    retains -- the relative R^2 across terms and the order-dependence are
    robust to the truncation, the absolute values are read against `captured`.
    """
    meta = metadata.set_index(SAMPLE_COL).loc[list(ids)][terms].copy()
    keep = meta.notna().all(axis=1).to_numpy()
    coords = np.asarray(coords, dtype=np.float64)[keep]
    coords = coords - coords.mean(axis=0, keepdims=True)
    meta = meta.loc[keep]

    X, spans = _design(meta, terms)
    ss, df, ss_resid, df_resid, total = _sequential_ss(coords, X, spans)
    obs_F = np.where(df > 0, (ss / np.where(df > 0, df, 1)) / (ss_resid / df_resid), np.nan)

    rng = np.random.default_rng(seed)
    ge = np.zeros(len(spans))
    n = coords.shape[0]
    for _ in range(permutations):
        p = rng.permutation(n)
        ss_p, df_p, ssr_p, dfr_p, _ = _sequential_ss(coords[p], X, spans)
        F_p = np.where(df_p > 0, (ss_p / np.where(df_p > 0, df_p, 1)) / (ssr_p / dfr_p), np.nan)
        ge += np.where(np.isnan(obs_F), 0.0, F_p >= obs_F - 1e-9)
    p_vals = (ge + 1.0) / (permutations + 1.0)

    rows = []
    for i, (term, s, e) in enumerate(spans):
        rows.append(dict(
            distance=distance_name, analysis=f"sequential:{order_name}",
            term_order=i + 1, factor=term, df=int(df[i]),
            R2=ss[i] / total if total else np.nan,
            pseudo_F=obs_F[i], p=p_vals[i] if df[i] > 0 else np.nan,
            n=n, n_permutations=permutations, note="",
        ))
    rows.append(dict(
        distance=distance_name, analysis=f"sequential:{order_name}",
        term_order=len(spans) + 1, factor="Residual", df=int(df_resid),
        R2=ss_resid / total if total else np.nan, pseudo_F=np.nan, p=np.nan,
        n=n, n_permutations=permutations, note="",
    ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4: project-predictability classifier (PyTorch)
# ---------------------------------------------------------------------------

_TORCH_HINT = (
    "PyTorch is required for the Phase 4 project-predictability classifier "
    "(run_project_cv). Install it with:\n"
    "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
)

try:  # noqa: E402  -- torch is needed ONLY for run_project_cv below
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    _HAS_TORCH = True
except ModuleNotFoundError:  # keep the module importable without torch:
    _HAS_TORCH = False       # PERMANOVA / sequential / plots / depth-alpha all work

    class _NoTorch:
        """Stand-in so `import src.variance` succeeds without PyTorch. Any
        real attribute access raises with an install hint; `no_grad()` is a
        harmless pass-through so the decorator below still applies."""

        def no_grad(self):
            def _decorator(fn):
                return fn
            return _decorator

        def __getattr__(self, _name):
            raise ModuleNotFoundError(_TORCH_HINT)

    class _NoNN:
        Module = object

        def __getattr__(self, _name):
            raise ModuleNotFoundError(_TORCH_HINT)

    torch = _NoTorch()
    nn = _NoNN()
    Dataset = object
    DataLoader = None


class _CompositionDataset(Dataset):
    """Wraps a scaled composition matrix + integer project labels as tensors
    -- the same custom-`Dataset` shape used in the course notebooks."""

    def __init__(self, X, y):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ProjectLinear(nn.Module):
    """Multinomial logistic regression -- a single linear layer, softmax
    supplied by the loss. If even this recovers the originating study far
    above the majority-class rate, the lab signature is low-dimensional and
    linearly accessible (IMPLEMENTATION.md Phase 4 step 4)."""

    def __init__(self, in_features, n_classes):
        super().__init__()
        self.net = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.net(x)


class ProjectMLP(nn.Module):
    """Two hidden layers with ReLU + dropout, then a linear head -- the
    feed-forward net from the course's Neural Networks notebooks, sized for a
    ~420-feature input and a few-hundred-way output."""

    def __init__(self, in_features, n_classes, hidden=(256, 128), dropout=0.3):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_classifier(model, loader, device, epochs, lr, seed):
    """Plain supervised loop: Adam + cross-entropy, no early stopping (fixed
    epoch budget), matching the course pattern."""
    torch.manual_seed(seed)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def _predict_logits(model, loader, device):
    model.eval()
    out = [model(xb.to(device)).cpu() for xb, _ in loader]
    return torch.cat(out).numpy()


def run_project_cv(clr_matrix, project_labels, seed, *, n_splits=5, epochs=15,
                    batch_size=512, lr=1e-3, hidden=(256, 128),
                    models=("linear", "mlp"), min_project_size=25, progress=True):
    """Stratified k-fold CV of a PyTorch classifier predicting `project` from
    CLR composition. Projects with fewer than `min_project_size` samples are
    dropped (too few for a stratified split) and reported. Per fold: a
    `StandardScaler` is fit on the training rows only, both model types are
    trained from scratch, and accuracy / balanced accuracy / top-5 accuracy
    are recorded against the majority-class baseline.

    Returns (cv_df, summary) where `summary` carries n_classes, n_samples,
    majority_baseline, the dropped-project accounting and the torch device.
    """
    if not _HAS_TORCH:
        raise ModuleNotFoundError(_TORCH_HINT)

    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 top_k_accuracy_score)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    labels = np.asarray(project_labels)
    vc = pd.Series(labels).value_counts()
    big_enough = set(vc[vc >= min_project_size].index)
    keep = np.array([p in big_enough for p in labels])
    dropped = vc[vc < min_project_size]

    X = np.asarray(clr_matrix, dtype=np.float32)[keep]
    le = LabelEncoder()
    y = le.fit_transform(labels[keep])
    n_classes = len(le.classes_)
    majority = float(np.bincount(y).max() / len(y))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_labels = np.arange(n_classes)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))
    if progress:
        try:
            from tqdm.auto import tqdm
            folds = tqdm(folds, desc="project-CV folds")
        except ImportError:
            pass

    rows = []
    for fold, (tr, te) in enumerate(folds):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        train_loader = DataLoader(_CompositionDataset(Xtr, y[tr]),
                                  batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(_CompositionDataset(Xte, y[te]),
                                 batch_size=2048, shuffle=False)
        for name in models:
            if name == "linear":
                model = ProjectLinear(X.shape[1], n_classes)
            elif name == "mlp":
                model = ProjectMLP(X.shape[1], n_classes, hidden=tuple(hidden))
            else:
                raise ValueError(f"unknown model {name!r}")
            _train_classifier(model, train_loader, device, epochs, lr, seed + fold)
            logits = _predict_logits(model, eval_loader, device)
            pred = logits.argmax(axis=1)
            rows.append(dict(
                model=name, fold=fold,
                accuracy=accuracy_score(y[te], pred),
                balanced_accuracy=balanced_accuracy_score(y[te], pred),
                top5_accuracy=top_k_accuracy_score(y[te], logits, k=5, labels=all_labels),
                n_train=int(len(tr)), n_test=int(len(te)),
            ))

    cv_df = pd.DataFrame(rows)
    summary = dict(
        n_classes=n_classes, n_samples=int(keep.sum()), n_features=int(X.shape[1]),
        majority_baseline=majority, device=str(device),
        n_projects_dropped=int(len(dropped)),
        n_samples_dropped=int((~keep).sum()),
        min_project_size=min_project_size,
    )
    return cv_df, summary


# ---------------------------------------------------------------------------
# Step 5: depth vs alpha diversity
# ---------------------------------------------------------------------------

def _stream_alpha(parquet_path, keys, kind, batch_size=3000):
    """Observed richness and Shannon index per sample, streamed from an
    abundance Parquet file. `kind='proportion'` treats each row as already
    closed (relative abundance); `kind='count'` closes it first (raw or
    rarefied counts)."""
    pf = pq.ParquetFile(parquet_path)
    taxon_cols = [c for c in pf.schema.names if c != SAMPLE_COL]
    want = set(keys)
    ids, richness, shannon = [], [], []
    for batch in pf.iter_batches(batch_size=batch_size):
        d = batch.to_pandas()
        mask = d[SAMPLE_COL].isin(want)
        if not mask.any():
            continue
        d = d.loc[mask]
        M = d[taxon_cols].to_numpy(dtype=np.float64)
        totals = M.sum(axis=1, keepdims=True)
        p = np.divide(M, totals, out=np.zeros_like(M), where=totals > 0) if kind == "count" else M
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
        ids.append(d[SAMPLE_COL].to_numpy())
        richness.append((M > 0).sum(axis=1))
        shannon.append(ent)
    return pd.DataFrame({
        SAMPLE_COL: np.concatenate(ids),
        "richness": np.concatenate(richness),
        "shannon": np.concatenate(shannon),
    })


def depth_alpha_association(sample_depth_df, relative_path, rarefied_path, keys,
                             seed=42):
    """Spearman association between sequencing depth and alpha diversity,
    computed on the raw relative-abundance matrix (depth-varying) and on the
    rarefied matrix (depth fixed at the QC floor). The raw association is
    expected to be clearly positive; rarefaction *reduces* it but need not
    erase it (deeper-sequenced samples also tend to be from studies with
    genuinely richer communities). Either way, quantifying it is the case for
    treating depth as a covariate in Phase 5 rather than assuming it away.
    Returns a tidy frame + the merged per-sample table.
    """
    want = set(keys)
    depth = (sample_depth_df[sample_depth_df[SAMPLE_COL].isin(want)]
             [[SAMPLE_COL, "depth", "n_nonzero"]]
             .rename(columns={"n_nonzero": "richness_raw_counts"}))

    raw = _stream_alpha(relative_path, keys, kind="proportion").rename(
        columns={"richness": "richness_relative", "shannon": "shannon_relative"})
    rare = _stream_alpha(rarefied_path, keys, kind="count").rename(
        columns={"richness": "richness_rarefied", "shannon": "shannon_rarefied"})

    merged = depth.merge(raw, on=SAMPLE_COL, how="inner").merge(rare, on=SAMPLE_COL, how="inner")

    rows = []
    for label, col in [
        ("observed richness", "richness_raw_counts"),
        ("observed richness", "richness_relative"),
        ("Shannon", "shannon_relative"),
        ("observed richness", "richness_rarefied"),
        ("Shannon", "shannon_rarefied"),
    ]:
        depth_controlled = col.endswith("rarefied")
        rho, p = stats.spearmanr(merged["depth"], merged[col])
        rows.append(dict(
            metric=label, column=col,
            depth_controlled="rarefied" if depth_controlled else "raw",
            spearman_rho=float(rho), p=float(p), n=int(len(merged)),
        ))
    return pd.DataFrame(rows), merged


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def variance_report_md(marginal_df, sequential_df, clf_cv, clf_summary, alpha_df):
    """Human-readable companion to `reports/variance_explained.csv` -- the
    headline numbers Phase 5 reads before weighting anything."""
    lines = ["# Phase 4 -- technical variance (Goal 2)", ""]

    lines += ["## Marginal PERMANOVA (each factor alone)", "",
              "| distance | factor | n | groups | R2 | pseudo-F | p | perms |",
              "|---|---|---|---|---|---|---|---|"]
    for r in marginal_df.itertuples(index=False):
        lines.append(
            f"| {r.distance} | {r.factor} | {r.n:,} | {r.n_groups} | "
            f"{r.R2:.4f} | {r.pseudo_F:.1f} | {r.p:.3g} | {r.n_permutations} |"
            if pd.notna(r.R2) else
            f"| {r.distance} | {r.factor} | {r.n:,} | {r.n_groups} | - | - | - | {r.n_permutations} |"
        )

    lines += ["", "## Sequential decomposition (term order matters)", "",
              "| distance | analysis | # | factor | df | R2 | pseudo-F | p |",
              "|---|---|---|---|---|---|---|---|"]
    for r in sequential_df.itertuples(index=False):
        f = "-" if pd.isna(r.pseudo_F) else f"{r.pseudo_F:.1f}"
        p = "-" if pd.isna(r.p) else f"{r.p:.3g}"
        lines.append(f"| {r.distance} | {r.analysis} | {r.term_order} | {r.factor} | "
                     f"{r.df} | {r.R2:.4f} | {f} | {p} |")

    agg = clf_cv.groupby("model")[["accuracy", "balanced_accuracy", "top5_accuracy"]].agg(["mean", "std"])
    lines += ["", "## Project-predictability classifier (PyTorch)", "",
              f"- {clf_summary['n_samples']:,} samples, {clf_summary['n_classes']} projects, "
              f"{clf_summary['n_features']} CLR features, device `{clf_summary['device']}`",
              f"- majority-class baseline: **{clf_summary['majority_baseline']:.4f}**",
              f"- dropped {clf_summary['n_projects_dropped']} projects "
              f"(< {clf_summary['min_project_size']} samples, {clf_summary['n_samples_dropped']:,} samples)",
              "",
              "| model | accuracy | balanced acc. | top-5 acc. |",
              "|---|---|---|---|"]
    for model in agg.index:
        a = agg.loc[model]
        lines.append(f"| {model} | {a[('accuracy', 'mean')]:.4f} ± {a[('accuracy', 'std')]:.4f} "
                     f"| {a[('balanced_accuracy', 'mean')]:.4f} ± {a[('balanced_accuracy', 'std')]:.4f} "
                     f"| {a[('top5_accuracy', 'mean')]:.4f} ± {a[('top5_accuracy', 'std')]:.4f} |")

    lines += ["", "## Depth vs alpha diversity", "",
              "| metric | column | depth | Spearman rho | p | n |",
              "|---|---|---|---|---|---|"]
    for r in alpha_df.itertuples(index=False):
        lines.append(f"| {r.metric} | {r.column} | {r.depth_controlled} | "
                     f"{r.spearman_rho:.3f} | {r.p:.3g} | {r.n:,} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Plotting (Phase 4 figures)
# ---------------------------------------------------------------------------

def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    })


def plot_ordination_grid(coords_df, metadata, factor_cols, x, y, title,
                          continuous=("depth",), max_legend=8):
    """Scatter grid of an ordination, one panel per colouring variable.
    Categorical factors get a discrete colour cycle (legend capped at
    `max_legend`, e.g. `project` with 300+ levels is drawn without one);
    `continuous` factors get a viridis ramp."""
    import matplotlib.pyplot as plt
    _style()

    meta = metadata.set_index(SAMPLE_COL).loc[coords_df[SAMPLE_COL]]
    cols = [c for c in factor_cols if c in meta.columns]
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        values = meta[col].to_numpy()
        if col in continuous:
            v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
            sc = ax.scatter(coords_df[x], coords_df[y], c=v, cmap="viridis",
                            s=6, alpha=0.6, linewidths=0)
            fig.colorbar(sc, ax=ax, shrink=0.8)
        else:
            cats = pd.Series(values).astype("string").fillna("unknown")
            levels = list(pd.unique(cats))
            cmap = plt.cm.tab20(np.linspace(0, 1, max(len(levels), 1)))
            lut = {lev: cmap[i % len(cmap)] for i, lev in enumerate(levels)}
            ax.scatter(coords_df[x], coords_df[y],
                       c=[lut[c] for c in cats], s=6, alpha=0.6, linewidths=0)
            if len(levels) <= max_legend:
                handles = [plt.Line2D([], [], marker="o", ls="", color=lut[lev], label=str(lev))
                           for lev in levels]
                ax.legend(handles=handles, fontsize=7, loc="best", framealpha=0.6)
            else:
                ax.set_xlabel(f"{col}: {len(levels)} levels (no legend)")
        ax.set_title(col)
        ax.set_xticklabels([]); ax.set_yticklabels([])

    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_ordination_headline(coords_df, metadata, x, y, prop_explained, factor="project"):
    """The single headline panel: the ordination coloured by originating
    study. A composition that clusters by study more tightly than by anything
    biological is the whole point of Goal 2."""
    import matplotlib.pyplot as plt
    _style()
    meta = metadata.set_index(SAMPLE_COL).loc[coords_df[SAMPLE_COL]]
    cats = meta[factor].astype("string").fillna("unknown")
    levels = list(pd.unique(cats))
    cmap = plt.cm.gist_ncar(np.linspace(0, 1, max(len(levels), 1)))
    lut = {lev: cmap[i % len(cmap)] for i, lev in enumerate(levels)}
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(coords_df[x], coords_df[y], c=[lut[c] for c in cats],
               s=8, alpha=0.6, linewidths=0)
    ax.set_xlabel(f"{x}  ({prop_explained[0]:.1%} of inertia)")
    ax.set_ylabel(f"{y}  ({prop_explained[1]:.1%} of inertia)")
    ax.set_title(f"Ordination coloured by {factor} ({len(levels)} studies)")
    fig.tight_layout()
    return fig


def plot_variance_bars(marginal_df):
    """Marginal R^2 per factor, grouped by distance -- the bar version of
    `variance_explained.csv`'s marginal section."""
    import matplotlib.pyplot as plt
    _style()
    d = marginal_df.dropna(subset=["R2"])
    factors = list(d["factor"].unique())
    distances = list(d["distance"].unique())
    width = 0.8 / max(len(distances), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, dist in enumerate(distances):
        sub = d[d["distance"] == dist].set_index("factor").reindex(factors)
        ax.bar(np.arange(len(factors)) + i * width, sub["R2"].to_numpy(),
               width, label=dist)
    ax.set_xticks(np.arange(len(factors)) + width * (len(distances) - 1) / 2)
    ax.set_xticklabels(factors, rotation=35, ha="right")
    ax.set_ylabel("marginal R² (variance explained)")
    ax.set_title("How much between-sample variation each factor explains")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_sequential_bars(sequential_df):
    """Stacked R^2 for each sequential ordering, side by side, so the
    order-dependence (technical-first vs disease-first) is visible at a
    glance."""
    import matplotlib.pyplot as plt
    _style()
    d = sequential_df[sequential_df["factor"] != "Residual"]
    orders = list(d["analysis"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, order in enumerate(orders):
        sub = d[d["analysis"] == order].sort_values("term_order")
        bottom = 0.0
        for r in sub.itertuples(index=False):
            ax.bar(i, r.R2, bottom=bottom, width=0.6,
                   label=r.factor if i == 0 else None)
            bottom += r.R2
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels([o.replace("sequential:", "") for o in orders])
    ax.set_ylabel("cumulative R²")
    ax.set_title("Sequential decomposition depends on term order")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_project_classifier(cv_df, summary):
    """Per-model CV accuracy (with the majority-class baseline drawn in).
    A tall bar over a tiny baseline is the negative control firing as
    expected: study identity is trivially recoverable from composition."""
    import matplotlib.pyplot as plt
    _style()
    agg = cv_df.groupby("model")[["accuracy", "balanced_accuracy", "top5_accuracy"]].mean()
    err = cv_df.groupby("model")[["accuracy", "balanced_accuracy", "top5_accuracy"]].std()
    metrics = ["accuracy", "balanced_accuracy", "top5_accuracy"]
    models = list(agg.index)
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, m in enumerate(metrics):
        ax.bar(np.arange(len(models)) + i * width, agg[m].to_numpy(), width,
               yerr=err[m].to_numpy(), capsize=3, label=m)
    ax.axhline(summary["majority_baseline"], color="firebrick", ls="--", lw=1.2,
               label=f"majority baseline ({summary['majority_baseline']:.3f})")
    ax.set_xticks(np.arange(len(models)) + width)
    ax.set_xticklabels(models)
    ax.set_ylabel("score")
    ax.set_title(f"Predicting project from composition "
                 f"({summary['n_classes']} classes, {summary['n_samples']:,} samples)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_depth_alpha(merged_df):
    """Observed richness against depth, raw (relative-abundance counts) vs
    rarefied. The raw panel slopes up; the rarefied panel is flatter -- how
    much flatter is the degree to which the depth-richness association is a
    sampling-effort artifact versus real between-study community difference."""
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    for ax, col, name in [
        (axes[0], "richness_raw_counts", "raw (per-sample depth)"),
        (axes[1], "richness_rarefied", "rarefied to QC floor"),
    ]:
        hb = ax.hexbin(merged_df["depth"], merged_df[col], xscale="log",
                       gridsize=40, cmap="viridis", mincnt=1)
        rho, _ = stats.spearmanr(merged_df["depth"], merged_df[col])
        ax.set_title(f"{name}\nSpearman ρ = {rho:.2f}")
        ax.set_xlabel("reads per sample (log)")
        ax.set_ylabel("observed genera")
        fig.colorbar(hb, ax=ax, label="samples")
    fig.suptitle("Sequencing depth inflates apparent richness", fontweight="bold")
    fig.tight_layout()
    return fig
