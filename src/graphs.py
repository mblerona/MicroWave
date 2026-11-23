"""Relational view of the compendium: the same samples and taxa the flat
pipeline uses, rebuilt as graphs a GNN can read.

Two framings, built for different jobs:

**One heterogeneous graph (Framing A).** Sample nodes, taxon (leaf) nodes and
internal taxonomy nodes in a single `HeteroData`. A sample's composition
lives *only* on its sample -> taxon edges; its own node features are
non-compositional scalars (depth, richness). Three variants:

- `G1` -- samples, taxa, taxonomy. No sample is connected to another sample
  except through shared taxon nodes.
- `G2` -- G1 plus k-NN sample -> sample edges on Aitchison distance. These
  edges are a deliberate leakage probe: under leave-one-project-out a
  held-out sample's nearest neighbours are mostly from its own study, so the
  edges can carry batch identity straight into a prediction.
- `G3` -- G1 plus one node per project, joined to its samples and carrying
  the project's technical covariates. The literal one-node-type-per-table
  relational framing; for a disease classifier it is an explicit path to the
  batch label, so it is an upper bound on leakage, not a candidate model.

**One small graph per sample (Framing B).** The fixed 666-node taxonomy
skeleton, with the sample's CLR values on the leaves. Graphs are disjoint,
so one sample's prediction cannot be influenced by any other sample -- the
property that makes a LOPO score comparable to the flat baselines.

Taxonomy nodes are keyed by the full lineage tuple, never by name: 46 of the
419 modelled taxa have genus `NA` and 44 share genus `Incertae Sedis`, and
keying on the name string would silently fuse unrelated clades into one node.

Plain functions, paths / dataframes in and tensors / graph objects out, same
shape as the rest of `src/`. The wide relative-abundance matrix is streamed a
batch of rows at a time; only its non-zero cells are ever materialised.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import sparse

SAMPLE_COL = "sample"  # "{project}_{srr}", matching the taxa Parquet / npz files
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]

# Every column a sample node may carry in Framing A. Composition must reach a
# sample only through its abundance edges; if the 419-dim CLR vector were a
# node feature, the GNN would hold the whole flat input before any message is
# passed and the comparison against the flat baselines would be meaningless.
SAMPLE_FEATURES = ("log10_depth", "n_nonzero")

# Technical covariates summarised onto project nodes in G3.
PROJECT_COVARIATES = ("amplicon", "kit_family", "instrument", "bead_beating", "region")

# Scales relative abundance before log1p so typical non-zero weights spread
# over ~0-9 rather than all sitting near log1p(0) ~ 0.
ABUNDANCE_SCALE = 1e4


def _one_hot(codes, n):
    out = np.zeros((len(codes), n), dtype=np.float32)
    out[np.arange(len(codes)), codes] = 1.0
    return out


# ---------------------------------------------------------------------------
# Taxonomy skeleton
# ---------------------------------------------------------------------------

def build_taxonomy_edges(taxon_table, taxa_cols):
    """The lineage tree over the modelled taxa.

    Node ids: the leaves come first, `0 .. len(taxa_cols)-1` in `taxa_cols`
    order, so a leaf's id equals its CLR column position; internal nodes
    follow, ordered by rank then lineage. Every node is a lineage *prefix*
    tuple (`("Bacteria", "Bacillota", "Clostridia")`), so two `NA` families
    under different orders are different nodes.

    The taxonomy is a forest, not a tree: the modelled taxa sit under four
    kingdom-level roots (Bacteria, Archaea, Eukaryota and an all-`NA` lineage).

    Returns (edge_index, node_index, rank_of_node):
    - `edge_index` -- LongTensor [2, n_nodes - n_roots], child -> parent only;
      callers add the reverse direction.
    - `node_index` -- DataFrame, one row per node: node_id, lineage (tuple),
      rank (0 = kingdom .. 5 = genus), rank_name, name, is_leaf, parent
      (-1 for a root), full_string (leaves only).
    - `rank_of_node` -- LongTensor [n_nodes].
    """
    tt = taxon_table.set_index("full_string").loc[list(taxa_cols), RANKS]
    leaves = [tuple(r) for r in tt.itertuples(index=False)]
    if len(set(leaves)) != len(leaves):
        raise ValueError("two modelled taxa share a full lineage")

    internal = sorted({leaf[:k] for leaf in leaves for k in range(1, len(RANKS))},
                      key=lambda t: (len(t), t))
    lineages = leaves + internal
    node_id = {lin: i for i, lin in enumerate(lineages)}
    parent = np.array([node_id[lin[:-1]] if len(lin) > 1 else -1 for lin in lineages])
    rank = np.array([len(lin) - 1 for lin in lineages])

    node_index = pd.DataFrame({
        "node_id": np.arange(len(lineages)),
        "lineage": lineages,
        "rank": rank,
        "rank_name": [RANKS[r] for r in rank],
        "name": [lin[-1] for lin in lineages],
        "is_leaf": np.arange(len(lineages)) < len(leaves),
        "parent": parent,
        "full_string": list(taxa_cols) + [None] * len(internal),
    })

    child = np.flatnonzero(parent >= 0)
    edge_index = torch.tensor(np.vstack([child, parent[child]]), dtype=torch.long)
    return edge_index, node_index, torch.tensor(rank, dtype=torch.long)


def check_taxonomy(node_index, edge_index, n_leaves, n_internal=None):
    """Structural invariants of `build_taxonomy_edges`, raised as errors so a
    notebook can't continue past a broken skeleton: node counts, one parent
    per non-root node, every leaf reaches a root, and poorly-named genera
    stay as distinct leaves rather than collapsing onto one shared name."""
    leaves = node_index[node_index["is_leaf"]]
    assert len(leaves) == n_leaves, f"{len(leaves)} leaves, expected {n_leaves}"
    if n_internal is not None:
        got = int((~node_index["is_leaf"]).sum())
        assert got == n_internal, f"{got} internal nodes, expected {n_internal}"

    child = edge_index[0].numpy()
    assert len(np.unique(child)) == len(child), "a node has more than one parent"
    n_roots = int((node_index["parent"] < 0).sum())
    assert edge_index.shape[1] == len(node_index) - n_roots

    parent = node_index["parent"].to_numpy()
    for start in leaves["node_id"]:
        node, steps = start, 0
        while parent[node] >= 0:
            node, steps = parent[node], steps + 1
            assert steps <= len(RANKS), f"cycle above leaf {start}"
        assert node_index.at[node, "rank"] == 0, f"leaf {start} does not reach a kingdom"

    for name in ("NA", "Incertae Sedis"):
        same = leaves[leaves["name"] == name]
        assert same["lineage"].nunique() == len(same), f"genus {name!r} leaves were merged"


def taxonomy_edge_attr(edge_index, rank_of_node):
    """One-hot (6-dim) of the parent's rank for each child -> parent edge.
    The same attribute is reused for the reversed parent -> child edge, so
    a message knows which level of the tree it is crossing."""
    return torch.from_numpy(_one_hot(rank_of_node[edge_index[1]].numpy(), len(RANKS)))


# ---------------------------------------------------------------------------
# Sample <-> taxon abundance edges
# ---------------------------------------------------------------------------

def build_bipartite_edges(relative_path, clr_path, samples, taxa_cols, batch_size=5000):
    """Sample -> taxon edges, one per non-zero relative abundance.

    Zeros are not edges: absence is represented by the absence of an edge,
    which keeps the graph ~13.5% dense instead of materialising all ~49M
    sample-taxon pairs. Two passes over Parquet, each streamed and reading
    only the `taxa_cols` columns:

    1. relative abundance -> the sparsity pattern and the non-negative edge
       weight `log1p(relative * 1e4)`. The weight must be non-negative for
       degree-normalised convolutions, which is why it is not the CLR value.
    2. CLR -> the signed CLR value at each of those cells, carried as an
       edge attribute for the architectures that read edge features.

    Sample node ids are positions in `samples`. Returns (edge_index,
    edge_weight, edge_attr) with edge_index [2, E] (sample id, leaf taxon
    id), edge_weight [E], edge_attr [E, 3] = [clr, log1p(rel * 1e4), 1.0].
    """
    samples = list(samples)
    pos = pd.Series(np.arange(len(samples)), index=pd.Index(samples))
    if not pos.index.is_unique:
        raise ValueError("duplicate sample keys")
    taxa_cols = list(taxa_cols)

    rows, cols, vals, seen = [], [], [], 0
    pf = pq.ParquetFile(relative_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=[SAMPLE_COL, *taxa_cols]):
        d = batch.to_pandas()
        d = d[d[SAMPLE_COL].isin(pos.index)]
        if d.empty:
            continue
        M = d[taxa_cols].to_numpy(dtype=np.float32)
        r, c = np.nonzero(M)
        rows.append(pos.loc[d[SAMPLE_COL]].to_numpy()[r])
        cols.append(c)
        vals.append(M[r, c])
        seen += len(d)
    if seen != len(samples):
        raise ValueError(f"{len(samples) - seen} samples not found in {relative_path}")

    rel = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(samples), len(taxa_cols)), dtype=np.float32)
    rel.sort_indices()

    # CLR values aligned to the csr's data order: for each streamed row, the
    # positions indptr[row] .. indptr[row+1] in rel.data are its non-zero cells.
    clr_vals = np.full(rel.nnz, np.nan, dtype=np.float32)
    pf = pq.ParquetFile(clr_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=[SAMPLE_COL, *taxa_cols]):
        d = batch.to_pandas()
        d = d[d[SAMPLE_COL].isin(pos.index)]
        if d.empty:
            continue
        r = pos.loc[d[SAMPLE_COL]].to_numpy()
        starts, counts = rel.indptr[r], np.diff(rel.indptr)[r]
        local = np.repeat(np.arange(len(r)), counts)
        flat = np.repeat(starts - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())
        clr_vals[flat] = d[taxa_cols].to_numpy(dtype=np.float32)[local, rel.indices[flat]]
    if np.isnan(clr_vals).any():
        # A sample can only be missing from the CLR file if none of its reads
        # fall in the modelled taxa -- in which case it has no edges here.
        raise ValueError("a non-zero abundance cell has no CLR value")

    src = np.repeat(np.arange(len(samples)), np.diff(rel.indptr))
    weight = np.log1p(rel.data * ABUNDANCE_SCALE).astype(np.float32)
    edge_index = torch.from_numpy(np.vstack([src, rel.indices]).astype(np.int64))
    edge_attr = torch.from_numpy(np.column_stack([clr_vals, weight, np.ones_like(weight)]))
    return edge_index, torch.from_numpy(weight), edge_attr


def edges_per_sample(edge_index, n_samples):
    """Number of abundance edges (non-zero modelled taxa) per sample node."""
    return np.bincount(edge_index[0].numpy(), minlength=n_samples)


def presence_from_edges(edge_index, n_samples, n_taxa):
    """Dense boolean (n_samples x n_taxa) presence matrix from abundance
    edges -- small at cohort scale (14k x 419), and the presence indicator
    Framing B puts on each leaf."""
    out = np.zeros((n_samples, n_taxa), dtype=bool)
    out[edge_index[0].numpy(), edge_index[1].numpy()] = True
    return out


# ---------------------------------------------------------------------------
# Sample <-> sample k-NN edges (G2, the leakage probe)
# ---------------------------------------------------------------------------

def build_knn_edges(clr_matrix, samples, projects, k, *, drop_same_project=False):
    """k nearest neighbours of every sample by Aitchison distance (Euclidean
    on CLR), made symmetric so messages pass both ways.

    `pct_same_project` is measured on the directed k-NN selections (n x k),
    before symmetrising or dropping anything -- it is the mechanism behind
    any leakage G2 shows. The same-project flag is returned *separately* from
    `edge_attr` on purpose: it is instrumentation for the leakage
    experiment, and keeping it out of the attribute tensor means no model can
    be handed it by accident.

    `drop_same_project=True` deletes every edge whose endpoints share a
    project -- the mitigation arm. Ties are broken by sklearn's deterministic
    brute-force ordering, so the edges are reproducible without a seed.

    Returns (edge_index [2, E], edge_weight [E] = 1 / (1 + distance),
    edge_attr [E, 1] = distance, same_project [E] bool, pct_same_project).
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(clr_matrix, dtype=np.float32)
    projects = np.asarray(projects)
    if len(X) != len(samples) or len(projects) != len(samples):
        raise ValueError("clr_matrix, samples and projects must align")

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(X)
    dist, idx = nn.kneighbors(X)
    # Drop each sample's own entry -- position 0, unless an exact duplicate
    # composition shares distance 0 and sorted ahead of it.
    own = idx == np.arange(len(X))[:, None]
    keep = ~own
    keep[~own.any(axis=1), -1] = False
    dist = dist[keep].reshape(len(X), k)
    idx = idx[keep].reshape(len(X), k)

    src = np.repeat(np.arange(len(X)), k)
    dst = idx.ravel()
    d = dist.ravel()
    pct_same = float((projects[src] == projects[dst]).mean())

    # symmetrise; an edge picked from both ends keeps one copy
    pairs = pd.DataFrame({"src": np.concatenate([src, dst]),
                          "dst": np.concatenate([dst, src]),
                          "dist": np.concatenate([d, d])})
    pairs = pairs.drop_duplicates(["src", "dst"]).sort_values(["src", "dst"])
    same = projects[pairs["src"].to_numpy()] == projects[pairs["dst"].to_numpy()]
    if drop_same_project:
        pairs, same = pairs[~same], same[~same]

    edge_index = torch.from_numpy(pairs[["src", "dst"]].to_numpy().T.astype(np.int64).copy())
    dist_t = torch.from_numpy(pairs["dist"].to_numpy(dtype=np.float32))
    return (edge_index, 1.0 / (1.0 + dist_t), dist_t[:, None],
            torch.from_numpy(same), pct_same)


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------

def sample_node_features(sample_depth_df, samples):
    """Non-compositional sample features in `SAMPLE_FEATURES` order:
    log10(read depth) and the number of non-zero taxa (over all 4,680, as
    recorded at ingest). Raw values -- scaling is fit per CV fold on training
    rows, not here."""
    d = sample_depth_df.set_index(SAMPLE_COL).loc[list(samples)]
    x = np.column_stack([np.log10(d["depth"].to_numpy(dtype=np.float64)),
                         d["n_nonzero"].to_numpy(dtype=np.float64)])
    return torch.from_numpy(x.astype(np.float32)), list(SAMPLE_FEATURES)


def assert_no_composition(feature_names, taxa_cols):
    """The sample-node rule: every sample feature is on the whitelist, and
    none is a taxon column."""
    bad = [f for f in feature_names if f not in SAMPLE_FEATURES]
    assert not bad, f"sample node features outside the non-compositional whitelist: {bad}"
    overlap = set(feature_names) & set(taxa_cols)
    assert not overlap, f"sample node features include taxon columns: {sorted(overlap)[:3]}"


def taxon_leaf_stats(clr_matrix, presence, rows=None):
    """Per-taxon prevalence and mean CLR over `rows` (default: all). In a CV
    experiment pass the training rows only -- like a scaler, these are
    statistics of the data and must not see the held-out fold.

    Returns (log10_prevalence, mean_clr), each [n_taxa]. Prevalence is
    floored at half a sample so a taxon absent from `rows` gets a finite log.
    """
    clr = np.asarray(clr_matrix, dtype=np.float64)
    pres = np.asarray(presence, dtype=bool)
    if rows is not None:
        clr, pres = clr[rows], pres[rows]
    n = len(pres)
    prev = np.maximum(pres.mean(axis=0), 0.5 / n)
    return (torch.from_numpy(np.log10(prev).astype(np.float32)),
            torch.from_numpy(clr.mean(axis=0).astype(np.float32)))


def project_node_features(harmonized_df, projects):
    """One row per project in `projects` order: for each technical covariate,
    the share of the project's samples at each level. Most covariates are
    constant within a project (so this is a one-hot), but `instrument` and
    `region` can vary, and a share keeps that rather than picking a mode.
    Blanks are an explicit `unknown` level, as in Phase 4."""
    from src import variance as var

    df = var.prepare_covariates(harmonized_df[["project", "srr", *PROJECT_COVARIATES]])
    blocks, names = [], []
    for col in PROJECT_COVARIATES:
        shares = pd.crosstab(df["project"], df[col], normalize="index")
        shares = shares.reindex(list(projects)).fillna(0.0)
        blocks.append(shares.to_numpy(dtype=np.float32))
        names += [f"{col}={lev}" for lev in shares.columns]
    return torch.from_numpy(np.hstack(blocks)), names


# ---------------------------------------------------------------------------
# Framing A: one heterogeneous graph
# ---------------------------------------------------------------------------

def build_hetero_graph(variant, samples, *, projects, taxonomy, bipartite,
                       sample_x, sample_feature_names, taxa_cols, leaf_stats,
                       y=None, knn=None, project_x=None):
    """Assemble G1 / G2 / G3 as a `HeteroData`.

    - `taxonomy` -- (edge_index, node_index, rank_of_node) from
      `build_taxonomy_edges`.
    - `bipartite` -- (edge_index, edge_weight, edge_attr) from
      `build_bipartite_edges`, sample ids = positions in `samples`.
    - `leaf_stats` -- (log10_prevalence, mean_clr) from `taxon_leaf_stats`.
    - `knn` -- G2 only: the 5-tuple from `build_knn_edges`.
    - `project_x` -- G3 only: features for `sorted(set(projects))`, from
      `project_node_features`.

    Taxon and taxon_internal nodes carry `node_id` for a learnable embedding
    plus their fixed features in `x`. Sample nodes carry `project` (integer
    codes into `project_names`) as split metadata -- never in `x`.
    """
    from torch_geometric.data import HeteroData

    if variant not in ("G1", "G2", "G3"):
        raise ValueError(f"unknown variant {variant!r}")
    if (knn is not None) != (variant == "G2"):
        raise ValueError("k-NN edges are given exactly when variant == 'G2'")
    if (project_x is not None) != (variant == "G3"):
        raise ValueError("project features are given exactly when variant == 'G3'")
    assert_no_composition(sample_feature_names, taxa_cols)
    assert sample_x.shape == (len(samples), len(sample_feature_names))

    tax_edges, node_index, rank_of_node = taxonomy
    n_leaf = int(node_index["is_leaf"].sum())
    n_internal = len(node_index) - n_leaf
    n_ranks = len(RANKS)
    rank_oh = torch.from_numpy(_one_hot(rank_of_node.numpy(), n_ranks))
    log_prev, mean_clr = leaf_stats

    data = HeteroData()
    data["sample"].x = sample_x
    data["sample"].num_nodes = len(samples)
    proj_codes, project_names = pd.factorize(pd.Series(projects), sort=True)
    data["sample"].project = torch.from_numpy(proj_codes.astype(np.int64))
    data.project_names = list(project_names)
    data.sample_keys = list(samples)
    if y is not None:
        data["sample"].y = torch.as_tensor(np.asarray(y), dtype=torch.long)

    data["taxon"].node_id = torch.arange(n_leaf)
    data["taxon"].x = torch.cat([log_prev[:, None], mean_clr[:, None], rank_oh[:n_leaf]], dim=1)
    data["taxon_internal"].node_id = torch.arange(n_internal)
    data["taxon_internal"].x = rank_oh[n_leaf:]

    ab_index, ab_weight, ab_attr = bipartite
    data["sample", "abundance", "taxon"].edge_index = ab_index
    data["sample", "abundance", "taxon"].edge_weight = ab_weight
    data["sample", "abundance", "taxon"].edge_attr = ab_attr
    data["taxon", "rev_abundance", "sample"].edge_index = ab_index.flip(0)
    data["taxon", "rev_abundance", "sample"].edge_weight = ab_weight
    data["taxon", "rev_abundance", "sample"].edge_attr = ab_attr

    # child -> parent edges split by whether the child is a leaf; ids shifted
    # into each node type's local numbering (internal ids start at n_leaf).
    tax_attr = taxonomy_edge_attr(tax_edges, rank_of_node)
    from_leaf = tax_edges[0] < n_leaf
    for child_type, mask, child_offset in (("taxon", from_leaf, 0),
                                           ("taxon_internal", ~from_leaf, n_leaf)):
        ei = torch.stack([tax_edges[0, mask] - child_offset, tax_edges[1, mask] - n_leaf])
        ones = torch.ones(ei.shape[1])
        data[child_type, "child_of", "taxon_internal"].edge_index = ei
        data[child_type, "child_of", "taxon_internal"].edge_weight = ones
        data[child_type, "child_of", "taxon_internal"].edge_attr = tax_attr[mask]
        data["taxon_internal", "parent_of", child_type].edge_index = ei.flip(0)
        data["taxon_internal", "parent_of", child_type].edge_weight = ones
        data["taxon_internal", "parent_of", child_type].edge_attr = tax_attr[mask]

    if variant == "G2":
        knn_index, knn_weight, knn_attr, same_project, pct_same = knn
        data["sample", "similar_to", "sample"].edge_index = knn_index
        data["sample", "similar_to", "sample"].edge_weight = knn_weight
        data["sample", "similar_to", "sample"].edge_attr = knn_attr
        data["sample", "similar_to", "sample"].same_project = same_project
        data.pct_same_project_neighbours = pct_same

    if variant == "G3":
        assert project_x.shape[0] == len(project_names)
        data["project"].x = project_x
        data["project"].num_nodes = len(project_names)
        ei = torch.stack([torch.arange(len(samples)), data["sample"].project])
        data["sample", "from", "project"].edge_index = ei
        data["project", "rev_from", "sample"].edge_index = ei.flip(0)

    data.variant = variant
    return data


# ---------------------------------------------------------------------------
# Framing B: one taxonomy graph per sample
# ---------------------------------------------------------------------------

LEAF_FEATURES = ("clr", "present", "log10_prevalence")


def build_per_sample_dataset(clr_df, taxonomy, labels=None, *, presence, log10_prevalence):
    """One `Data` per row of `clr_df` (indexed by `sample`, columns in leaf
    order), all sharing the same taxonomy skeleton.

    Node features `x` [666, 3] = `LEAF_FEATURES`: the sample's CLR value, a
    presence indicator (CLR alone can't say whether a taxon was observed --
    a zero count still gets a finite pseudocount CLR), and the taxon's
    prevalence. Internal nodes get zeros there and are told apart by `rank`
    (for a learnable rank embedding) and `is_leaf`. `edge_index` is the
    child -> parent edges plus their reverse, and is the *same tensor
    object* in every graph -- only features differ.

    `presence` -- bool array aligned with `clr_df`. `log10_prevalence` --
    [n_leaves], from `taxon_leaf_stats` on the training rows.
    `labels` -- optional Series indexed by sample -> graph-level `y`.
    """
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected

    tax_edges, node_index, rank_of_node = taxonomy
    n_leaf = int(node_index["is_leaf"].sum())
    n_nodes = len(node_index)
    if list(clr_df.columns) != list(node_index.loc[node_index["is_leaf"], "full_string"]):
        raise ValueError("clr_df columns must be the taxonomy leaves, in leaf order")
    presence = np.asarray(presence, dtype=bool)
    assert presence.shape == clr_df.shape

    edge_index = to_undirected(tax_edges, num_nodes=n_nodes)
    is_leaf = torch.zeros(n_nodes, dtype=torch.bool)
    is_leaf[:n_leaf] = True

    clr = clr_df.to_numpy(dtype=np.float32)
    x_all = np.zeros((len(clr_df), n_nodes, len(LEAF_FEATURES)), dtype=np.float32)
    x_all[:, :n_leaf, 0] = clr
    x_all[:, :n_leaf, 1] = presence
    x_all[:, :n_leaf, 2] = np.asarray(log10_prevalence, dtype=np.float32)
    x_all = torch.from_numpy(x_all)

    y = None if labels is None else labels.loc[clr_df.index].to_numpy()
    graphs = []
    for i, key in enumerate(clr_df.index):
        g = Data(x=x_all[i], edge_index=edge_index, rank=rank_of_node,
                 is_leaf=is_leaf, num_nodes=n_nodes)
        g.sample = key
        if y is not None:
            g.y = torch.tensor([int(y[i])])
        graphs.append(g)
    return graphs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(x, fmt=".3f"):
    return "—" if x is None or pd.isna(x) else format(x, fmt)


def _pm(mean, sd, fmt=".3f"):
    return _f(mean, fmt) if sd is None or pd.isna(sd) else f"{_f(mean, fmt)} ± {_f(sd, '.3f')}"


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _noise_verdict(delta, sd, up="win", down="loss"):
    return up if delta > sd else (down if delta < -sd else "within fold noise")


def graph_report_md(*, project_cv, committed_project_cv, leakage_gap, label_agreement,
                    flip_test, label_graph, disease_cv, disease_summary, pretrain_summary,
                    pretrain_reconstruction, pretrain_study, pretrain_history):
    """Human-readable companion to the graph-model CSVs: every graph result
    beside the baseline it is judged against, limitations first, verdicts read
    against the noise bars stated before the runs (fold-to-fold sd, or 2x the
    seed sd for the leakage and tree gaps). Inputs are the frames the graph
    notebooks write; disease AUCs are means over folds, then over seeds."""
    from src import graph_disease as gd

    lopo, rand = "leave_project_out", "random_kfold"
    summ = pd.concat([disease_summary, pretrain_summary], ignore_index=True)
    s = summ.set_index(["disease_category", "model", "cv_scheme"])
    cats = list(dict.fromkeys(disease_summary["disease_category"]))
    n_studies = (disease_cv[disease_cv["cv_scheme"] == lopo]
                 .groupby("disease_category")["test_project"].nunique())
    lopo_sd = s.xs(lopo, level="cv_scheme")["fold_sd"].dropna()
    recon = pretrain_reconstruction.set_index(pretrain_reconstruction.columns[0])
    study = pretrain_study.set_index(pretrain_study.columns[0]).iloc[:, 0]
    flip = flip_test.set_index(flip_test.columns[0])
    label_graph = label_graph.set_index(label_graph.columns[0])

    lines = ["# Graph models: relational extension", "",
             "Graph neural networks over the same 419 genus-level CLR features as the flat "
             "pipeline, judged under the same leave-one-project-out (LOPO) evaluation, on the "
             "same folds. Disease AUCs are averaged over folds, then over three model seeds "
             "(42, 43, 44); `±` is the seed-to-seed sd unless a column says otherwise. LOPO is "
             "the number that counts; random k-fold shares studies between training and test "
             "and is shown only to measure how much that flatters a model.", ""]

    # ---- limitations -------------------------------------------------------
    lines += ["## Limitations, read these first", "",
              "- **Few studies per disease.** LOPO has one fold per study: "
              + ", ".join(f"{c} {n_studies.get(c, 0)}" for c in cats)
              + f". The fold-to-fold sd of a LOPO AUC is therefore large ({lopo_sd.min():.2f}"
              f"–{lopo_sd.max():.2f}), and most differences below sit inside it. A difference "
              "that is consistent across seeds is not one that would survive a different set "
              "of studies.",
              "- **Seed sd measures model randomness only.** Folds are fixed (built from seed "
              "42), so the seed sd says nothing about which studies happen to be in the data. "
              "The pretrained encoder is a single run (seed 42); its downstream seed sd "
              "excludes pretraining randomness.",
              "- **The flat gradient-boosting baseline was re-run, not read from the committed "
              "file.** Logistic regression reproduces `classifier_cv.csv` exactly; gradient "
              "boosting moves individual folds by up to ~0.035 AUC (means within fold noise). "
              "The cause is not identified. Disease tables use the re-run so that every model "
              "shares one environment.",
              "- **No tuning.** Every neural model uses one fixed configuration (60 epochs, no "
              "early stopping, one learning rate) set before the runs. A tuned model of any "
              "kind could score higher; this compares like with like, not best achievable.",
              "- **Three diseases have no defensible signal.** Phase 5 found nothing for T1D, "
              "asthma and cancer_other, and several of their LOPO AUCs are below 0.5. They are "
              "kept for completeness; a graph model \"finding something\" there is a reason "
              "for suspicion, not a result.",
              "- **Scope.** Genus-level 16S profiles only; only GraphSAGE was pretrained; the "
              "study identity in the pretraining neighbour check is the accession prefix of "
              "the sample key.", ""]

    # ---- verdicts used by the headline and the sections --------------------
    win_rows, n_win = [], {"win": 0, "within fold noise": 0, "loss": 0}
    for cat in cats:
        for model, mlp in (("gcn", "mlp_gcn"), ("sage_g1", "mlp_sage"), ("gat_g1", "mlp_gat")):
            rivals = {m: s.loc[(cat, m, lopo), "auc_mean"] for m in ("logreg", "gboost", mlp)}
            best = max(rivals, key=rivals.get)
            auc, sd = s.loc[(cat, model, lopo), "auc_mean"], s.loc[(cat, model, lopo), "fold_sd"]
            verdict = _noise_verdict(auc - rivals[best], sd)
            n_win[verdict] += 1
            win_rows.append([cat, gd.LABEL[model], _f(auc), gd.LABEL[best], _f(rivals[best]),
                             f"{auc - rivals[best]:+.3f}", _f(sd), verdict])
    pt_rows, n_pt = [], {"helps": 0, "within fold noise": 0, "hurts": 0}
    for cat in cats:
        ref = s.loc[(cat, "sage_g1", lopo)]
        for m in gd.PRETRAINED_MODELS:
            row = s.loc[(cat, m, lopo)]
            delta = row["auc_mean"] - ref["auc_mean"]
            verdict = _noise_verdict(delta, ref["fold_sd"], "helps", "hurts")
            n_pt[verdict] += 1
            pt_rows.append([cat, gd.LABEL[m], _pm(ref["auc_mean"], ref["auc_seed_sd"]),
                            _pm(row["auc_mean"], row["auc_seed_sd"]), f"{delta:+.3f}",
                            _f(ref["fold_sd"]), verdict])
    proj = project_cv.groupby("model")["accuracy"].mean()
    outside = int((leakage_gap["graph_leakage_gap"].abs()
                   > 2 * leakage_gap["graph_leakage_gap_seed_sd"]).sum())

    lines += ["## Answers in one table", ""]
    lines += _md_table(
        ["question", "answer", "evidence"],
        [["Does a graph model encode the study as well as a flat one? (calibration)",
          "yes, a tie",
          f"study accuracy {proj['GraphSAGE on G1']:.3f} (GraphSAGE, G1) vs "
          f"{proj['linear']:.3f} (linear, CLR)"],
         ["Does a sample-similarity graph leak the study into a disease model?",
          "no, under all three protocols",
          f"G2 − G1 within ±{leakage_gap['graph_leakage_gap'].abs().max():.3f}; {outside} of "
          f"{len(leakage_gap)} gaps beyond 2× seed sd, about what chance gives"],
         ["Can a graph leak at all?", "yes, if it is built from labels",
          f"G2 with label-built edges: AUC {label_graph.iloc[:, 0].min():.2f} even inductively"],
         ["Does a graph model beat the flat models on honest splits?", "no",
          f"{n_win['win']} wins, {n_win['within fold noise']} within fold noise, "
          f"{n_win['loss']} losses ({len(win_rows)} graph model × disease tests)"],
         ["Does label-free pretraining on the unlabelled samples help?",
          "small gains, mostly inside fold noise",
          f"{n_pt['helps']} helps, {n_pt['within fold noise']} within fold noise, "
          f"{n_pt['hurts']} hurts ({len(pt_rows)} tests); it lifts GraphSAGE to the flat "
          "models' level, not past them"],
         ["Did pretraining teach the encoder study identity?", "no more than the data has",
          f"same-study share of 10 neighbours {study['encoder, pretrained']:.3f} vs "
          f"{study['CLR composition']:.3f} in CLR space (chance {study['chance']:.3f})"]])
    lines.append("")

    # ---- experiment A ------------------------------------------------------
    committed = committed_project_cv.groupby("model")["accuracy"].mean()
    lines += ["## Experiment A — calibration: predicting the study", "",
              "A graph model must at least see what a flat model sees. Task: which study a "
              "sample came from (5 folds, the committed project-prediction folds). The "
              "committed flat models are shown beside their re-run. G3 contains study nodes, "
              "so it is a ceiling, not a model.", ""]
    rows = []
    for model, g in project_cv.groupby("model", sort=False):
        rows.append([model, g["inputs"].iloc[0], g["role"].iloc[0],
                     _pm(g["accuracy"].mean(), g["accuracy"].std()),
                     _f(g["balanced_accuracy"].mean()), _f(g["top5_accuracy"].mean()),
                     _f(committed.get(model)), _f(g["n_params"].iloc[0], ",.0f")])
    lines += _md_table(["model", "inputs", "role", "accuracy (± fold sd)", "balanced acc.",
                        "top-5", "committed accuracy", "parameters"], rows)
    lines += ["", f"Majority-class accuracy: {project_cv['majority_baseline'].iloc[0]:.3f}.", ""]

    # ---- experiment C ------------------------------------------------------
    pct = leakage_gap.drop_duplicates("disease_category")
    lines += ["## Experiment C — does a graph leak the study into a disease model?", "",
              "The same GraphSAGE on four graphs: G1 (samples, taxa, taxonomy), G2 (+ 10 "
              "nearest-neighbour sample edges), G2 filtered (same-study neighbour edges "
              "removed), G3 (+ study nodes). Three protocols, strict to leaky: **inductive** "
              "(held-out samples only receive messages), **transductive** (they also send), "
              "**transductive_labels** (training labels are an input feature). A gap counts if "
              "it is beyond 2× its seed sd.", "",
              "**The mechanism exists:** share of a sample's 10 nearest neighbours (CLR, "
              "Aitchison) from its own study: "
              + ", ".join(f"{r.disease_category} {r.pct_same_project_neighbours:.0%}"
                          for r in pct.itertuples()) + ".", ""]
    lines += _md_table(
        ["protocol", "disease", "split", "G1 AUC", "G2 − G1", "G2 filtered − G1", "G3 − G1",
         "flat logreg", "flat gboost"],
        [[r.protocol, r.disease_category, r.cv_scheme.replace("_", " "), _f(r.auc_G1),
          _pm(r.graph_leakage_gap, r.graph_leakage_gap_seed_sd, "+.3f"),
          _pm(r.gap_G2_filtered, r.gap_G2_filtered_seed_sd, "+.3f"),
          _pm(r.gap_G3, r.gap_G3_seed_sd, "+.3f"), _f(r.flat_auc_logreg),
          _f(r.flat_auc_gboost)] for r in leakage_gap.itertuples(index=False)])
    lines += ["", "**Why there is no leak, in three checks:**", "",
              "1. Neighbours' labels say no more than the study's case rate: label agreement "
              "between k-NN neighbours sits at the same-study chance level.", ""]
    lines += _md_table(
        ["disease", "k-NN label agreement", "chance (random)", "chance (random, same study)"],
        [[r.disease_category, _f(r.knn_label_agreement), _f(r.chance_random),
          _f(r.chance_same_study)] for r in label_agreement.itertuples(index=False)])
    lines += ["", "2. The labels-as-input model ignores the label channel: scored with the "
              "neighbours' labels true, flipped or withheld (IBD, random k-fold, seed 42), its "
              "AUC barely moves.", ""]
    lines += _md_table(["graph", "labels true", "labels flipped", "labels withheld"],
                       [[g, _f(r["true"]), _f(r["flipped"]), _f(r["withheld"])]
                        for g, r in flip.iterrows()])
    lines += ["", "3. What a real leak looks like: G2 with its sample edges built from the "
              "labels (IBD), trained under the strict inductive protocol. The protocol guards "
              "against leaks through messages, not against a graph whose construction used "
              "the labels.", ""]
    lines += _md_table(["split", *label_graph.columns],
                       [[i.replace("_", " "), *(_f(v) for v in r)]
                        for i, r in label_graph.iterrows()])
    lines.append("")

    # ---- experiment B ------------------------------------------------------
    order = [m for m in gd.ORDER if m in set(disease_summary["model"])]
    lines += ["## Experiment B — disease classification", "",
              "The flat baselines (logistic regression, gradient boosting), a GCN on one "
              "taxonomy tree per sample, the same GCN on a shuffled tree, GraphSAGE and GAT on "
              "G1 (inductive), and an MLP size-matched to each graph model.", ""]
    for scheme, title in ((lopo, "### LOPO AUC"),
                          (rand, "### Random k-fold AUC (studies shared; for the inflation only)")):
        lines += [title, ""]
        lines += _md_table(["disease", *(gd.LABEL[m] for m in order)],
                           [[c, *(_f(s.loc[(c, m, scheme), "auc_mean"])
                                  if (c, m, scheme) in s.index else "—" for m in order)]
                            for c in cats])
        lines.append("")
    lines += ["### The pre-stated test", "",
              "A graph model wins only if its LOPO AUC beats the best of logistic regression, "
              "gradient boosting and its own size-matched MLP by more than its fold-to-fold "
              "sd.", ""]
    lines += _md_table(["disease", "graph model", "LOPO AUC", "best rival", "rival AUC",
                        "margin", "fold sd", "verdict"], win_rows)

    per_seed = (disease_cv[disease_cv["model"].isin(["gcn", "gcn_shuffled"])]
                .groupby(["disease_category", "cv_scheme", "seed", "model"])["auc"].mean()
                .unstack("model"))
    tree = ((per_seed["gcn"] - per_seed["gcn_shuffled"])
            .groupby(["disease_category", "cv_scheme"]).agg(["mean", "std"]))
    lines += ["", "### Does the real taxonomy matter?", "",
              "GCN on the real tree minus GCN on a shuffled tree of the same shape (a different "
              "random tree per seed), paired by seed; counted if beyond 2× its seed sd.", ""]
    rows = []
    for (c, scheme), r in tree.iterrows():
        reading = ("real tree better" if r["mean"] > 2 * r["std"] else
                   "shuffled tree better" if r["mean"] < -2 * r["std"] else "within seed noise")
        rows.append([c, scheme.replace("_", " "), _pm(r["mean"], r["std"], "+.3f"), reading])
    lines += _md_table(["disease", "split", "real − shuffled", "reading"], rows)
    lines.append("")

    # ---- pretraining -------------------------------------------------------
    last = pretrain_history.iloc[-1]
    lines += ["## Pretraining — what the unlabelled samples add", "",
              "GraphSAGE pretrained without labels by masked edge reconstruction on every "
              "QC-passed sample outside the disease cohorts, so no sample later tested on was "
              f"seen; {int(last['epoch'])} epochs. Then fine-tuned whole, or frozen (only the "
              "final layer learns), and compared with the random-start GraphSAGE of Experiment "
              "B on the same folds and seeds.", "",
              "**Reconstruction on held-out samples' hidden edges**, against a baseline that "
              "ignores the sample (a taxon's prevalence; its mean weight):", ""]
    lines += _md_table(["measure", *recon.columns],
                       [[i, *(("yes" if v else "no") if isinstance(v, (bool, np.bool_))
                              else _f(v) for v in r)] for i, r in recon.iterrows()])
    lines += ["", "**Study identity:** share of a sample's 10 nearest neighbours from its own "
              "study (5,000 pretraining samples).", ""]
    lines += _md_table(["space", "same-study share"], [[i, _f(v)] for i, v in study.items()])
    lines += ["", "**Disease, LOPO:** pretrained minus random start, counted if beyond the "
              "random-start model's fold-to-fold sd.", ""]
    lines += _md_table(["disease", "variant", "random start", "pretrained", "Δ", "fold sd",
                        "verdict"], pt_rows)
    flattery = ("logreg", "gboost", "sage_g1", *gd.PRETRAINED_MODELS)
    lines += ["", "**Random k-fold minus LOPO** (how much a shared-study split flatters each "
              "model):", ""]
    lines += _md_table(["disease", *(gd.LABEL[m] for m in flattery)],
                       [[c, *(_f(s.loc[(c, m, rand), "auc_mean"] - s.loc[(c, m, lopo), "auc_mean"],
                                 "+.3f") for m in flattery)] for c in cats])

    lines += ["", "## Figures", "",
              "- `figures/phase7_project_compare.png` — Experiment A",
              "- `figures/phase7_leakage_gap.png` — Experiment C",
              "- `figures/phase7_disease_lopo.png` — Experiment B",
              "- `figures/phase7_pretrain_curve.png`, `figures/phase7_pretrain_lopo.png` — "
              "pretraining"]
    return "\n".join(lines) + "\n"
