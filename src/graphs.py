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
