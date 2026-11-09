"""Graph neural networks over the graphs `src/graphs.py` builds, the
controls they are judged against, and one training loop shared by all of
them.

Models, each answering a question the others cannot:

- `HeteroSAGE` -- the one-big-graph encoder. Inductive: it learns how to
  aggregate a neighbourhood rather than a fixed per-node embedding, so it
  can embed samples from a project it never saw. Messages are weighted by
  the edge weight (mean of `w_uv * h_u`), because an unweighted mean over a
  sample's taxa would reduce its composition to presence/absence.
- `HeteroGAT` -- the same graph with attention, so a trained model can be
  asked which taxa it attended to. The signed CLR value reaches the
  attention through the edge attributes.
- `TaxonomyGCN` -- one small taxonomy graph per sample. The skeleton is
  identical for every sample, so the normalised adjacency is valid for
  held-out graphs too and plain GCN is the simplest "does hierarchy help"
  test.
- `GraphAutoencoder` -- pretrains the `HeteroSAGE` encoder without labels by
  reconstructing masked abundance edges.

Controls: `matched_mlp` (a flat MLP with the GNN's parameter count -- if it
ties the GNN, the gain was capacity, not structure) and `shuffle_taxonomy`
(the real tree replaced by a random one with the same shape -- if that ties,
any hierarchy would have done). The logistic-regression and boosting
baselines are the existing `src.differential` ones.

`fit` / `predict` take any of the three batch kinds: neighbour-sampled
`HeteroData` (one big graph), a PyG `Batch` (per-sample graphs) or an
`(x, y)` tensor pair (flat MLP).
"""
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from src import graphs as gr

# Edge types that are never subsampled. A sample's composition arrives on its
# incoming abundance edges, so sampling a few of them would randomly delete
# most of the input. The taxonomy and sample -> project edges are tiny
# (one parent per node, one project per sample).
FULL_FANOUT_RELATIONS = ("rev_abundance", "child_of", "parent_of", "rev_from")


# ---------------------------------------------------------------------------
# One big graph: input encoding shared by both encoders
# ---------------------------------------------------------------------------

class _NodeInput(nn.Module):
    """Projects every node type's features to a common width. Taxon and
    taxon_internal nodes also get a learnable identity embedding -- their
    fixed features (rank, prevalence) don't tell two genera apart."""

    def __init__(self, node_counts, hidden, embed_dim):
        super().__init__()
        self.embed = nn.ModuleDict({
            nt: nn.Embedding(n, embed_dim) for nt, n in node_counts.items()
            if nt in ("taxon", "taxon_internal")})
        self.proj = nn.ModuleDict({nt: nn.LazyLinear(hidden) for nt in node_counts})

    def forward(self, data):
        out = {}
        for nt, proj in self.proj.items():
            x = data[nt].x
            if nt in self.embed:
                x = torch.cat([x, self.embed[nt](data[nt].node_id)], dim=1)
            out[nt] = proj(x)
        return out


def _node_counts(data):
    return {nt: data[nt].num_nodes for nt in data.node_types}


def _keep_missing(new, old):
    """`HeteroConv` drops node types that received no messages; carry their
    previous embedding forward so the next layer still has them as sources."""
    return {nt: new.get(nt, h) for nt, h in old.items()}


class HeteroSAGE(nn.Module):
    """GraphSAGE on a G1 / G2 / G3 `HeteroData`, edge-weighted mean
    aggregation (`GraphConv(aggr="mean")`: `W1 h_v + W2 mean(w_uv h_u)`),
    one convolution per edge type, a linear head on the sample nodes.
    `n_out = 1` for case/control, the number of projects for project
    prediction."""

    def __init__(self, data, n_out, *, layers=2, hidden=128, dropout=0.3, embed_dim=64):
        super().__init__()
        from torch_geometric.nn import GraphConv, HeteroConv

        self.input = _NodeInput(_node_counts(data), hidden, embed_dim)
        self.convs = nn.ModuleList([
            HeteroConv({et: GraphConv((hidden, hidden), hidden, aggr="mean")
                        for et in data.edge_types}, aggr="sum")
            for _ in range(layers)])
        self.dropout = dropout
        self.head = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def encode(self, data):
        h = self.input(data)
        weights = {et: data[et].edge_weight for et in data.edge_types
                   if "edge_weight" in data[et]}
        for i, conv in enumerate(self.convs):
            out = conv(h, data.edge_index_dict, edge_weight_dict=weights)
            if i < len(self.convs) - 1:
                out = {nt: F.dropout(F.relu(v), self.dropout, self.training)
                       for nt, v in out.items()}
            h = _keep_missing(out, h)
        return h

    def forward(self, data):
        return self.head(self.encode(data)["sample"])


class HeteroGAT(nn.Module):
    """GAT on a G1 / G2 / G3 `HeteroData`. Layer 1 concatenates its heads,
    the last layer averages them. Each edge type's attributes feed its
    attention (`edge_dim` read from the graph: 3 for abundance, 6 for
    taxonomy, 1 for k-NN distance). Bipartite attention has no self-loops,
    so a linear skip keeps each node's own state."""

    def __init__(self, data, n_out, *, layers=2, heads=4, hidden=32, dropout=0.3, embed_dim=64):
        super().__init__()
        from torch_geometric.nn import GATConv, HeteroConv

        width = heads * hidden
        self.input = _NodeInput(_node_counts(data), width, embed_dim)
        self.convs, self.skips = nn.ModuleList(), nn.ModuleList()
        for i in range(layers):
            last = i == layers - 1
            out_dim = hidden if last else width
            self.convs.append(HeteroConv({
                et: GATConv((width, width), hidden, heads=heads, concat=not last,
                            dropout=dropout, add_self_loops=False,
                            edge_dim=(data[et].edge_attr.shape[1]
                                      if "edge_attr" in data[et] else None))
                for et in data.edge_types}, aggr="sum"))
            self.skips.append(nn.ModuleDict({nt: nn.Linear(width, out_dim)
                                             for nt in data.node_types}))
        self.dropout = dropout
        self.head = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def encode(self, data):
        h = self.input(data)
        attrs = {et: data[et].edge_attr for et in data.edge_types if "edge_attr" in data[et]}
        for i, (conv, skip) in enumerate(zip(self.convs, self.skips)):
            out = conv(h, data.edge_index_dict, edge_attr_dict=attrs)
            out = {nt: v + skip[nt](h[nt]) for nt, v in out.items()}
            if i < len(self.convs) - 1:
                out = {nt: F.dropout(F.elu(v), self.dropout, self.training)
                       for nt, v in out.items()}
                h = _keep_missing(out, h)
            else:
                h = out
        return h

    def forward(self, data):
        return self.head(self.encode(data)["sample"])


# ---------------------------------------------------------------------------
# One graph per sample
# ---------------------------------------------------------------------------

class TaxonomyGCN(nn.Module):
    """GCN over the shared taxonomy skeleton, then a mean + max + attention
    readout over the leaves and a 2-layer MLP head. The skeleton comes from
    each batch, so the shuffled-taxonomy control is just a dataset built on
    the shuffled tree.

    Every skeleton node has a learnable identity embedding (node *i* is the
    same taxon in every graph). Without it GCN is blind to *which* leaf a CLR
    value sits on -- the only cues would be prevalence and tree position.

    Each node also gets `clr * slope[i]`, a second learnable embedding scaled
    by its own CLR value, so every taxon's value points in its own learned
    direction. After mean pooling this alone is a linear model over taxa --
    the per-taxon weight logistic regression has by construction. Without
    it a per-taxon effect can only be built indirectly through ReLU gates,
    and on a 200-sample overfit check the model sat at chance for 40 epochs;
    a GCN that loses to the flat baselines for that reason says nothing
    about whether the hierarchy helps.

    The attention pool's gate scores are the per-taxon interpretability hook.
    """

    def __init__(self, n_nodes, n_out, *, in_features=len(gr.LEAF_FEATURES),
                 layers=3, hidden=128, dropout=0.3, embed_dim=64):
        super().__init__()
        from torch_geometric.nn import GCNConv
        from torch_geometric.nn.aggr import AttentionalAggregation

        self.embed = nn.Embedding(n_nodes, embed_dim)
        self.slope = nn.Embedding(n_nodes, embed_dim)
        dims = [in_features + 2 * embed_dim] + [hidden] * layers
        self.convs = nn.ModuleList([GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.attn_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_out))
        self.dropout = dropout
        self.n_nodes = n_nodes

    def forward(self, batch):
        from torch_geometric.nn import global_max_pool, global_mean_pool

        g = batch.num_graphs
        pos = torch.arange(batch.num_nodes) - batch.ptr[batch.batch]
        clr = batch.x[:, gr.LEAF_FEATURES.index("clr"), None]  # 0 on internal nodes
        h = torch.cat([batch.x, self.embed(pos), clr * self.slope(pos)], dim=1)
        for conv in self.convs:
            h = F.dropout(F.relu(conv(h, batch.edge_index)), self.dropout, self.training)
        leaf, idx = h[batch.is_leaf], batch.batch[batch.is_leaf]
        pooled = torch.cat([global_mean_pool(leaf, idx, size=g),
                            global_max_pool(leaf, idx, size=g),
                            self.attn_pool(leaf, idx, dim_size=g)], dim=1)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Graph autoencoder (label-free pretraining of the HeteroSAGE encoder)
# ---------------------------------------------------------------------------

ABUNDANCE = ("sample", "abundance", "taxon")
REV_ABUNDANCE = ("taxon", "rev_abundance", "sample")


class GraphAutoencoder(nn.Module):
    """Encoder = a `HeteroSAGE` (its head unused); decoder = two small MLPs
    on `[h_sample || h_taxon]`, one scoring whether the edge exists, one
    predicting its weight. After pretraining, `encoder` is handed to the
    disease model."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        h = encoder.hidden
        self.exists = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
        self.weight = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, data, pairs):
        h = self.encoder.encode(data)
        z = torch.cat([h["sample"][pairs[0]], h["taxon"][pairs[1]]], dim=1)
        return self.exists(z).squeeze(-1), self.weight(z).squeeze(-1)


def mask_abundance_edges(data, mask_frac, generator):
    """Split the abundance edges: `mask_frac` of them are removed from the
    graph (both directions) and become reconstruction targets -- a model
    that can still see an edge would just copy it. Returns
    (graph without the masked edges, masked edge_index, masked weights)."""
    e = data[ABUNDANCE].edge_index.shape[1]
    perm = torch.randperm(e, generator=generator)
    n_mask = max(1, int(round(mask_frac * e)))
    masked, kept = perm[:n_mask], perm[n_mask:]

    out = copy.copy(data)
    for et in (ABUNDANCE, REV_ABUNDANCE):
        store = data[et]
        out[et].edge_index = store.edge_index[:, kept]
        out[et].edge_weight = store.edge_weight[kept]
        out[et].edge_attr = store.edge_attr[kept]
    return out, data[ABUNDANCE].edge_index[:, masked], data[ABUNDANCE].edge_weight[masked]


def sample_negative_pairs(edge_index, n_samples, n_taxa, n, generator):
    """`n` random (sample, taxon) pairs that are *not* abundance edges. The
    graph is ~13.5% dense, so uniform pairs collide with real edges often
    enough that they must be filtered rather than assumed absent."""
    existing = edge_index[0] * n_taxa + edge_index[1]
    out = []
    while sum(o.shape[1] for o in out) < n:
        s = torch.randint(n_samples, (2 * n,), generator=generator)
        t = torch.randint(n_taxa, (2 * n,), generator=generator)
        keep = ~torch.isin(s * n_taxa + t, existing)
        out.append(torch.stack([s[keep], t[keep]]))
    return torch.cat(out, dim=1)[:, :n]


def gae_loss(model, data, *, mask_frac, neg_ratio, generator):
    """One reconstruction step on `data`: mask edges, encode the rest, and
    score MSE on the masked edges' weights + BCE on masked edges vs absent
    pairs. Returns (loss, existence logits, existence labels)."""
    graph, pos, pos_w = mask_abundance_edges(data, mask_frac, generator)
    neg = sample_negative_pairs(data[ABUNDANCE].edge_index, data["sample"].num_nodes,
                                data["taxon"].num_nodes, int(neg_ratio * pos.shape[1]),
                                generator)
    pairs = torch.cat([pos, neg], dim=1)
    labels = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])])
    exist_logit, weight_pred = model(graph, pairs)
    loss = (F.mse_loss(weight_pred[:pos.shape[1]], pos_w)
            + F.binary_cross_entropy_with_logits(exist_logit, labels))
    return loss, exist_logit, labels


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def matched_mlp(in_features, n_out, target_params, *, tolerance=0.10, dropout=0.3):
    """Two equal-width hidden layers, width solved so the parameter count
    lands on `target_params`:
    in*h + h + h*h + h + h*n_out + n_out = target  (a quadratic in h)."""
    b = in_features + n_out + 2
    h = int(round((-b + np.sqrt(b * b + 4 * (target_params - n_out))) / 2))
    h = max(h, 1)
    model = nn.Sequential(nn.Linear(in_features, h), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(h, h), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(h, n_out))
    got = count_parameters(model)
    if abs(got - target_params) > tolerance * target_params:
        raise ValueError(f"matched MLP has {got:,} params, target {target_params:,}")
    return model


def shuffle_taxonomy(taxonomy, seed):
    """Rewire the tree at random while keeping its shape: at every rank, the
    children's parent pointers are permuted among themselves, so each parent
    keeps its number of children and every leaf still reaches a kingdom --
    only *which* taxa share a parent changes. Same output shape as
    `graphs.build_taxonomy_edges`."""
    _, node_index, rank_of_node = taxonomy
    rng = np.random.default_rng(seed)
    node_index = node_index.copy()
    parent = node_index["parent"].to_numpy().copy()
    for r in range(1, len(gr.RANKS)):
        kids = np.flatnonzero((node_index["rank"].to_numpy() == r) & (parent >= 0))
        parent[kids] = rng.permutation(parent[kids])
    node_index["parent"] = parent
    child = np.flatnonzero(parent >= 0)
    edge_index = torch.tensor(np.vstack([child, parent[child]]), dtype=torch.long)
    return edge_index, node_index, rank_of_node


# ---------------------------------------------------------------------------
# Data plumbing for the one-big-graph models
# ---------------------------------------------------------------------------

def neighbor_fanout(edge_types, neighbor_sampling):
    """Per-edge-type neighbour fan-out for `NeighborLoader`: the configured
    fan-out per hop, except for `FULL_FANOUT_RELATIONS`, which are taken
    whole (-1)."""
    return {et: ([-1] * len(neighbor_sampling) if et[1] in FULL_FANOUT_RELATIONS
                 else list(neighbor_sampling))
            for et in edge_types}


def sample_loader(data, sample_idx, *, batch_size, neighbor_sampling, shuffle):
    """Mini-batches seeded on `sample_idx` sample nodes. The first
    `batch['sample'].batch_size` sample rows of each batch are the seeds."""
    from torch_geometric.loader import NeighborLoader

    return NeighborLoader(
        data, input_nodes=("sample", torch.as_tensor(sample_idx, dtype=torch.long)),
        num_neighbors=neighbor_fanout(data.edge_types, neighbor_sampling),
        batch_size=batch_size, shuffle=shuffle)


def restrict_to_samples(data, sample_idx):
    """The graph induced on a subset of sample nodes, every taxon and
    taxonomy node kept. Training on this graph keeps held-out samples (and
    their abundance edges) entirely out of what the model sees -- the
    inductive setting leave-one-project-out needs."""
    return data.subgraph({"sample": torch.as_tensor(sample_idx, dtype=torch.long)})


def scale_sample_features(data, train_idx):
    """Copy of `data` with sample features standardised by mean / sd of the
    training rows only."""
    out = copy.copy(data)
    x = data["sample"].x
    mu = x[train_idx].mean(0)
    sd = x[train_idx].std(0).clamp_min(1e-6)
    out["sample"].x = (x - mu) / sd
    return out


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------

def _logits_and_targets(model, batch):
    from torch_geometric.data import HeteroData

    if isinstance(batch, (tuple, list)):
        return model(batch[0]), batch[1]
    if isinstance(batch, HeteroData):
        n = batch["sample"].batch_size
        return model(batch)[:n], batch["sample"].y[:n]
    return model(batch), batch.y


def _loss_fn(n_out, pos_weight=None):
    if n_out == 1:
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return lambda logits, y: bce(logits.squeeze(-1), y.float())
    ce = nn.CrossEntropyLoss()
    return lambda logits, y: ce(logits, y.long())


def fit(model, train_loader, *, n_out, epochs, lr, weight_decay, seed,
        pos_weight=None, val_loader=None, patience=None):
    """Adam + BCE (`n_out == 1`) or cross-entropy, fixed epoch budget.
    `pos_weight` = n_negative / n_positive reweights the binary loss the way
    `class_weight="balanced"` does for the logistic-regression baseline.
    With `val_loader` and `patience`, stops once validation loss hasn't
    improved for `patience` epochs and restores the best weights.
    Returns (model, per-epoch history frame)."""
    torch.manual_seed(seed)
    loss_fn = _loss_fn(n_out, None if pos_weight is None
                       else torch.tensor(float(pos_weight)))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    history, best, best_state, stale = [], np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, y = _logits_and_targets(model, batch)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(y)
            n += len(y)
        row = dict(epoch=epoch, train_loss=total / n)

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                parts = [_logits_and_targets(model, b) for b in val_loader]
                val = float(loss_fn(torch.cat([p[0] for p in parts]),
                                    torch.cat([p[1] for p in parts])))
            row["val_loss"] = val
            if val < best:
                best, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
        history.append(row)
        if patience is not None and val_loader is not None and stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict(model, loader):
    """(scores, targets) over a loader, in loader order. Binary models
    return P(case); multi-class models return the softmax matrix."""
    model.eval()
    parts = [_logits_and_targets(model, b) for b in loader]
    logits = torch.cat([p[0] for p in parts])
    y = torch.cat([p[1] for p in parts]).numpy()
    if logits.shape[1] == 1:
        return torch.sigmoid(logits.squeeze(-1)).numpy(), y
    return torch.softmax(logits, dim=1).numpy(), y


def tensor_loader(X, y, *, batch_size, shuffle):
    """`(x, y)` mini-batches for the flat MLP control."""
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.as_tensor(np.asarray(X), dtype=torch.float32),
                       torch.as_tensor(np.asarray(y)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
