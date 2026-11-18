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
import time

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
    return (keep_abundance_edges(data, kept),
            data[ABUNDANCE].edge_index[:, masked], data[ABUNDANCE].edge_weight[masked])


def keep_abundance_edges(data, kept):
    """Copy of `data` with only the abundance edges `kept` (indices or a
    boolean mask), in both directions. The two directions are stored in the
    same order, so one index selects an edge and its reverse."""
    out = copy.copy(data)
    for et in (ABUNDANCE, REV_ABUNDANCE):
        store = data[et]
        for key in ("edge_weight", "edge_attr"):
            out[et][key] = store[key][kept]
        out[et].edge_index = store.edge_index[:, kept]
    return out


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


def _is_edge(keys_sorted, keys):
    pos = torch.searchsorted(keys_sorted, keys).clamp_max(len(keys_sorted) - 1)
    return keys_sorted[pos] == keys


def matched_negatives(pos, n_taxa, keys_sorted, generator, *, ratio=1, max_tries=100):
    """`ratio` absent pairs per positive (sample, taxon) pair, each with the
    positive's own sample and a random taxon that sample does not carry --
    so a decoder cannot separate the classes by sample alone (e.g. by how
    many taxa it has). `keys_sorted` = sorted `sample * n_taxa + taxon` of
    every abundance edge, masked ones included."""
    s = pos[0].repeat(ratio)
    t = torch.randint(n_taxa, (len(s),), generator=generator)
    bad = _is_edge(keys_sorted, s * n_taxa + t)
    for _ in range(max_tries):
        if not bad.any():
            break
        t[bad] = torch.randint(n_taxa, (int(bad.sum()),), generator=generator)
        bad = _is_edge(keys_sorted, s * n_taxa + t)
    return torch.stack([s[~bad], t[~bad]])


def _local_pairs(batch, pairs, n_samples, n_taxa):
    """Global (sample, taxon) pairs whose sample is a seed of `batch` and
    whose taxon is in it, re-indexed into the batch. Returns (local pairs,
    boolean mask over `pairs`)."""
    def lookup(n_id, size):
        loc = torch.full((size,), -1, dtype=torch.long)
        loc[n_id] = torch.arange(len(n_id))
        return loc

    s = lookup(batch["sample"].n_id[: batch["sample"].batch_size], n_samples)[pairs[0]]
    t = lookup(batch["taxon"].n_id, n_taxa)[pairs[1]]
    keep = (s >= 0) & (t >= 0)
    return torch.stack([s[keep], t[keep]]), keep


def _reconstruction_pass(model, loader, pos, pos_w, neg, n_samples, n_taxa, optimizer=None):
    """One pass over `loader`: each batch is scored on the masked positives
    and negatives whose sample is one of its seeds. Trains when `optimizer`
    is given. Returns (mean loss, existence logits, existence labels,
    weight predictions, weight targets)."""
    totals, n, out = 0.0, 0, ([], [], [], [])
    for batch in loader:
        p, keep_p = _local_pairs(batch, pos, n_samples, n_taxa)
        q, _ = _local_pairs(batch, neg, n_samples, n_taxa)
        if p.shape[1] == 0:
            continue
        exist_logit, weight_pred = model(batch, torch.cat([p, q], dim=1))
        labels = torch.cat([torch.ones(p.shape[1]), torch.zeros(q.shape[1])])
        target = pos_w[keep_p]
        loss = (F.mse_loss(weight_pred[: p.shape[1]], target)
                + F.binary_cross_entropy_with_logits(exist_logit, labels))
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        totals += float(loss.detach()) * len(labels)
        n += len(labels)
        for lst, v in zip(out, (exist_logit, labels, weight_pred[: p.shape[1]], target)):
            lst.append(v.detach())
    return (totals / max(n, 1), *(torch.cat(v) for v in out))


def pretrain_graph_autoencoder(graph, *, encoder_kwargs, epochs, batch_size, lr,
                               weight_decay, mask_frac, neg_ratio, neighbor_sampling,
                               eval_frac, seed, log=print):
    """Pretrain a `HeteroSAGE` encoder on `graph` (a G1 without labels) by
    masked abundance-edge reconstruction.

    Each epoch hides a fresh `mask_frac` of the training samples' abundance
    edges; for every hidden edge the model must say that it exists (against
    one matched absent pair, `matched_negatives`) and predict its weight.

    A fixed `eval_frac` of the samples are evaluation samples: `mask_frac` of
    their edges are removed for the whole run and never trained on, and they
    are never batch seeds in training. Their remaining edges stay in the
    graph as context, as a new sample's would. They are scored after every
    epoch, beside two baselines that know nothing about the sample: a
    taxon's prevalence (existence AUC) and its mean weight (weight MSE).

    Returns (encoder, per-epoch history frame, baseline dict)."""
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    n_s, n_t = graph["sample"].num_nodes, graph["taxon"].num_nodes
    ei, w = graph[ABUNDANCE].edge_index, graph[ABUNDANCE].edge_weight
    keys_sorted = (ei[0] * n_t + ei[1]).sort().values

    n_eval = max(1, int(round(eval_frac * n_s)))
    is_eval = torch.zeros(n_s, dtype=torch.bool)
    is_eval[torch.randperm(n_s, generator=gen)[:n_eval]] = True
    held = is_eval[ei[0]] & (torch.rand(ei.shape[1], generator=gen) < mask_frac)
    eval_pos, eval_w = ei[:, held], w[held]
    eval_neg = matched_negatives(eval_pos, n_t, keys_sorted, gen, ratio=neg_ratio)
    base = keep_abundance_edges(graph, ~held)
    train_samples = torch.nonzero(~is_eval).squeeze(1)
    eval_samples = torch.nonzero(is_eval).squeeze(1)

    # sample-blind baselines, fit on the edges the model can see
    b_ei, b_w = base[ABUNDANCE].edge_index, base[ABUNDANCE].edge_weight
    prevalence = torch.bincount(b_ei[1], minlength=n_t).float() / n_s
    mean_w = (torch.zeros(n_t).index_add_(0, b_ei[1], b_w)
              / torch.bincount(b_ei[1], minlength=n_t).clamp_min(1))
    eval_pairs = torch.cat([eval_pos, eval_neg], dim=1)
    eval_labels = torch.cat([torch.ones(eval_pos.shape[1]), torch.zeros(eval_neg.shape[1])])
    baselines = dict(
        exist_auc_prevalence=float(roc_auc_score(eval_labels, prevalence[eval_pairs[1]])),
        weight_mse_taxon_mean=float(F.mse_loss(mean_w[eval_pos[1]], eval_w)),
        n_eval_samples=n_eval, n_eval_edges=int(eval_pos.shape[1]))

    model = GraphAutoencoder(HeteroSAGE(base, 1, **encoder_kwargs))
    base_idx = torch.nonzero(~held).squeeze(1)
    trainable = ~is_eval[ei[0, base_idx]]
    eval_loader = sample_loader(base, eval_samples, batch_size=1024,
                                neighbor_sampling=neighbor_sampling, shuffle=False)
    optimizer, history = None, []
    for epoch in range(epochs):
        t0 = time.time()
        cand = base_idx[trainable]
        hide = torch.rand(len(cand), generator=gen) < mask_frac
        keep = torch.ones(ei.shape[1], dtype=torch.bool)
        keep[held] = False
        keep[cand[hide]] = False
        pos, pos_w = ei[:, cand[hide]], w[cand[hide]]
        neg = matched_negatives(pos, n_t, keys_sorted, gen, ratio=neg_ratio)
        loader = sample_loader(keep_abundance_edges(graph, keep), train_samples,
                               batch_size=batch_size, neighbor_sampling=neighbor_sampling,
                               shuffle=True)
        if optimizer is None:  # lazy input layers get their size on the first batch
            with torch.no_grad():
                b = next(iter(loader))
                model(b, torch.zeros(2, 1, dtype=torch.long))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        train_loss = _reconstruction_pass(model, loader, pos, pos_w, neg, n_s, n_t, optimizer)[0]

        model.eval()
        with torch.no_grad():
            eval_loss, logit, lab, wp, wt = _reconstruction_pass(
                model, eval_loader, eval_pos, eval_w, eval_neg, n_s, n_t)
        row = dict(epoch=epoch + 1, train_loss=train_loss, eval_loss=eval_loss,
                   eval_exist_auc=float(roc_auc_score(lab, logit)),
                   eval_weight_mse=float(F.mse_loss(wp, wt)),
                   minutes=(time.time() - t0) / 60)
        history.append(row)
        log(f"epoch {row['epoch']:2d}/{epochs}  train loss {train_loss:.4f}  "
            f"held-out: loss {eval_loss:.4f}, existence AUC {row['eval_exist_auc']:.3f}, "
            f"weight MSE {row['eval_weight_mse']:.3f}  [{row['minutes']:.1f} min]")
    return model.encoder, pd.DataFrame(history), baselines


@torch.no_grad()
def embed_samples(encoder, graph, sample_idx, *, neighbor_sampling, batch_size=2048):
    """Encoder output for the given sample nodes, in the order given."""
    encoder.eval()
    parts = []
    for batch in sample_loader(graph, sample_idx, batch_size=batch_size,
                               neighbor_sampling=neighbor_sampling, shuffle=False):
        parts.append(encoder.encode(batch)["sample"][: batch["sample"].batch_size])
    return torch.cat(parts).numpy()


def same_group_neighbour_rate(x, groups, k=10):
    """Share of each row's `k` nearest neighbours (Euclidean) that carry its
    own group, averaged over rows."""
    from sklearn.neighbors import NearestNeighbors

    groups = np.asarray(groups)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x, return_distance=False)[:, 1:]
    return float((groups[idx] == groups[:, None]).mean())


def encoder_state(encoder):
    """The pretrained encoder's weights without its (untrained) head."""
    return {k: v for k, v in encoder.state_dict().items() if not k.startswith("head.")}


def load_pretrained(model, state, *, frozen):
    """Load `encoder_state` weights into a fresh `HeteroSAGE`; only the head
    stays newly initialised. `frozen` stops the encoder from training, so
    only the head learns (`fit` skips parameters without gradients)."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("head.") for k in missing):
        raise ValueError(f"encoder mismatch: missing {missing}, unexpected {unexpected}")
    if frozen:
        for name, p in model.named_parameters():
            if not name.startswith("head."):
                p.requires_grad_(False)
    return model


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


def receivers_only(data, train_idx):
    """Copy of `data` in which only training samples send messages: every
    edge whose source is a non-training sample is dropped. Held-out samples
    keep their incoming edges (their own taxa, their project), so they can
    still be embedded, but nothing about them reaches any other node -- a
    held-out prediction depends on its own composition and the training
    samples only. One such graph serves both training and evaluation."""
    out = copy.copy(data)
    is_train = torch.zeros(data["sample"].num_nodes, dtype=torch.bool)
    is_train[torch.as_tensor(train_idx, dtype=torch.long)] = True
    for et in data.edge_types:
        if et[0] != "sample":
            continue
        keep = is_train[data[et].edge_index[0]]
        for key, value in data[et].items():
            if torch.is_tensor(value) and value.dim() > 0 and key != "edge_index" \
                    and value.shape[0] == keep.shape[0]:
                out[et][key] = value[keep]
        out[et].edge_index = data[et].edge_index[:, keep]
    return out


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


# ---------------------------------------------------------------------------
# Project-prediction cross-validation (one big graph)
# ---------------------------------------------------------------------------

def project_folds(project_labels, seed, *, n_splits=5, min_project_size=25):
    """The exact folds `variance.run_project_cv` uses: drop projects below
    `min_project_size`, label-encode, `StratifiedKFold(shuffle, seed)`. The
    split depends only on the labels, so a GNN scored on these folds is
    scored on the same test samples as the committed flat baselines.
    Returns (keep mask, encoded y over kept rows, class names, folds)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    labels = np.asarray(project_labels)
    vc = pd.Series(labels).value_counts()
    keep = np.isin(labels, vc[vc >= min_project_size].index)
    le = LabelEncoder()
    y = le.fit_transform(labels[keep])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return keep, y, le.classes_, list(skf.split(np.zeros(len(y)), y))


def multiclass_scores(y_true, prob, n_classes):
    """accuracy / balanced accuracy / top-5, as `variance.run_project_cv`
    reports them."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 top_k_accuracy_score)

    pred = prob.argmax(axis=1)
    return dict(accuracy=accuracy_score(y_true, pred),
                balanced_accuracy=balanced_accuracy_score(y_true, pred),
                top5_accuracy=top_k_accuracy_score(y_true, prob, k=5,
                                                   labels=np.arange(n_classes)))


def graph_project_cv(make_graph, make_model, folds, n_classes, seed, *, epochs,
                     batch_size, lr, weight_decay, neighbor_sampling, progress=True):
    """Cross-validated project prediction with a one-big-graph model.

    Per fold, `make_graph(train_idx)` returns the fold's `HeteroData`
    (statistics fit on training rows, held-out samples as receivers only,
    `sample.y` set) and `make_model(graph)` a fresh model; training and
    scoring both run on that graph, seeded on the fold's own samples.
    With `progress`, prints one line per finished fold -- a fold takes
    minutes, and a plain line survives logging where a progress bar doesn't.
    Returns one row per fold."""
    rows = []
    for fold, (tr, te) in enumerate(folds):
        t0 = time.time()
        graph = make_graph(tr)
        torch.manual_seed(seed + fold)
        model = make_model(graph)
        train_loader = sample_loader(graph, tr, batch_size=batch_size,
                                     neighbor_sampling=neighbor_sampling, shuffle=True)
        fit(model, train_loader, n_out=n_classes, epochs=epochs, lr=lr,
            weight_decay=weight_decay, seed=seed + fold)
        prob, y_te = predict(model, sample_loader(graph, te, batch_size=2048,
                                                  neighbor_sampling=neighbor_sampling,
                                                  shuffle=False))
        rows.append(dict(fold=fold, **multiclass_scores(y_te, prob, n_classes),
                         n_train=len(tr), n_test=len(te), n_params=count_parameters(model),
                         seconds=round(time.time() - t0, 1)))
        if progress:
            r = rows[-1]
            print(f"  fold {fold + 1}/{len(folds)}: accuracy {r['accuracy']:.4f}  "
                  f"balanced {r['balanced_accuracy']:.4f}  top-5 {r['top5_accuracy']:.4f}  "
                  f"[{r['seconds'] / 60:.1f} min]", flush=True)
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# Disease cross-validation (one big graph)
# ---------------------------------------------------------------------------

def disease_folds(y, groups, seed, *, n_splits=5):
    """The two split schemes `differential.batch_leakage_cv` uses, built the
    same way so each fold holds the same test samples as the committed
    flat baselines: `leave_project_out` (one fold per project) and
    `random_kfold` (stratified, k = min(n_splits, smaller class count))."""
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

    y = np.asarray(y)
    zeros = np.zeros(len(y))
    schemes = {"leave_project_out": list(LeaveOneGroupOut().split(zeros, y, groups))}
    k = int(min(n_splits, np.bincount(y).min()))
    if k >= 2:
        schemes["random_kfold"] = list(
            StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(zeros, y))
    return schemes


def binary_cv(run_fold, schemes, y, groups):
    """The fold loop of `graph_disease_cv` for any model: `run_fold(train_idx,
    test_idx, fold)` trains a fresh model and returns P(case) for the test
    rows. Same skipping rule and output columns, so per-sample graphs, flat
    MLPs and one-big-graph models land in one comparable table."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    y = np.asarray(y)
    groups = np.asarray(groups)
    rows = []
    for scheme, folds in schemes.items():
        for fold, (tr, te) in enumerate(folds):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            t0 = time.time()
            prob = np.asarray(run_fold(tr, te, fold))
            rows.append(dict(
                cv_scheme=scheme, fold=fold, auc=float(roc_auc_score(y[te], prob)),
                balanced_accuracy=float(balanced_accuracy_score(y[te], (prob >= 0.5).astype(int))),
                n_train=len(tr), n_test=len(te), n_test_pos=int(y[te].sum()),
                test_project=str(groups[te][0]) if scheme == "leave_project_out" else "",
                seconds=round(time.time() - t0, 1)))
    return pd.DataFrame(rows)


# How a fold's graph is used. Strict to leaky:
# - inductive: held-out samples only receive (`receivers_only`).
# - transductive: held-out samples stay in the graph and send messages; only
#   their labels are hidden. Their *features* reach other nodes, never labels.
# - transductive_labels: as transductive, plus every training sample's label
#   is an input feature (the "label as feature" practice behind label
#   propagation-style GNNs), so a held-out sample can read its neighbours'
#   labels.
PROTOCOLS = ("inductive", "transductive", "transductive_labels")


def with_label_features(x, y, known_idx):
    """`x` with two columns appended: `label_known` (1 for rows in
    `known_idx`) and `label_value` (their label, 0 elsewhere). Rows outside
    `known_idx` look exactly like unlabelled samples."""
    known = torch.zeros(x.shape[0], 1)
    value = torch.zeros(x.shape[0], 1)
    idx = torch.as_tensor(known_idx, dtype=torch.long)
    known[idx] = 1.0
    value[idx, 0] = torch.as_tensor(np.asarray(y)[known_idx], dtype=torch.float32)
    return torch.cat([x, known, value], dim=1)


def fit_label_trick(model, graph, train_idx, y, *, epochs, batch_size, lr, weight_decay,
                    neighbor_sampling, seed, pos_weight=None):
    """Training for `transductive_labels`. A sample must never see its own
    label, so each epoch the training samples are split at random in half:
    one half supplies its labels as input features, the other half are the
    loss targets with their label columns blanked. `graph["sample"].x` is the
    label-free features; it is restored before returning."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    base_x = graph["sample"].x
    loss_fn = _loss_fn(1, None if pos_weight is None else torch.tensor(float(pos_weight)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_idx = np.asarray(train_idx)
    for _ in range(epochs):
        perm = rng.permutation(train_idx)
        known, target = perm[: len(perm) // 2], perm[len(perm) // 2:]
        graph["sample"].x = with_label_features(base_x, y, known)
        model.train()
        for batch in sample_loader(graph, target, batch_size=batch_size,
                                   neighbor_sampling=neighbor_sampling, shuffle=True):
            optimizer.zero_grad()
            logits, yb = _logits_and_targets(model, batch)
            loss_fn(logits, yb).backward()
            optimizer.step()
    graph["sample"].x = base_x
    return model


def graph_disease_cv(make_graph, make_model, schemes, y, groups, seed, *, epochs,
                     batch_size, lr, weight_decay, neighbor_sampling, protocol="inductive"):
    """Case/control classification with a one-big-graph model, per fold of
    every scheme in `schemes` (from `disease_folds`). `make_graph(train_idx)`
    returns the fold's graph with statistics fit on the training rows;
    `protocol` (one of `PROTOCOLS`) decides what held-out samples may do in
    it. `make_model(graph)` returns a fresh binary model. The loss is
    reweighted by n_control / n_case of the training rows, as
    `class_weight="balanced"` is for the logistic-regression baseline.
    Folds whose test set holds a single class are skipped, as in the flat
    baseline. Returns one row per (scheme, fold): AUC, balanced accuracy at
    0.5, sizes, test project."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}")
    y = np.asarray(y)
    groups = np.asarray(groups)
    rows = []
    for scheme, folds in schemes.items():
        for fold, (tr, te) in enumerate(folds):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            t0 = time.time()
            graph = make_graph(tr)
            if protocol == "inductive":
                graph = receivers_only(graph, tr)
            if protocol == "transductive_labels":
                # label columns present (all unknown) so the model is built
                # for the input width it will be trained and scored with
                base_x = graph["sample"].x
                graph["sample"].x = with_label_features(base_x, y, [])
            torch.manual_seed(seed + fold)
            model = make_model(graph)
            pos_weight = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
            train_kw = dict(epochs=epochs, lr=lr, weight_decay=weight_decay,
                            seed=seed + fold, pos_weight=pos_weight)
            if protocol == "transductive_labels":
                graph["sample"].x = base_x
                fit_label_trick(model, graph, tr, y, batch_size=batch_size,
                                neighbor_sampling=neighbor_sampling, **train_kw)
                # scoring: every training label is known, no held-out one is
                graph["sample"].x = with_label_features(base_x, y, tr)
            else:
                fit(model, sample_loader(graph, tr, batch_size=batch_size,
                                         neighbor_sampling=neighbor_sampling, shuffle=True),
                    n_out=1, **train_kw)
            prob, y_te = predict(model, sample_loader(graph, te, batch_size=2048,
                                                      neighbor_sampling=neighbor_sampling,
                                                      shuffle=False))
            rows.append(dict(
                cv_scheme=scheme, fold=fold, auc=float(roc_auc_score(y_te, prob)),
                balanced_accuracy=float(balanced_accuracy_score(y_te, (prob >= 0.5).astype(int))),
                n_train=len(tr), n_test=len(te), n_test_pos=int(y_te.sum()),
                test_project=str(groups[te][0]) if scheme == "leave_project_out" else "",
                seconds=round(time.time() - t0, 1)))
    return pd.DataFrame(rows)


# colour encodes a model's role, not its identity: blue = flat model, orange =
# graph model, hatched grey = the label is reachable inside the graph (a ceiling)
ROLE_STYLE = {
    "flat": dict(color="#2a78d6"),
    "graph": dict(color="#eb6834"),
    "label in graph": dict(color="#b5b4ae", hatch="///", edgecolor="#52514e"),
}


def plot_project_compare(summary, majority_baseline, *, committed=None):
    """One panel per metric, one horizontal bar per model (mean, +/- sd over
    folds), coloured by `role`. `summary` columns: model, role,
    <metric>_mean, <metric>_std for accuracy / balanced_accuracy /
    top5_accuracy. The majority-class rate is drawn on the accuracy panels;
    `committed` (model -> committed accuracy) adds a tick for each flat
    baseline's committed value, so a reproduction is visible at a glance."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from src import variance as var

    var._style()
    metrics = [("accuracy", "accuracy"), ("balanced_accuracy", "balanced accuracy"),
               ("top5_accuracy", "top-5 accuracy")]
    d = summary.iloc[::-1].reset_index(drop=True)  # first row on top
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.45 * len(d) + 1.8), sharey=True)
    for ax, (m, label) in zip(axes, metrics):
        for i, r in d.iterrows():
            ax.barh(i, r[f"{m}_mean"], height=0.62, xerr=r[f"{m}_std"],
                    error_kw=dict(ecolor="#52514e", lw=1, capsize=2),
                    linewidth=0.8, **ROLE_STYLE[r["role"]])
            ax.text(min(r[f"{m}_mean"] + r[f"{m}_std"] + 0.015, 0.99), i,
                    f"{r[f'{m}_mean']:.3f}", va="center", fontsize=8, color="#0b0b0b")
        if m == "accuracy" and committed:
            for i, r in d.iterrows():
                if r["model"] in committed:
                    ax.plot([committed[r["model"]]] * 2, [i - 0.38, i + 0.38],
                            color="#0b0b0b", lw=1.5)
        if m in ("accuracy", "balanced_accuracy"):
            ax.axvline(majority_baseline, color="#52514e", ls="--", lw=1)
        ax.set_xlim(0, 1.08)
        ax.set_title(label)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(d)))
    axes[0].set_yticklabels(d["model"])
    handles = [Patch(label=k, **v) for k, v in ROLE_STYLE.items() if k in set(d["role"])]
    handles.append(plt.Line2D([], [], color="#52514e", ls="--",
                              label=f"majority class ({majority_baseline:.3f})"))
    if committed:
        handles.append(plt.Line2D([], [], color="#0b0b0b", lw=1.5,
                                  label="committed Phase 4 value"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Which study did a sample come from? Flat models vs the graph",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig



LEAKAGE_STYLE = {
    "G1": dict(color="#2a78d6"),
    "G2": dict(color="#eb6834"),
    "G2_filtered": dict(color="#1baf7a"),
    "G3": dict(color="#b5b4ae", hatch="///", edgecolor="#52514e"),
}
LEAKAGE_LABEL = {
    "G1": "G1: samples + taxa + taxonomy",
    "G2": "G2: G1 + k-NN sample edges",
    "G2_filtered": "G2 with same-study edges deleted",
    "G3": "G3: G1 + study nodes",
}


PROTOCOL_TITLE = {
    "inductive": "inductive: held-out samples only receive",
    "transductive": "transductive: held-out samples also send features",
    "transductive_labels": "transductive + training labels as input",
}


def plot_leakage_gap(summary, flat=None):
    """Grouped bars of mean AUC per disease, one bar per graph variant; one
    column per CV scheme (leave-one-project-out first) and, when `summary`
    has a `protocol` column, one row per protocol. `summary` columns:
    disease_category, cv_scheme, graph, auc_mean, auc_seed_sd[, protocol].
    `flat` (optional: disease_category, cv_scheme, auc) draws the committed
    flat logistic-regression AUC as a tick over each disease group."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from src import variance as var

    var._style()
    if "protocol" not in summary.columns:
        summary = summary.assign(protocol="inductive")
    protocols = [p for p in PROTOCOLS if p in set(summary["protocol"])]
    schemes = [("leave_project_out", "leave-one-project-out (honest)"),
               ("random_kfold", "random k-fold (studies shared)")]
    graphs = [g for g in LEAKAGE_STYLE if g in set(summary["graph"])]
    cats = sorted(summary["disease_category"].unique())
    width = 0.8 / len(graphs)
    fig, axes = plt.subplots(len(protocols), 2, figsize=(14, 4.2 * len(protocols) + 0.6),
                             sharey=True, squeeze=False)
    for row, protocol in enumerate(protocols):
        for ax, (scheme, title) in zip(axes[row], schemes):
            d = summary[(summary["cv_scheme"] == scheme) & (summary["protocol"] == protocol)]
            d = d.set_index(["disease_category", "graph"])
            for j, g in enumerate(graphs):
                xs = np.arange(len(cats)) + (j - (len(graphs) - 1) / 2) * width
                vals = [d["auc_mean"].get((c, g), np.nan) for c in cats]
                errs = [d["auc_seed_sd"].get((c, g), 0.0) for c in cats]
                ax.bar(xs, vals, width * 0.92, yerr=errs, linewidth=0.8,
                       error_kw=dict(ecolor="#52514e", lw=1, capsize=2), **LEAKAGE_STYLE[g])
            if flat is not None:
                f = flat[flat["cv_scheme"] == scheme].set_index("disease_category")["auc"]
                for i, c in enumerate(cats):
                    if c in f.index:
                        ax.plot([i - 0.42, i + 0.42], [f[c]] * 2, color="#0b0b0b", lw=1.5)
            ax.axhline(0.5, color="#52514e", ls="--", lw=1)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats)
            ax.set_ylim(0.3, 1.0)
            ax.set_title(f"{title}\n{PROTOCOL_TITLE[protocol]}", fontsize=10)
            ax.grid(axis="x", visible=False)
        axes[row, 0].set_ylabel("AUC (mean over folds and seeds)")
    handles = [Patch(label=LEAKAGE_LABEL[g], **LEAKAGE_STYLE[g]) for g in graphs]
    handles.append(plt.Line2D([], [], color="#52514e", ls="--", label="chance"))
    if flat is not None:
        handles.append(plt.Line2D([], [], color="#0b0b0b", lw=1.5,
                                  label="committed flat logistic regression"))
    legend_h = 0.6 / fig.get_figheight()
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Does building a graph leak the study into a disease classifier?",
                 fontweight="bold")
    fig.tight_layout(rect=(0, legend_h, 1, 1))
    return fig
