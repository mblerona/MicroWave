"""Case/control classification with the graph models, against the flat
baselines and the controls that make the comparison honest.

Every model is trained and scored on exactly the folds of the committed
`reports/classifier_cv.csv` (leave-one-project-out and random k-fold), one
(disease, model, seed) at a time. Each such job is self-contained -- it
rebuilds its disease's data from the Parquet artifacts -- so jobs can run in
separate processes, and each writes its own small results file, so an
interrupted run resumes where it stopped.

Models (`MODELS`):

- `gcn` -- `TaxonomyGCN` on one taxonomy graph per sample. Cannot leak:
  the graphs are disjoint. The headline model.
- `gcn_shuffled` -- the same GCN on a taxonomy whose parent pointers are
  permuted within each rank (a different random tree per seed). Separates
  "the real taxonomy helps" from "any tree of that shape helps".
- `sage_g1`, `gat_g1` -- GraphSAGE / GAT on the one-big-graph G1, held-out
  samples as receivers only.
- `mlp_gcn`, `mlp_sage`, `mlp_gat` -- a flat MLP on the 419 CLR features with
  the parameter count of the named graph model. If it ties the graph model,
  the gain was capacity, not structure.

The flat logistic-regression and boosting baselines are
`differential.batch_leakage_cv`, run as committed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from src import differential as diff
from src import gnn
from src import graphs as gr
from src import variance as var

MODELS = ("gcn", "gcn_shuffled", "sage_g1", "gat_g1", "mlp_gcn", "mlp_sage", "mlp_gat")
FRAMING = {"gcn": "per-sample graph", "gcn_shuffled": "per-sample graph",
           "sage_g1": "one big graph", "gat_g1": "one big graph",
           "mlp_gcn": "flat", "mlp_sage": "flat", "mlp_gat": "flat"}


def run_path(run_dir, category, model, seed):
    return Path(run_dir) / f"{category}__{model}__{seed}.csv"


def category_data(root, params, category):
    """Everything one disease's jobs need: samples, labels, projects, CLR,
    abundance edges, presence, sample features, taxonomy, and the folds of
    the committed flat baselines (built from `params["seed"]`)."""
    root = Path(root)
    interim, processed = root / "data" / "interim", root / "data" / "processed"
    clr_path, rel_path = processed / "abund_clr.parquet", processed / "abund_relative.parquet"

    cohort = pd.read_parquet(processed / "cohort_within_project.parquet")
    frame = diff.build_analysis_frame(cohort, params["differential"]["category_groups"])
    sub = frame[frame["disease_category"] == category].drop_duplicates("sample").reset_index(drop=True)
    samples = sub["sample"].tolist()
    y = sub["y"].to_numpy(int)
    projects = sub["project"].to_numpy()

    taxa = [c for c in pq.ParquetFile(clr_path).schema.names if c != gr.SAMPLE_COL]
    taxonomy = gr.build_taxonomy_edges(pd.read_parquet(interim / "taxon_table.parquet"), taxa)
    clr_df, _ = diff.load_clr(clr_path, samples)
    clr_df = clr_df.loc[samples, taxa]
    bipartite = gr.build_bipartite_edges(rel_path, clr_path, samples, taxa)
    sx, names = gr.sample_node_features(pd.read_parquet(interim / "sample_depth.parquet"), samples)
    return dict(
        category=category, samples=samples, y=y, projects=projects, taxa=taxa,
        taxonomy=taxonomy, clr_df=clr_df, clr=clr_df.to_numpy(), bipartite=bipartite,
        presence=gr.presence_from_edges(bipartite[0], len(samples), len(taxa)),
        sample_x=sx, sample_feature_names=names,
        schemes=gnn.disease_folds(y, projects, params["seed"],
                                  n_splits=params["differential"]["batch_leakage"]["cv_folds"]))


def _g1_builder(d):
    def build(train_idx):
        g = gr.build_hetero_graph(
            "G1", d["samples"], projects=d["projects"], taxonomy=d["taxonomy"],
            bipartite=d["bipartite"], sample_x=d["sample_x"],
            sample_feature_names=d["sample_feature_names"], taxa_cols=d["taxa"],
            leaf_stats=gr.taxon_leaf_stats(d["clr"], d["presence"], train_idx), y=d["y"])
        return gnn.scale_sample_features(g, train_idx)
    return build


def _graph_model(gcfg, kind):
    emb = gcfg["taxon_embed_dim"]
    if kind == "sage_g1":
        cfg = {k: v for k, v in gcfg["models"]["graphsage"].items() if k != "aggr"}
        return lambda g: gnn.HeteroSAGE(g, 1, embed_dim=emb, **cfg)
    cfg = {k: v for k, v in gcfg["models"]["gat"].items() if k != "edge_dim"}
    return lambda g: gnn.HeteroGAT(g, 1, embed_dim=emb, **cfg)


def graph_model_params(d, gcfg, kind):
    """Parameter count of a one-big-graph model on this disease's G1. Its
    input layers are lazy, so one forward pass on a single mini-batch is
    needed before they have a size."""
    tr = np.arange(len(d["samples"]))
    graph = gnn.receivers_only(_g1_builder(d)(tr), tr)
    model = _graph_model(gcfg, kind)(graph)
    batch = next(iter(gnn.sample_loader(graph, tr[:8], batch_size=8,
                                        neighbor_sampling=gcfg["neighbor_sampling"],
                                        shuffle=False)))
    with torch.no_grad():
        model(batch)
    return gnn.count_parameters(model)


def gcn_params(d, gcfg):
    return gnn.count_parameters(gnn.TaxonomyGCN(len(d["taxonomy"][1]), 1,
                                                embed_dim=gcfg["taxon_embed_dim"],
                                                **gcfg["models"]["gcn"]))


def run_job(root, params, category, model, seed, threads=None, data=None):
    """Train and score one (disease, model, seed) on every fold of both
    schemes. Returns the per-fold rows (`gnn.binary_cv` columns plus the
    job's identity and the model's parameter count)."""
    from torch_geometric.loader import DataLoader
    from sklearn.preprocessing import StandardScaler

    if threads:
        torch.set_num_threads(threads)
    gcfg = params["graph"]
    tcfg = gcfg["train"]
    d = data if data is not None else category_data(root, params, category)
    y = d["y"]
    train_kw = dict(epochs=tcfg["epochs"], lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    n_params = {}

    def pos_weight(tr):
        return (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)

    if model in ("gcn", "gcn_shuffled"):
        taxonomy = (d["taxonomy"] if model == "gcn"
                    else gnn.shuffle_taxonomy(d["taxonomy"], seed))
        labels = pd.Series(y, index=d["samples"])

        def run_fold(tr, te, fold):
            log_prev = gr.taxon_leaf_stats(d["clr"], d["presence"], tr)[0]
            ds = gr.build_per_sample_dataset(d["clr_df"], taxonomy, labels,
                                             presence=d["presence"], log10_prevalence=log_prev)
            torch.manual_seed(seed + fold)
            net = gnn.TaxonomyGCN(len(taxonomy[1]), 1, embed_dim=gcfg["taxon_embed_dim"],
                                  **gcfg["models"]["gcn"])
            n_params["n"] = gnn.count_parameters(net)
            gnn.fit(net, DataLoader([ds[i] for i in tr], batch_size=tcfg["batch_size"],
                                    shuffle=True),
                    n_out=1, seed=seed + fold, pos_weight=pos_weight(tr), **train_kw)
            prob, _ = gnn.predict(net, DataLoader([ds[i] for i in te], batch_size=512))
            return prob

    elif model in ("sage_g1", "gat_g1"):
        cv = gnn.graph_disease_cv(
            _g1_builder(d), _graph_model(gcfg, model), d["schemes"], y, d["projects"], seed,
            batch_size=tcfg["batch_size"], neighbor_sampling=gcfg["neighbor_sampling"],
            protocol="inductive", **train_kw)
        n_params["n"] = graph_model_params(d, gcfg, model)
        run_fold = None

    elif model.startswith("mlp_"):
        target = (gcn_params(d, gcfg) if model == "mlp_gcn"
                  else graph_model_params(d, gcfg, model.replace("mlp_", "") + "_g1"))
        tol = gcfg["models"]["mlp_control"]["tolerance"]

        def run_fold(tr, te, fold):
            scaler = StandardScaler().fit(d["clr"][tr])
            torch.manual_seed(seed + fold)
            net = gnn.matched_mlp(d["clr"].shape[1], 1, target, tolerance=tol)
            n_params["n"] = gnn.count_parameters(net)
            gnn.fit(net, gnn.tensor_loader(scaler.transform(d["clr"][tr]), y[tr],
                                           batch_size=tcfg["batch_size"], shuffle=True),
                    n_out=1, seed=seed + fold, pos_weight=pos_weight(tr), **train_kw)
            prob, _ = gnn.predict(net, gnn.tensor_loader(scaler.transform(d["clr"][te]), y[te],
                                                         batch_size=2048, shuffle=False))
            return prob
    else:
        raise ValueError(f"unknown model {model!r}")

    if run_fold is not None:
        cv = gnn.binary_cv(run_fold, d["schemes"], y, d["projects"])
    return cv.assign(disease_category=category, model=model, framing=FRAMING[model],
                     pretrained=False, seed=seed, n_params=n_params.get("n"))


def run_job_to_file(root, params, category, model, seed, run_dir, threads=None):
    """`run_job`, written to its own results file (the unit of resuming).
    Returns the file path and the wall time in minutes."""
    import time

    t0 = time.time()
    out = run_job(root, params, category, model, seed, threads=threads)
    path = run_path(run_dir, category, model, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return str(path), (time.time() - t0) / 60


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

# colour encodes a model's role: blue = committed flat baseline, grey =
# control (matched MLP, shuffled tree), orange = graph model
ROLE = {"logreg": "flat baseline", "gboost": "flat baseline", "mlp_gcn": "control",
        "mlp_sage": "control", "mlp_gat": "control", "gcn_shuffled": "control",
        "gcn": "graph model", "sage_g1": "graph model", "gat_g1": "graph model"}
ROLE_COLOR = {"flat baseline": "#2a78d6", "control": "#b5b4ae", "graph model": "#eb6834"}
LABEL = {"logreg": "logistic regression", "gboost": "gradient boosting",
         "mlp_gcn": "MLP, GCN-sized", "gcn_shuffled": "GCN, shuffled tree",
         "gcn": "GCN, per-sample taxonomy", "mlp_sage": "MLP, SAGE-sized",
         "sage_g1": "GraphSAGE, G1", "mlp_gat": "MLP, GAT-sized", "gat_g1": "GAT, G1"}
ORDER = ["logreg", "gboost", "mlp_gcn", "gcn_shuffled", "gcn", "mlp_sage", "sage_g1",
         "mlp_gat", "gat_g1"]


def plot_disease_lopo(summary, ncols=3):
    """One panel per disease (`ncols` per row): a horizontal bar per model for
    the leave-one-project-out AUC (mean over folds and seeds, +/- seed sd),
    and an open diamond at the random k-fold AUC of the same model -- the
    distance between the two is how much a shared-study split flatters it.
    `summary` columns: disease_category, model, cv_scheme, auc_mean,
    auc_seed_sd."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    var._style()
    cats = list(dict.fromkeys(summary["disease_category"]))
    models = [m for m in ORDER if m in set(summary["model"])]
    ncols = min(ncols, len(cats))
    nrows = int(np.ceil(len(cats) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.6 * ncols, (0.42 * len(models) + 1.3) * nrows + 0.6),
                             sharey=True, squeeze=False)
    for ax in axes.ravel()[len(cats):]:
        ax.set_visible(False)
    for ax, cat in zip(axes.ravel(), cats):
        d = summary[summary["disease_category"] == cat].set_index(["model", "cv_scheme"])
        for i, m in enumerate(models[::-1]):
            if (m, "leave_project_out") not in d.index:
                continue
            r = d.loc[(m, "leave_project_out")]
            sd = 0.0 if pd.isna(r["auc_seed_sd"]) else r["auc_seed_sd"]
            ax.barh(i, r["auc_mean"] - 0.3, left=0.3, height=0.62, xerr=sd,
                    color=ROLE_COLOR[ROLE[m]], error_kw=dict(ecolor="#52514e", lw=1, capsize=2))
            # values in a fixed column at the panel edge, clear of bars and diamonds
            ax.text(0.995, i, f"{r['auc_mean']:.3f}", va="center", ha="right",
                    fontsize=8, color="#0b0b0b")
            if (m, "random_kfold") in d.index:
                ax.plot(d.loc[(m, "random_kfold"), "auc_mean"], i, marker="D", ms=6,
                        mfc="none", mec="#0b0b0b", mew=1.2, ls="")
        ax.axvline(0.5, color="#52514e", ls="--", lw=1)
        ax.set_xlim(0.3, 1.0)
        ax.set_title(cat)
        ax.set_xlabel("AUC")
        ax.grid(axis="y", visible=False)
    for row in axes:
        row[0].set_yticks(range(len(models)))
        row[0].set_yticklabels([LABEL[m] for m in models[::-1]])
    handles = [Patch(color=c, label=f"{r}: leave-one-project-out AUC")
               for r, c in ROLE_COLOR.items()]
    handles += [plt.Line2D([], [], marker="D", mfc="none", mec="#0b0b0b", ls="",
                           label="random k-fold AUC (studies shared)"),
                plt.Line2D([], [], color="#52514e", ls="--", label="chance")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Disease classification under honest evaluation: graph models vs flat models "
                 "and controls", fontweight="bold")
    fig.tight_layout(rect=(0, 0.6 / fig.get_figheight() + 0.04, 1, 1))
    return fig
