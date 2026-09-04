# MicroWave

**Technical vs biological signal in the Human Microbiome Compendium.**

A reproducible re-analysis of ~168k publicly aggregated human gut 16S samples that
asks two questions and answers them limitations-first:

1. **Goal 1 — disease-associated genera.** Which genera differ between disease
   cases and controls, and which of those differences survive being computed two
   different ways (batch-safe within-study meta-analysis vs naive cross-study
   pooling)?
2. **Goal 2 — how much of the variation is the lab, not the person.** How much of
   the between-sample variation is explained by which study produced a sample,
   relative to disease status and every measurable technical factor?

The headline deliverable is [`reports/final_report.md`](reports/final_report.md).
Everything under [`reports/`](reports/) is regenerated from the numbered notebooks
and is committed so the results can be read without rerunning the pipeline.

---

## Findings in brief

- **The lab is the story.** Across 341 studies, study identity explains **39–43%**
  of community variation (marginal PERMANOVA), while case/control status explains
  about **1%**, and only ~0.2 percentage points once technical factors are fitted
  first. Every technical factor tested individually outweighs disease. A plain
  linear model recovers which of 267 studies a sample came from at **76% accuracy**
  against a 4% baseline; an MLP does no better. Any analysis that pools samples
  across studies risks reporting lab differences as biology.
- **IBD has a clean, replicable genus-level signature.** 76 genera move the same
  direction under both the batch-safe and the pooled design (effect ρ = 0.98, sign
  agreement 97%): depletion of butyrate-producing Firmicutes, enrichment of
  oral/opportunistic taxa. It survives label-shuffling, depth thresholds, transform
  choice and the prevalence filter.
- **HIV shows a small real signal mostly buried under a batch artifact** (8 genera
  replicate, 33 more are pooling-only; ρ = 0.43).
- **NAFLD is weak and depth-fragile; T1D, asthma and a mixed cancer group yield
  nothing defensible here.** The mixed-cancer group's within-study and pooled
  effects are *negatively* correlated (ρ = −0.76) — a signal that those studies
  should not be combined.
- **Batch leakage, measured per disease.** Leave-one-project-out CV vs random
  k-fold CV: the random score is inflated for every disease and model (up to +0.25
  AUC). Under honest evaluation only IBD rises clearly above chance.

Full numbers, tables and caveats are in
[`reports/final_report.md`](reports/final_report.md).

---

## Repository layout

```
config/          pipeline thresholds (params.yaml) + hand-curated maps and review tables
notebooks/       the pipeline, run in numeric order (00 → 06)
src/             one module per phase; notebooks are thin runners over these
reports/         committed outputs: markdown reports, CSV tables, PNG figures
docs/DATA.md     upstream Human Microbiome Compendium file reference (input data)
requirements.txt Python dependencies
Data/            raw compendium download — NOT in the repo, you provide it (gitignored)
data/            generated Parquet artifacts — interim/ and processed/ (gitignored)
audit/           Phase 0b profiling output that feeds the hand curation (gitignored)
```

## The pipeline

Each notebook runs its `src/` module once, end to end, and checks the result. Run
them in order; every phase reads the previous phase's output.

| # | Notebook | Module | Reads | Writes |
|---|---|---|---|---|
| 0 | `00_ingest.ipynb` | `src/io.py` | `Data/raw_taxa_110.csv` (1.6 GB, 168,464 × 4,680) | `data/interim/`: `taxa_full.parquet`, `taxa_nonzero.parquet`, `taxa_prev01.npz`, `taxon_table.parquet`, `sample_depth.parquet` |
| 0b | `00b_explore.ipynb` | `src/profile.py` | `Data/tags.tsv`, `Data/sample_metadata.tsv`, `Data/projects.csv` | `audit/` field census + health-label catalogue (candidates for the hand curation in `config/within_study_review.csv`) |
| — | `00c_eda.ipynb` | `src/plots.py` | interim + processed + audit + raw metadata | `reports/figures/eda_*.png` |
| 1 | `01_harmonize.ipynb` | `src/harmonize.py` | raw metadata + `config/*.yaml` + `config/within_study_review.csv` | `data/interim/samples_harmonized.parquet`, `reports/harmonization_coverage.md` |
| 2 | `02_cohorts.ipynb` | `src/cohorts.py` | `samples_harmonized.parquet`, `sample_depth.parquet` | `data/processed/cohort_{labeled_all,within_project,healthy_baseline}.parquet`, `reports/cohort_flow.md`, `reports/project_summary.csv` |
| 3 | `03_transform.ipynb` | `src/transforms.py` | `taxa_full.parquet`, `taxa_prev01.npz`, QC-passed set | `data/processed/abund_{relative,clr,rarefied}.parquet` |
| 4 | `04_variance.ipynb` | `src/variance.py` | transforms + harmonized metadata | `reports/variance_explained.csv`, `reports/variance_report.md`, `reports/project_classifier_cv.csv`, `reports/depth_alpha_association.csv`, `reports/figures/phase4_*.png` |
| 5 | `05_differential.ipynb` | `src/differential.py` | `abund_clr.parquet`, cohorts | `reports/differential_{per_project,meta,pooled}.csv`, `reports/concordance.csv`, `reports/classifier_cv.csv`, `reports/differential_report.md`, `reports/figures/phase5_*.png` |
| 6 | `06_validation.ipynb` | `src/validation.py` | Phase 5 artifacts + cohorts | `reports/validation_*.csv`, `reports/literature_crosscheck.csv`, `reports/final_report.md`, `reports/figures/phase6_*.png` |

**Design notes.** No cohort's membership is ever read from `projects.csv.condition`
(that field misdescribes its own contents in both directions); membership comes
only from `disease_label`, built from per-sample values hand-curated in
`config/within_study_review.csv`. Wide count matrices are streamed from Parquet a
batch of rows at a time — nothing loads a full 116k × 4,680 matrix densely.
Distance-based steps run on a project-stratified subsample (`variance.distance_subsample_n`,
default 7,000) because a full dense distance matrix would be ~100 GB.

## Reproducing the analysis

### 1. Environment

Python ≥ 3.10.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on POSIX
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Phase 4 classifier; CPU build is enough
```

`rpy2` is optional and only needed if you swap the Python differential-abundance
fallback for the R implementations (ANCOM-BC / ALDEx2).

### 2. Get the data

The raw compendium is **not** in this repo. Download it (see
[`docs/DATA.md`](docs/DATA.md) for the source and file formats) and place these
files under `Data/`:

| Expected path | Compendium file |
|---|---|
| `Data/raw_taxa_110.csv` | `taxonomic_table.csv.gz`, decompressed |
| `Data/tags.tsv` | `tags.tsv.gz`, decompressed |
| `Data/sample_metadata.tsv` | `sample_metadata.tsv` |
| `Data/projects.csv` | `projects.csv` |

### 3. Run the notebooks in order

```bash
jupyter lab
```

Execute `00_ingest` → `00b_explore` → `00c_eda` → `01_harmonize` → `02_cohorts` →
`03_transform` → `04_variance` → `05_differential` → `06_validation`. Each writes
its outputs to `data/` and `reports/` and asserts its own sanity checks. The run
is deterministic (`config/params.yaml` → `seed: 42`).

## Outputs

All committed under [`reports/`](reports/):

| File | Phase | Contents |
|---|---|---|
| `final_report.md` | 6 | **The deliverable** — limitations, what was built, findings, all supporting tables |
| `harmonization_coverage.md` | 1 | per-column non-null coverage of the harmonised sample table |
| `cohort_flow.md` / `project_summary.csv` | 2 | CONSORT-style QC exclusion flow; per-project summary |
| `variance_report.md` / `variance_explained.csv` | 4 | marginal + sequential PERMANOVA, depth vs alpha diversity |
| `project_classifier_cv.csv` | 4 | PyTorch project-predictability negative control (linear + MLP) |
| `differential_report.md` | 5 | within-study meta-analysis hits, concordance, batch-leakage |
| `differential_{per_project,meta,pooled}.csv` | 5 | per-project contrasts, DerSimonian-Laird meta-analysis, naive pooled + ComBat/mixedlm |
| `concordance.csv` | 5 | every genus classified *replicated* / *pooling-only* / *within-only* |
| `classifier_cv.csv` | 5 | disease-classifier AUC, leave-one-project-out vs random k-fold |
| `validation_*.csv` | 6 | label-shuffling null, depth / transform / prevalence sensitivity |
| `literature_crosscheck.csv` | 6 | replicated hits vs curated genus-direction priors (orientation only) |
| `figures/eda_*.png`, `phase4_*.png`, `phase5_*.png`, `phase6_*.png` | — | all figures |

## Configuration

Every threshold lives in [`config/params.yaml`](config/params.yaml) — nothing is
hardcoded in a notebook. Key values: QC depth threshold 10,000 reads (sensitivity
at 5k/10k/20k); prevalence filter 1% (sensitivity at 10%); PERMANOVA ≥ 999
permutations; CLR pseudocount 0.5; disease category needs ≥ 2 projects and ≥ 100
pooled cases to enter Goal 1.

The `config/*.yaml` maps and `config/within_study_review.csv` are **data, not
code** — the label-detection rules, tag→concept mapping, kit-family and
condition-name normalisation, null vocabulary, and per-project human review of
which field is the real health label. They are meant to be reviewed and extended
without touching `src/`.

## Limitations (see the final report for the full list)

- **Not the world.** ~61% of samples are from Europe / North America, 36% from the
  United States; ~17% have no usable country. Claims are about a mostly Western gut.
- **Genus-level resolution only** — no species or strains.
- **Observational** — no causal claims.
- **Most samples are unlabelled** (~81.5%). Goal 1 rests on 41 studies / ~14k
  samples; "healthy" is each study's own definition, harmonised at the word level
  only.
- Diet / smoking / antibiotic history come from essentially one cohort.

## Data source & citation

Data is the **Human Microbiome Compendium** (16S rRNA amplicon, human gut),
processed as described in the preprint:
<https://www.biorxiv.org/content/10.1101/2023.10.11.560955v1>. Some derived files
live at <https://doi.org/10.5281/zenodo.13733483>. This repository redistributes
none of it — see [`docs/DATA.md`](docs/DATA.md).

This repo has no `LICENSE` file; the analysis code is unlicensed by default (all
rights reserved) unless a licence is added. The compendium data carries its own
terms from the source above.
