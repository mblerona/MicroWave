# Implementation Specification — Human Microbiome Compendium

**Technical companion to [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md).** That document explains *why*; this one specifies *what to build*. It assumes familiarity with compositional data analysis, PERMANOVA, mixed-effects models and cross-validation design.

Seven phases, 0 through 6. Each phase declares inputs, steps, outputs and acceptance criteria. Phases are sequential — Phase 4 (technical variance) deliberately precedes Phase 5 (disease association), because its output calibrates how much weight Phase 5's findings can bear.

---

## Reference facts

All figures measured directly from `Data/`. Cite these rather than re-deriving them ad hoc; any discrepancy found during implementation should be treated as a bug and reconciled here.

### File inventory

| File | Bytes | Shape |
|---|---|---|
| `Data/raw_taxa_110.csv` | 1,596,313,516 | 168,464 rows × 4,682 cols (unnamed index, `sample`, 4,680 taxa) |
| `Data/sample_metadata.tsv` | 23,444,832 | 168,464 rows × 11 cols |
| `Data/projects.csv` | 68,637 | 482 rows × 10 cols, UTF-8 **with BOM** |
| `Data/tags.tsv` | 215,869,529 | 3,489,745 rows × 5 cols (long format) |

`Data/` is gitignored. Do not commit derived data either — write everything to `data/interim/` and `data/processed/`, both gitignored.

### Keys and join

- `raw_taxa_110.csv.sample` is `{project}_{srr}`, e.g. `PRJDB10485_DRR243823`
- `sample_metadata.tsv` has `project` and `srr` as separate columns
- Join key: `metadata.project + "_" + metadata.srr == taxa.sample`
- **Verified exact 1:1**: 168,464 matched, 0 orphans on either side. Assert this, don't re-check interactively.
- `tags.tsv` joins on `srr` (also carries `project`, `srs`)
- `projects.csv` joins on `project`

### Matrix characteristics

- Median 45 non-zero taxa per sample (p10 = 6, p90 = 105, max = 717)
- Taxa in ≥50% of samples: 10 · ≥10%: 151 · ≥1%: 421
- Never observed in a 4,000-sample probe: ~2,068 of 4,680 — **estimate, confirm on full data in Phase 0**
- Kingdom split: Bacteria 4,517 · Archaea 161 · Eukaryota 1 · `NA` 1
- Genus resolution: 3,289 of 4,680 resolved; 496 have genus == `NA`
- Column headers are 6-level taxonomy strings, period-separated: `Bacteria.Bacillota.Clostridia.Lachnospirales.Lachnospiraceae.Blautia`. Unresolved ranks are literal `NA`; some contain `Incertae Sedis` and spaces (`Lachnospiraceae NK4A136 group`) and brackets (`[Eubacterium] siraeum group`). **Do not** split on `.` without accounting for exactly 6 fields.

### Read depth

Row sums (4,000-sample probe): min 1 · p10 9,957 · median 35,224 · p90 134,349 · max 3,490,478.
`sample_metadata.total_bases` is missing for 2,668 samples (1.58%) — use computed row sums, not `total_bases`, as the depth variable.

### Technical covariates

| Variable | Source | Distribution |
|---|---|---|
| `project` | both | 482 levels; min 50, median 162, max 4,850 samples |
| `amplicon` | `projects.csv` | v4 218, v3-v4 118, v1-v2 32, v3 23, +8 rarer; **73 blank** |
| `instrument` | `sample_metadata` | 12 levels; MiSeq 134,353, HiSeq 2500 10,975, 454 GS 8,790 |
| `bead_beating` | `projects.csv` | TRUE 318, FALSE 86, blank 78 |
| `kit` | `projects.csv` | 283 distinct strings / 358 non-blank projects |
| `region` | `sample_metadata` | 9 levels; Europe & N. America 103,564 (61%) |
| `iso` | `sample_metadata` | 69 values; `unknown` 28,192 (17%); US 59,994 |
| `library_source` | `sample_metadata` | GENOMIC 19,939 / METAGENOMIC 148,525 — **annotation artifact; do not model as biology** |

### Label availability — the number the project hinges on

- Samples with ≥1 non-null disease/health label: **31,280 (18.6%) across 85 projects**
- Projects whose `condition` names both disease and controls: 42 (16,892 samples), of which **5,996 across 12 projects carry per-sample labels**; 10,896 across 30 projects do not
- Healthy-only projects: 125 (54,571 samples)

**Label tags have no standard name — this is the main trap in Phase 1.** Three naming families coexist:

1. *Generic*: `host_disease` 4,411 · `diagnosis` 1,190 · `health_state` 977 · `condition_at_sampling` 454 · `disease` 533 · `host_phenotype` 679 · `metabolic_phenotype` 414
2. *Study-structural*: `group` 5,955 · `sample group` 4,054 · `clinical condition` 4,820 · `subset_healthy` 2,139 · `cohort`
3. *Named after the disease itself* — the family a naive regex misses: `asd` (yes/no), `parkinson` (yes/no), `hiv` (positive/negative), `ibd_diagnosis_refined` 4,787, `ibd diagnosis` 1,651, `cardiovascular_disease` 4,394, `kidney_disease` 3,978, `liver_disease` 3,978, `lung disease` 3,562, `lung/pulmonary disorder` (asthma/control)

A pattern search for `disease|diagnos|condition|group|health` alone finds only ~24,172 samples / 71 projects and silently drops ~7,000. Detection **must** additionally derive keywords from each project's own `projects.csv.condition` string and match them against that project's tag names and values.

Automated detection is necessary but not sufficient — it also picks wrong tags. Example: PRJEB4335 ("HIV + healthy") has both `hiv` (positive 30 / negative 22, the correct label) and `cohort` (a2/b/b2/b1, an internal grouping); a "most samples covered" heuristic selects `cohort`. Per-project human review is required, which is why step 8 below produces a curated config file rather than a runtime rule.

- Confirmed genuinely absent, not merely misnamed (full tag listing inspected): PRJNA237362 (1,379, "CD and healthy") carries only `isolation_source` (body site) + encoded `strain`/`isolate` IDs; PRJEB5482 (1,961) encodes identity in `Submitter Id` (`bgtw1.f.m10`); PRJEB8463 (793) has only numeric submitter IDs; PRJNA685914 (1,683) uses `sample_title` (`p0780d1`).
- **No other compendium file can supply these.** Per the upstream README, `tags.tsv` is a complete dump of BioSample attributes; the remaining files (ASV tables, Greengenes2 tables, `obs_md.txt`, SILVA) carry no sample-phenotype dimension. `gg2-2022.10-cref99.biom.qza` is the sole structural possibility since BIOM supports optional per-sample metadata, but it re-classifies the same samples and any embedded metadata is expected to duplicate `sample_metadata.tsv`. Recovery for the 10,896 requires the source publications via `projects.csv.link`.

### Known metadata corruption

- `sex`/`host_sex` contains numeric ages: `"47"` ×2,794, `"48"` ×1,501. Also `neuter`, `"not providednot provided"`.
- `age_unit` contains numbers: `78` ×1,663, `75` ×276. Note both `age_unit` and `age units` and `host_age_units` exist.
- Age: 32,926 numeric / 9,870 non-numeric of 42,796. Non-numeric forms: `"6 months"`, `"3.5 years"`, `"0-100 days"`, `"17-29 yo"`, `"85-89"`, `">=100"`, `"child 2-year visit"`.
- Null vocabulary (case-insensitive, at minimum): `missing`, `not provided`, `not applicable`, `na`, `not collected`, `none`, `unknown`, `restricted access`, `""`.
- `host` non-human: `mus musculus` 221, `rhesus macaque` 122, `simulated gastrointestinal` 96, `labcontrol test` 86. Human variants: `homo sapiens` 119,972, `homo_sapiens` 745, `homo sapiens sapiens` 201, `human beings` 130, `homo` 92, `homosapiens` 80, `human male adult` 66, `infant` 407.
- `collection_date`: 145,112 samples have the tag; **45,402 values unparseable**; parseable range 1998–2021 peaking 2015–2017. `pubdate` in `sample_metadata` runs 2012–2021 and is *not* a proxy for it.
- `projects.csv.condition`: 204 distinct free-text values; `CRC` and `colorectal cancer` are separate strings.
- `projects.csv.sample_type`: 346 stool, plus saliva/rectal swab/biopsy combinations, and 3 projects of `mice`.
- Lifestyle tags (`diet type`, `smoking_frequency`, `antibiotic_history`, `weight_change`, …) all cover ≈4,842 samples — a single cohort. Treat as cohort-specific.

---

## Repository layout

```
config/
  tag_map.yaml            # tag name -> canonical concept
  condition_map.yaml      # 204 free-text conditions -> controlled vocabulary
  kit_map.yaml            # 283 kit strings -> manufacturer families
  null_values.yaml        # the null vocabulary
  params.yaml             # thresholds (depth cutoff, prevalence filter, seeds)
src/
  io.py                   # chunked readers, parquet writers
  harmonize.py            # Phase 1 logic
  cohorts.py              # Phase 2 filters + flow table
  transforms.py           # Phase 3
  variance.py             # Phase 4
  differential.py         # Phase 5
notebooks/
  00_ingest.ipynb ... 06_validation.ipynb
data/interim/             # gitignored
data/processed/           # gitignored
reports/figures/
```

`TestNotebook.ipynb` at repo root is currently empty (0 bytes) — delete it and use the numbered notebooks. Add `data/` to `.gitignore` (it currently contains only `Data/`).

**Dependencies** (`requirements.txt`): `pandas>=2.0`, `pyarrow`, `numpy`, `scipy`, `scikit-learn`, `scikit-bio`, `statsmodels`, `umap-learn`, `matplotlib`, `seaborn`, `pyyaml`, `tqdm`. Optional: `rpy2` + R (`ANCOMBC`, `ALDEx2`, `MMUPHin`) if the R route is taken in Phase 5.

Set a global seed in `config/params.yaml` and thread it through every stochastic step (UMAP, permutation tests, CV splits, classifiers).

---

## Phase 0 — Ingest & storage

**Goal:** turn a 1.6 GB CSV into something that loads in seconds.

**Inputs:** `Data/raw_taxa_110.csv`

**Steps**

1. Read the header separately; parse the 4,680 taxonomy strings into a taxon table with columns `col_index, full_string, kingdom, phylum, class, order, family, genus`. Split on `.` with `maxsplit` guarded to exactly 6 fields — headers contain spaces and brackets but the rank count is fixed.
2. Stream the body with `pd.read_csv(..., chunksize=2000, dtype=np.int32, index_col=0)`. Counts fit int32 comfortably (observed max ~3.5M). Drop the unnamed index column.
3. Accumulate per-chunk: row sums (depth), per-column non-zero counts, per-column totals.
4. Write the full matrix to `data/interim/taxa_full.parquet` (row group size ~5,000, snappy).
5. Compute the prevalence filter from the accumulated column statistics — **on full data, not the probe**. Persist two derived artifacts:
   - `taxa_nonzero.parquet` — all-zero columns dropped (expected ~2,612 retained; the ~2,068 dropped figure is a probe estimate to be replaced by the exact count)
   - `taxa_prev01.npz` — `scipy.sparse` CSR of taxa present in ≥1% of samples (expected ~421 columns) plus an aligned taxon index, for modelling
6. Write `data/interim/sample_depth.parquet` — `sample`, `depth`, `n_nonzero`.

**Outputs:** `taxa_full.parquet`, `taxa_nonzero.parquet`, `taxa_prev01.npz`, `taxon_table.parquet`, `sample_depth.parquet`

**Acceptance criteria**
- Row count is exactly 168,464 in every artifact
- Sum of all counts in `taxa_full.parquet` equals the streaming checksum from step 3
- `taxa_nonzero` row sums are identical to `taxa_full` row sums (dropping all-zero columns cannot change any row sum)
- The exact all-zero column count is recorded and this document's estimate updated

---

## Phase 1 — Metadata harmonisation

**Goal:** one tidy row per sample with trustworthy columns.

**Inputs:** `Data/tags.tsv`, `Data/sample_metadata.tsv`, `Data/projects.csv`

**Steps**

1. **Build `config/tag_map.yaml`** — an explicit, reviewed mapping from tag name to canonical concept. Do not infer at runtime; a hand-checked file is the deliverable. Minimum concepts: `age`, `age_unit`, `sex`, `bmi`, `host_species`, `collection_date`, `disease_label`, `body_site`, `subject_id`, `geo`. Merge the known duplicates (`age`/`host_age`, `sex`/`host_sex`, `body_mass_index`/`host_body_mass_index`, the three latitude spellings).
2. **Stream and pivot.** Single pass over 3.49M rows, keeping only tags present in `tag_map.yaml`, pivoting long→wide keyed on `srr`. Do not pivot all 2,608 tags — the result is unusably sparse.
3. **Apply the null vocabulary** from `config/null_values.yaml`, case-insensitively, before any type coercion.
4. **Repair known corruptions.** In `sex`, values matching `^\d+$` are leaked ages — null the sex and, if `age` is absent for that sample, log as a candidate age recovery (do not silently promote it). In `age_unit`, purely numeric values are invalid — null them and fall back to project-level unit inference.
5. **Parse age → `age_years` + `age_confidence`.** Handle: plain numeric with an explicit unit tag; embedded units (`"6 months"`, `"3.5 years"`, `"0-100 days"`); ranges (`"17-29 yo"`, `"85-89"` → midpoint, confidence `range`); open bounds (`">=100"` → 100, confidence `bound`). For bare numerics with no unit, infer the unit **per project** from the distribution of that project's values and its `subjects` field in `projects.csv` (`infants` implies months) — mark these `inferred`. Never silently assume years.
6. **Normalise `host_species`** — collapse the Homo sapiens variants; flag `mus musculus`, `rhesus macaque`, `simulated gastrointestinal`, `labcontrol test` as non-human. `infant` in the host field means human — handle explicitly.
7. **Parse `collection_date`** to `collection_year`, tolerating the formats present; leave the ~45,402 unparseable values null rather than guessing.
8. **Derive `disease_label`.** Two stages, and the second is not optional.
   - *Candidate detection*: for each project, flag tags that (a) match the generic patterns, **or** (b) whose name contains a keyword from that project's `projects.csv.condition` string, **or** (c) whose value set combines a case/control vocabulary (`yes|no`, `positive|negative`, `case|control`, `healthy`) with a condition keyword. Restrict to tags with 2–10 distinct non-null values.
   - *Human curation*: review every candidate and record the chosen tag in `config/condition_map.yaml`, keyed by `(project, tag, raw_value) → {healthy, case, unknown} + disease_category`. Detection alone picks wrong tags (see PRJEB4335 above); the config file, not the heuristic, is the source of truth. Expect ~85 projects to curate.
9. **Normalise `projects.csv`.** `condition`: 204 → ~30 controlled categories (unify `CRC` ≡ `colorectal cancer`). `kit`: 283 → manufacturer families (all QIAamp spellings → one). Keep `amplicon` verbatim with blanks explicit as `unknown` — do not impute. Read with `encoding='utf-8-sig'` for the BOM.
10. Join all three sources onto the 168,464-row spine.

**Outputs:** `data/interim/samples_harmonized.parquet`; `reports/harmonization_coverage.md` (non-null count and % per canonical concept, before/after repair); the three populated config maps.

**Acceptance criteria**
- Exactly 168,464 rows; join introduces no duplicates and no nulls in `project`/`srr`
- Every canonical concept's coverage is reported and no null-vocabulary string survives in any harmonised column
- `age_years` has no value < 0 or > 120; every non-null value carries an `age_confidence`
- `disease_label` non-null count lands near **31,280 across ~85 projects**. Materially *below* that means the detector is missing disease-named tags; materially above may mean it is capturing non-label groupings. Either way, investigate before proceeding — this number gates the whole of Phase 5.

---

## Phase 2 — QC & cohort construction

**Goal:** an auditable path from 168,464 to the working set.

**Inputs:** `samples_harmonized.parquet`, `sample_depth.parquet`

**Steps**

1. Apply exclusions **in this fixed order**, logging count in / removed / count out at each step:
   1. non-human `host_species`
   2. non-stool `sample_type` (retain stool; drop saliva, swab, biopsy, mixed)
   3. depth below threshold — default **10,000** reads
2. Assemble the flow into a CONSORT-style table (`reports/cohort_flow.md`) and a figure.
3. Justify the depth threshold with rarefaction curves — plot observed richness against depth and show where it plateaus. Run the whole downstream pipeline at 5,000 and 20,000 as a Phase 6 sensitivity analysis; the default is a decision, not a fact.
4. Build and persist three cohorts:
   - `labeled_all` — non-null `disease_label` (~31,280 pre-filter, 85 projects) → Phase 5b/5c
   - `within_project` — projects containing **both** `healthy` and `case` samples with per-sample labels (~5,996 pre-filter, 12 projects) → Phase 5a
   - `healthy_baseline` — samples from the 125 healthy-only projects (~54,571 pre-filter) → optional extensions
5. Emit a per-project summary: n samples, n cases, n controls, amplicon, kit family, instrument, region — the table used to decide which projects enter the meta-analysis.

**Outputs:** `data/processed/cohort_{labeled_all,within_project,healthy_baseline}.parquet`, `reports/cohort_flow.md`, `reports/project_summary.csv`

**Acceptance criteria**
- Flow table arithmetic reconciles exactly to 168,464
- `within_project` ⊆ `labeled_all`
- Every project in `within_project` has ≥10 cases and ≥10 controls after filtering (projects failing this are excluded and listed)

---

## Phase 3 — Normalisation & transformation

**Goal:** put counts on a comparable scale without baking in an untested assumption.

**Steps**

1. **Relative abundance** — counts / row sum.
2. **CLR (centred log-ratio)** — the compositionally correct transform. Zero replacement must be explicit and documented; default to a pseudocount of 0.5 applied after prevalence filtering, with multiplicative replacement as the documented alternative. Given ~99% sparsity, **apply CLR only to the prevalence-filtered matrix** (≥1% prevalence, ~421 taxa) — CLR over 4,680 mostly-zero columns is dominated by the pseudocount and is meaningless.
3. **Rarefaction** — subsample to even depth. Use **only** for alpha/beta diversity metrics, not for differential abundance.

Carry all three forward so that transform choice is tested in Phase 6 rather than assumed.

**Outputs:** `data/processed/abund_{relative,clr,rarefied}.parquet`

**Acceptance criteria** — relative abundance rows sum to 1.0 within floating-point tolerance; CLR rows sum to ~0; rarefied rows all have identical depth.

---

## Phase 4 — Goal 2: technical variance (runs before Phase 5)

**Goal:** quantify how much variation is lab rather than biology. This output sets the prior for Phase 5.

**Steps**

1. **Distances.** Bray-Curtis on relative abundance; Aitchison (Euclidean on CLR) on the filtered matrix. At 24k+ samples a full dense distance matrix is ~4.6 GB — subsample (project-stratified, e.g. 10–15k) for the distance-based tests and state that clearly, or use a memory-mapped implementation.
2. **Ordination.** PCoA and UMAP, each coloured separately by `project`, `amplicon`, `kit_family`, `instrument`, `depth` (binned), `region`, `disease_label`. The `project`-coloured panel is the headline figure.
3. **Variance partitioning.** PERMANOVA (`skbio.stats.distance.permanova`) or `adonis`-style sequential decomposition for R² per factor. Fit technical factors first, then disease, and also disease-first, and report both orderings — sequential R² depends on term order and hiding that would be misleading. Constrain permutations within project where the design requires it.
4. **Negative control — project predictability.** Train a classifier to predict `project` from composition under stratified CV. High accuracy is the *expected* result and is the point: it demonstrates the lab signature is strong, low-dimensional and trivially learnable, which is exactly what a naive disease model would latch onto.
5. Also report the marginal association of `depth` with alpha diversity — richness rises with depth and this must be controlled, not assumed away.

**Outputs:** `reports/variance_explained.csv` (factor, df, R², pseudo-F, p, term order), ordination figures, project-classifier accuracy.

**Acceptance criteria** — R² reported for every factor under both term orderings; permutation count ≥999 and recorded; project-classifier accuracy reported against the majority-class baseline.

---

## Phase 5 — Goal 1: disease association

**Goal:** identify genera that differ between cases and controls, and establish which findings survive both designs.

Restrict to disease categories with sufficient representation after Phase 2 — decide the minimum from `reports/project_summary.csv` (suggested: ≥2 projects and ≥100 cases per category). Run each disease category separately; do not pool distinct diseases into a generic "case" group.

**5a — Within-project differential abundance**
Per project in `within_project`, test each taxon (prevalence-filtered) for case/control difference. Primary method **ANCOM-BC** or **ALDEx2** (via `rpy2`); pure-Python fallback is Wilcoxon rank-sum on CLR values with Benjamini-Hochberg FDR. Adjust for available covariates (`age_years`, `sex`) where coverage permits; report which projects had them. Output per project: taxon, effect size, SE, p, q, n cases, n controls.

**5b — Random-effects meta-analysis**
Combine 5a's per-project effect sizes with inverse-variance weighting and a DerSimonian-Laird random-effects model. Report pooled effect, CI, I² heterogeneity, and per-taxon project count. **This is the statistically correct way to use multiple studies** — it never pools raw samples across labs, so batch effects cannot leak into the pooled estimate.

**5c — Naive pooled + batch correction**
Pool `labeled_all`, correct for study, and test. Run at least two correction strategies: `project` as a random effect in a mixed model, and an explicit correction (ComBat-seq or MMUPHin). Included specifically as the contrast case for 5d.

**5d — Concordance (the centrepiece)**
Compare the ranked taxon lists from 5b and 5c: overlap at fixed FDR, rank correlation, sign agreement, and a scatter of 5b effect size against 5c effect size with disagreements highlighted. Classify each taxon as *replicated* (both), *pooling-only* (5c only — likely batch artifact), or *within-only* (5a/5b only — likely real but under-powered). This classification is the project's primary result.

**5e — Batch leakage quantification**
Train a disease classifier (regularised logistic regression and gradient boosting) under two CV schemes:
- **Leave-one-project-out** — every test sample comes from a study the model never saw
- **Random stratified k-fold** — test samples share studies with training data

Report both accuracies/AUCs. **The gap is a direct measurement of how much apparent predictive performance is study recognition rather than disease signal.** Expect the random-CV figure to be substantially higher; that gap, quantified, is a publishable observation about how microbiome ML is commonly evaluated.

**Outputs:** `reports/differential_{per_project,meta,pooled}.csv`, `reports/concordance.csv`, concordance figure, `reports/classifier_cv.csv`

**Acceptance criteria**
- Every result carries effect size, CI and FDR-corrected q — no bare p-values anywhere
- Taxon counts tested and the multiple-testing universe stated explicitly per analysis
- 5e reports both CV schemes side by side; reporting only random CV is a defect

---

## Phase 6 — Validation & reporting

**Steps**

1. **Label-shuffling null.** Re-run 5b and 5c with `disease_label` permuted within project. Any surviving "significant" hit reveals a flaw in the pipeline — investigate before publishing anything.
2. **Threshold sensitivity.** Re-run the pipeline at depth thresholds 5,000 / 10,000 / 20,000. Report how the replicated-taxon set changes. Conclusions that only hold at one threshold must be reported as such.
3. **Transform sensitivity.** Repeat key tests under relative abundance vs CLR; report agreement.
4. **Prevalence-filter sensitivity.** Repeat at ≥1% and ≥10% prevalence.
5. **Literature cross-check.** For the top replicated hits per disease, check against the source publications via `projects.csv.link` DOIs. Agreement is supporting evidence; disagreement is worth reporting.
6. **Assemble the final report** from the artifacts, with the Section 8 limitations from [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) stated up front rather than buried.

**Acceptance criteria** — shuffled null produces no significant hits at the chosen FDR; every headline claim is accompanied by its sensitivity result; the notebooks run start to finish from raw `Data/` on a clean checkout.

---

## Cross-cutting rules

- **Never `pd.read_csv` the full taxa matrix.** After Phase 0 it exists as Parquet; read that.
- **`library_source` (GENOMIC vs METAGENOMIC) is an annotation artifact.** Do not model it as biology. It may be used as a technical covariate in Phase 4.
- **`pubdate` is not sampling date.** Use `collection_year` from Phase 1 for anything temporal.
- **Prefer computed row sums over `total_bases`** for depth (the latter is missing for 1.58% and measures bases submitted, not reads assigned).
- **Every threshold lives in `config/params.yaml`**, never inline in a notebook.
- **Log every row-count change.** If a step alters the sample count, it writes a line to the flow table.
