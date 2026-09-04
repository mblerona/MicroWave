# Human Microbiome Compendium — final report

## Limitations, stated up front

- **The sample is not the world.** ~61% of samples are from Europe and North America, 36% from the United States alone; ~17% have no usable country. Any claim about "the human gut microbiome" is really a claim about a mostly Western one.
- **Genus-level resolution only.** Broad bacterial groups, not species or strains. Strain-level effects are invisible here.
- **Observational, so no causal claims.** An elevated genus in a disease may be a consequence of the disease, its treatment, or an associated diet change.
- **Most samples have no health label.** ~81.5% are unlabelled; the disease work rests on the labelled minority, and the within-study comparison on 41 projects.
- **Health labels are self-defined by each study.** One study's "healthy" is another's "mild symptoms" -- the words are harmonised, the underlying clinical assessments are not.
- **Some labels are participant-reported, not clinician-assigned.** Only clinician diagnoses are counted as cases; weaker levels are retained as a graded covariate for sensitivity use only.
- **Diet / smoking / antibiotic history come from essentially one cohort** and describe that cohort, not the compendium.

## What was built

A cleaned per-sample table, an auditable filtering flow, a variance-partitioning answer for the technical vs biological split, a ranked list of disease-associated genera with a replication verdict, and the robustness checks below. The numbered notebooks run in order from the raw files; this report consumes their outputs.

## Findings in brief

**The lab is the story.** Across 341 studies, the single largest source of difference between two stool samples is which study produced them — project identity explains 39–43% of community variation (marginal PERMANOVA), while case/control status explains about 1%. Every purely technical factor tested — 16S region, sequencing instrument, extraction kit, bead-beating, sequencing depth, world region — individually outweighs disease. A plain linear model identifies which of 267 studies a sample came from with 76% accuracy against a 4% chance baseline, and a neural network does no better: the lab signature is strong, low-dimensional and trivially learnable. This one fact governs everything else — any analysis that pools samples across studies is at high risk of reporting lab differences as biology.

**One disease has a clean, replicable genus-level signature: IBD.** 76 genera move in the same direction whether the comparison is done the batch-safe way (within-study meta-analysis) or the naive way (pooled across labs with correction) — effect-size correlation 0.98, sign agreement 97%, almost no between-study heterogeneity. The pattern is the one the published literature describes: near-uniform depletion of butyrate- and other short-chain-fatty-acid-producing Firmicutes (*Roseburia*, *Agathobacter*, *Faecalibacterium*-adjacent *Gemmiger*, *Fusicatenibacter*, *Coprococcus*, *Butyribacter*, several *Eubacterium* and Oscillospiraceae groups, Christensenellaceae), alongside enrichment of oral-cavity and opportunistic taxa (*Peptostreptococcus*, *Veillonella*, *Rothia*, *Gemella*, *Granulicatella*, *Enterococcus*, *Klebsiella*, *Fusobacterium*). *Akkermansia* and *Alistipes* are down. The result survives label-shuffling, depth thresholds from 5,000 to 20,000 reads, the choice of transform, and the prevalence filter, and shows zero disagreements with textbook priors.

**HIV shows a small real signal mostly buried under a batch artifact.** Only 8 genera replicate across both designs (*Fusobacterium* up; *Romboutsia*, *Christensenellaceae R-7 group*, an *[Eubacterium] siraeum* group, *Clostridium* and others down), but the pooled analysis alone flags 33 further genera, and the two designs' effect sizes correlate only 0.43. This is the concrete demonstration of the pooling hazard: most of what a naive cross-study HIV analysis "finds" does not hold up once batch effects are properly excluded.

**NAFLD is weak and fragile; T1D, asthma and the mixed cancer group yield nothing defensible here.** NAFLD produces 8 replicated depletions, but they vanish when the depth threshold is raised to 20,000 reads. T1D and asthma have no hits that survive multiple-testing correction in the batch-safe analysis — they are honestly under-powered (100 and 248 cases across a handful of small studies). The heterogeneous "other cancers" category — colorectal adenoma/carcinoma plus one gastric-cancer study — shows within-study and pooled effects that are *negatively* correlated (ρ = −0.76), which is a signal that these studies should not be combined at all, not a finding.

**How much of a disease "prediction" is really lab recognition — measured.** A classifier tested on studies it has never seen (leave-one-project-out CV) is compared with the same classifier tested on held-out samples from studies it *has* seen (random k-fold CV). The random-CV score is inflated for every disease and every model — by up to +0.25 AUC for HIV and +0.21 for T1D. Under the honest evaluation, only IBD rises clearly above chance (AUC ≈ 0.70–0.73); for the rest, a model that cannot see the originating lab can barely tell case from control.

## Metadata harmonisation — coverage

168,464 samples.

| column | non-null | % |
|---|---|---|
| `age_years` | 39,045 | 23.2% |
| `age_confidence` | 39,045 | 23.2% |
| `sex` | 35,063 | 20.8% |
| `bmi` | 13,607 | 8.1% |
| `host_species_human` | 122,459 | 72.7% |
| `collection_year` | 103,822 | 61.6% |
| `subject_id` | 45,653 | 27.1% |
| `disease_label` | 19,732 | 11.7% |
| `disease_category` | 7,299 | 4.3% |
| `condition_category` | 168,464 | 100.0% |
| `kit_family` | 168,464 | 100.0% |
| `amplicon` | 168,464 | 100.0% |

## Cohort flow

### QC exclusions (fixed order)

| step | n in | n removed | n out | note |
|---|---|---|---|---|
| exclude confirmed non-human host_species | 168,464 | 503 | 167,961 | 46,005 samples with unresolved host_species (<NA>) kept, not dropped |
| exclude non-stool / missing sample_type | 167,961 | 40,800 | 127,161 | kept only sample_type == "stool" exactly; mixed body sites and missing values dropped |
| exclude depth < 10,000 reads | 127,161 | 10,504 | 116,657 | depth = row sum from sample_depth.parquet (Phase 0), not sample_metadata.total_bases |

### Cohorts (built from the QC-passed population)

| cohort | n samples | n projects |
|---|---|---|
| labeled_all | 14,954 | 48 |
| within_project | 14,033 | 41 |
| healthy_baseline | 921 | 7 |

## Goal 2 — how much variation is the lab, not the person

### Marginal PERMANOVA (each factor alone)

| distance | factor | n | groups | R2 | pseudo-F | p | perms |
|---|---|---|---|---|---|---|---|
| bray_curtis | project | 7,015 | 341 | 0.4338 | 15.0 | 0.001 | 999 |
| bray_curtis | amplicon | 7,015 | 11 | 0.1108 | 87.3 | 0.001 | 999 |
| bray_curtis | kit_family | 7,015 | 9 | 0.0474 | 43.6 | 0.001 | 999 |
| bray_curtis | instrument | 7,015 | 12 | 0.0385 | 25.5 | 0.001 | 999 |
| bray_curtis | region | 7,015 | 8 | 0.0406 | 42.3 | 0.001 | 999 |
| bray_curtis | depth_bin | 7,015 | 5 | 0.0049 | 8.7 | 0.001 | 999 |
| bray_curtis | library_source | 7,015 | 2 | 0.0023 | 16.4 | 0.001 | 999 |
| bray_curtis | bead_beating | 7,015 | 3 | 0.0121 | 42.9 | 0.001 | 999 |
| aitchison | project | 7,015 | 341 | 0.3942 | 12.8 | 0.001 | 999 |
| aitchison | amplicon | 7,015 | 11 | 0.0586 | 43.6 | 0.001 | 999 |
| aitchison | kit_family | 7,015 | 9 | 0.0339 | 30.7 | 0.001 | 999 |
| aitchison | instrument | 7,015 | 12 | 0.0422 | 28.0 | 0.001 | 999 |
| aitchison | region | 7,015 | 8 | 0.0360 | 37.4 | 0.001 | 999 |
| aitchison | depth_bin | 7,015 | 5 | 0.0233 | 41.9 | 0.001 | 999 |
| aitchison | library_source | 7,015 | 2 | 0.0014 | 10.1 | 0.001 | 999 |
| aitchison | bead_beating | 7,015 | 3 | 0.0062 | 21.7 | 0.001 | 999 |
| aitchison_labeled | project | 7,001 | 48 | 0.2515 | 49.7 | 0.001 | 999 |
| aitchison_labeled | amplicon | 7,001 | 5 | 0.0322 | 58.1 | 0.001 | 999 |
| aitchison_labeled | kit_family | 7,001 | 6 | 0.0395 | 57.6 | 0.001 | 999 |
| aitchison_labeled | instrument | 7,001 | 7 | 0.0641 | 79.8 | 0.001 | 999 |
| aitchison_labeled | region | 7,001 | 8 | 0.0164 | 16.7 | 0.001 | 999 |
| aitchison_labeled | depth_bin | 7,001 | 5 | 0.0245 | 43.9 | 0.001 | 999 |
| aitchison_labeled | library_source | 7,001 | 2 | 0.0105 | 73.9 | 0.001 | 999 |
| aitchison_labeled | bead_beating | 7,001 | 3 | 0.0352 | 127.8 | 0.001 | 999 |
| aitchison_labeled | disease_label | 7,001 | 2 | 0.0097 | 68.3 | 0.001 | 999 |

### Sequential decomposition (term order matters)

| distance | analysis | # | factor | df | R2 | pseudo-F | p |
|---|---|---|---|---|---|---|---|
| aitchison | sequential:technical_first | 1 | amplicon | 4 | 0.0322 | 68.1 | 0.001 |
| aitchison | sequential:technical_first | 2 | kit_family | 5 | 0.0304 | 51.5 | 0.001 |
| aitchison | sequential:technical_first | 3 | instrument | 6 | 0.0534 | 75.4 | 0.001 |
| aitchison | sequential:technical_first | 4 | region | 7 | 0.0146 | 17.7 | 0.001 |
| aitchison | sequential:technical_first | 5 | depth_bin | 4 | 0.0146 | 30.9 | 0.001 |
| aitchison | sequential:technical_first | 6 | library_source | 1 | 0.0093 | 78.9 | 0.001 |
| aitchison | sequential:technical_first | 7 | bead_beating | 2 | 0.0208 | 88.0 | 0.001 |
| aitchison | sequential:technical_first | 8 | disease_label | 1 | 0.0016 | 13.4 | 0.001 |
| aitchison | sequential:technical_first | 9 | Residual | 6970 | 0.8232 | - | - |
| aitchison | sequential:disease_first | 1 | disease_label | 1 | 0.0097 | 81.8 | 0.001 |
| aitchison | sequential:disease_first | 2 | amplicon | 4 | 0.0319 | 67.6 | 0.001 |
| aitchison | sequential:disease_first | 3 | kit_family | 5 | 0.0262 | 44.3 | 0.001 |
| aitchison | sequential:disease_first | 4 | instrument | 6 | 0.0515 | 72.7 | 0.001 |
| aitchison | sequential:disease_first | 5 | region | 7 | 0.0135 | 16.3 | 0.001 |
| aitchison | sequential:disease_first | 6 | depth_bin | 4 | 0.0146 | 31.0 | 0.001 |
| aitchison | sequential:disease_first | 7 | library_source | 1 | 0.0096 | 81.0 | 0.001 |
| aitchison | sequential:disease_first | 8 | bead_beating | 2 | 0.0200 | 84.5 | 0.001 |
| aitchison | sequential:disease_first | 9 | Residual | 6970 | 0.8232 | - | - |
| aitchison | sequential:project_vs_disease | 1 | project | 47 | 0.2515 | 49.9 | 0.001 |
| aitchison | sequential:project_vs_disease | 2 | disease_label | 1 | 0.0025 | 23.5 | 0.001 |
| aitchison | sequential:project_vs_disease | 3 | Residual | 6952 | 0.7460 | - | - |

### Project-predictability classifier (PyTorch)

- 38,645 samples, 267 projects, 419 CLR features, device `cpu`
- majority-class baseline: **0.0404**
- dropped 74 projects (< 25 samples, 1,355 samples)

| model | accuracy | balanced acc. | top-5 acc. |
|---|---|---|---|
| linear | 0.7576 ± 0.0045 | 0.6836 ± 0.0079 | 0.9380 ± 0.0016 |
| mlp | 0.7421 ± 0.0034 | 0.6430 ± 0.0047 | 0.9404 ± 0.0024 |

## Goal 1 — disease-associated genera

### Disease categories analysed

| category | projects | cases | controls |
|---|---|---|---|
| cancer_other | 4 | 657 | 303 |
| IBD | 4 | 557 | 151 |
| HIV | 3 | 359 | 148 |
| asthma | 3 | 248 | 220 |
| NAFLD | 3 | 104 | 111 |
| T1D | 2 | 100 | 123 |

### Concordance: meta-analysis vs pooled combat_ols

| category | replicated | pooling-only | within-only | effect ρ | sign agree |
|---|---|---|---|---|---|
| HIV | 8 | 33 | 0 | 0.43 | 71% |
| IBD | 76 | 40 | 5 | 0.98 | 97% |
| NAFLD | 8 | 8 | 2 | 0.86 | 87% |
| T1D | 0 | 0 | 0 | 0.98 | 96% |
| asthma | 0 | 0 | 0 | 0.93 | 89% |
| cancer_other | 0 | 3 | 2 | -0.76 | 15% |

### Batch leakage: disease-classifier AUC by CV scheme

| category | model | leave-project-out | random k-fold | gap |
|---|---|---|---|---|
| HIV | gboost | 0.532 | 0.785 | +0.253 |
| HIV | logreg | 0.542 | 0.680 | +0.138 |
| IBD | gboost | 0.730 | 0.819 | +0.089 |
| IBD | logreg | 0.701 | 0.747 | +0.047 |
| NAFLD | gboost | 0.673 | 0.757 | +0.084 |
| NAFLD | logreg | 0.626 | 0.761 | +0.134 |
| T1D | gboost | 0.480 | 0.579 | +0.100 |
| T1D | logreg | 0.401 | 0.614 | +0.213 |
| asthma | gboost | 0.546 | 0.605 | +0.059 |
| asthma | logreg | 0.588 | 0.598 | +0.010 |
| cancer_other | gboost | 0.566 | 0.710 | +0.144 |
| cancer_other | logreg | 0.595 | 0.640 | +0.044 |

## Robustness checks

### Label-shuffling null

Case/control permuted within every (project, category); the meta-analysis and pooled test re-run. Under a working pipeline the FDR hit count collapses to near zero.

| | meta hits (Σ over categories) | pooled hits |
|---|---|---|
| real labels | 101 | 176 |
| shuffled (mean ± max over 10 permutations) | 0.1 ± 1 | 1.3 ± 4 |

### Depth-threshold sensitivity

Within-study cohort rebuilt at each minimum-reads cut; replicated genera per category:

| category | 5,000 | 10,000 | 20,000 |
|---|---|---|---|
| HIV | 7 | 8 | 21 |
| IBD | 78 | 76 | 78 |
| NAFLD | 8 | 8 | 0 |
| T1D | 0 | 0 | 0 |
| asthma | 0 | 0 | 0 |
| cancer_other | 0 | 0 | 0 |

79 of 112 replicated genera hold at every threshold (`validation_depth_sensitivity.csv` lists the rest).

### Transform sensitivity (CLR vs log-relative abundance)

| category | meta hits CLR | meta hits log-rel | shared | effect ρ |
|---|---|---|---|---|
| cancer_other | 2 | 1 | 1 | 0.99 |
| IBD | 81 | 76 | 59 | 0.99 |
| HIV | 8 | 26 | 7 | 1.00 |
| asthma | 0 | 0 | 0 | 0.99 |
| NAFLD | 10 | 8 | 8 | 1.00 |
| T1D | 0 | 0 | 0 | 0.99 |

### Prevalence-filter sensitivity

| category | ≥1% | ≥10% |
|---|---|---|
| HIV | 8 | 8 |
| IBD | 76 | 75 |
| NAFLD | 8 | 7 |
| T1D | 0 | 0 |
| asthma | 0 | 0 |
| cancer_other | 0 | 0 |

### Literature cross-check

Replicated hits vs curated genus-direction expectations (orientation only — not extracted from these studies' papers; verify against the DOIs in `literature_crosscheck.csv`).

| category | checked | concordant | discordant | no prior |
|---|---|---|---|---|
| HIV | 8 | 1 | 0 | 7 |
| IBD | 25 | 8 | 0 | 17 |
| NAFLD | 8 | 0 | 0 | 8 |

## Reading the result

The centrepiece is the *replicated* set: genera significant in the within-study meta-analysis **and** the naive pooled test. Genera significant only when samples are pooled across labs are flagged as likely batch artifacts, not findings. The batch-leakage gap (random CV minus leave-one-project-out CV) quantifies, per disease, how much of a classifier's apparent accuracy is study recognition rather than biology.

## Discussion and conclusions

### Goal 2 — how much of what we see is the lab

The answer is: most of it. Study identity accounts for roughly 40% of between-sample variation; disease status accounts for about 1%, and adds only ~0.2 percentage points once the technical factors are fitted first (`sequential:technical_first`). Fitted first instead, disease reaches ~1% — the gap between those two numbers (0.16% vs 0.97%) is the honest uncertainty band on any pooled disease effect. The ordering is not close, and it is not an artifact of one distance metric or one cohort: Bray-Curtis and Aitchison agree, and the labelled sub-cohort shows the same shape. The project-prediction classifier makes the point a second way — the lab signature is not subtle structure that careful modelling might remove, it is the dominant, linearly separable axis of the data. Sequencing depth is a further confounder specifically for richness-based comparisons (Spearman correlation with observed richness ≈ 0.32, still ≈ 0.25 after rarefaction), so depth has to be controlled wherever alpha diversity enters an analysis.

The practical consequence, which the disease results then bear out, is that cross-study pooling in this compendium cannot be trusted on its own. It is not that pooling is always wrong — for IBD it happened to agree closely with the batch-safe analysis — but that pooling alone gives no way to tell agreement from artifact.

### Goal 1 — disease-associated genera

Six disease categories had enough within-study case/control data to analyse (IBD, HIV, NAFLD, T1D, asthma, and a mixed "other cancers" group). The per-category verdicts are in *Findings in brief* above; the shape of the overall result is that **the batch-safe design is far more conservative than the pooled one, and the gap between them is the useful measurement.** IBD replicates almost completely (76 genera, ρ = 0.98). HIV replicates partially with a large pooled-only excess (8 vs 33, ρ = 0.43). NAFLD replicates weakly (8, and threshold-fragile). Cancer anti-replicates (ρ = −0.76). T1D and asthma produce nothing in either direction that survives correction.

Only IBD, then, supports a genus-level claim from this data — and that claim is a replication of existing knowledge rather than a new discovery: loss of fibre-fermenting, butyrate-producing commensals and a shift toward oral-derived and facultatively anaerobic taxa. Its value here is that it was recovered from four independent studies with near-zero heterogeneity under a design that structurally cannot import batch effects; it is a strong internal positive control for the pipeline as much as a biological result. The label-shuffling null (101 real meta hits → ~0.1 per shuffled permutation) confirms the q-values driving it are not a multiple-testing or degrees-of-freedom artifact.

Many of the significant genera carry placeholder names (`UCG-005`, `Christensenellaceae R-7 group`, `NK4A214 group`, and numerous `Incertae Sedis` entries within Oscillospiraceae and Ruminococcaceae). This is a limit of genus-level 16S with incomplete reference taxonomy, and it caps how far the biological interpretation can go even for IBD.

### The methodological findings

Three results here are methodological, and are arguably more transferable than the biology:

1. **Health labels in public microbiome metadata are not findable by any single rule.** The label audit catalogued six distinct naming families — from the MIxS `*_disord` fields, through semantically empty names like `group` and `category`, to labels sitting inside free-text `description` fields — and documented traps such as `none` meaning "no disorder" (a control) rather than missing data. Detection had to run three routes at once (generic name patterns, each study's own condition keywords, and value-vocabulary matching that ignores the field name) and still needed per-project human review. The cleaned label layer and its catalogue are a reusable artifact in their own right.
2. **Meta-analysis-vs-pooled concordance works as a routine artifact filter.** Classifying each genus as *replicated* / *pooling-only* / *within-only*, rather than reporting a single list, is what separates the IBD result from the HIV over-count and the cancer anti-correlation.
3. **The leave-one-project-out vs random-CV gap should be reported as standard.** It converts the abstract "batch effects" concern into a per-disease number, and here it shows that most apparent case/control predictability in this compendium is study recognition, not disease signal.

### What this does not establish

- **Nothing causal.** An elevated or depleted genus in a disease may be a consequence of the disease, its treatment, or an associated change in diet.
- **Nothing about species or strains** — the resolution is genus only.
- **Nothing outside a mostly Western population** (~61% Europe/North America, 36% United States).
- **Nothing about the bulk of the compendium.** Roughly seven in eight samples carry no usable per-sample health label in the harmonised table. The disease analysis rests on 41 studies / ~14,000 samples; the broader 116-study labelled catalogue identified in the audit (~18.5% of samples, in principle) was scoped but not curated, so categories that are under-powered here — T1D, asthma — might be answerable with that extension.
- **"Healthy" is each study's own definition,** harmonised at the word level only; the underlying clinical assessments are not comparable across studies.

