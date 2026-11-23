# Graph models: relational extension

Graph neural networks over the same 419 genus-level CLR features as the flat pipeline, judged under the same leave-one-project-out (LOPO) evaluation, on the same folds. Disease AUCs are averaged over folds, then over three model seeds (42, 43, 44); `±` is the seed-to-seed sd unless a column says otherwise. LOPO is the number that counts; random k-fold shares studies between training and test and is shown only to measure how much that flatters a model.

## Limitations, read these first

- **Few studies per disease.** LOPO has one fold per study: HIV 3, IBD 4, NAFLD 3, T1D 2, asthma 3, cancer_other 4. The fold-to-fold sd of a LOPO AUC is therefore large (0.02–0.17), and most differences below sit inside it. A difference that is consistent across seeds is not one that would survive a different set of studies.
- **Seed sd measures model randomness only.** Folds are fixed (built from seed 42), so the seed sd says nothing about which studies happen to be in the data. The pretrained encoder is a single run (seed 42); its downstream seed sd excludes pretraining randomness.
- **The flat gradient-boosting baseline was re-run, not read from the committed file.** Logistic regression reproduces `classifier_cv.csv` exactly; gradient boosting moves individual folds by up to ~0.035 AUC (means within fold noise). The cause is not identified. Disease tables use the re-run so that every model shares one environment.
- **No tuning.** Every neural model uses one fixed configuration (60 epochs, no early stopping, one learning rate) set before the runs. A tuned model of any kind could score higher; this compares like with like, not best achievable.
- **Three diseases have no defensible signal.** Phase 5 found nothing for T1D, asthma and cancer_other, and several of their LOPO AUCs are below 0.5. They are kept for completeness; a graph model "finding something" there is a reason for suspicion, not a result.
- **Scope.** Genus-level 16S profiles only; only GraphSAGE was pretrained; the study identity in the pretraining neighbour check is the accession prefix of the sample key.

## Answers in one table

| question | answer | evidence |
|---|---|---|
| Does a graph model encode the study as well as a flat one? (calibration) | yes, a tie | study accuracy 0.755 (GraphSAGE, G1) vs 0.757 (linear, CLR) |
| Does a sample-similarity graph leak the study into a disease model? | no, under all three protocols | G2 − G1 within ±0.016; 5 of 36 gaps beyond 2× seed sd, about what chance gives |
| Can a graph leak at all? | yes, if it is built from labels | G2 with label-built edges: AUC 1.00 even inductively |
| Does a graph model beat the flat models on honest splits? | no | 0 wins, 11 within fold noise, 7 losses (18 graph model × disease tests) |
| Does label-free pretraining on the unlabelled samples help? | small gains, mostly inside fold noise | 1 helps, 11 within fold noise, 0 hurts (12 tests); it lifts GraphSAGE to the flat models' level, not past them |
| Did pretraining teach the encoder study identity? | no more than the data has | same-study share of 10 neighbours 0.255 vs 0.300 in CLR space (chance 0.011) |

## Experiment A — calibration: predicting the study

A graph model must at least see what a flat model sees. Task: which study a sample came from (5 folds, the committed project-prediction folds). The committed flat models are shown beside their re-run. G3 contains study nodes, so it is a ceiling, not a model.

| model | inputs | role | accuracy (± fold sd) | balanced acc. | top-5 | committed accuracy | parameters |
|---|---|---|---|---|---|---|---|
| linear | CLR | flat | 0.757 ± 0.005 | 0.682 | 0.938 | 0.758 | — |
| mlp | CLR | flat | 0.742 ± 0.003 | 0.643 | 0.940 | 0.742 | — |
| linear: depth + richness only | depth, richness | flat | 0.082 ± 0.017 | 0.013 | 0.254 | — | — |
| linear: CLR + depth + richness | CLR, depth, richness | flat | 0.768 ± 0.005 | 0.694 | 0.945 | — | — |
| GraphSAGE on G1 | graph G1 | graph | 0.755 ± 0.006 | 0.678 | 0.954 | — | 490,635 |
| GraphSAGE on G3 | graph G3 (study nodes) | label in graph | 0.962 ± 0.005 | 0.942 | 0.999 | — | 627,979 |

Majority-class accuracy: 0.040.

## Experiment C — does a graph leak the study into a disease model?

The same GraphSAGE on four graphs: G1 (samples, taxa, taxonomy), G2 (+ 10 nearest-neighbour sample edges), G2 filtered (same-study neighbour edges removed), G3 (+ study nodes). Three protocols, strict to leaky: **inductive** (held-out samples only receive messages), **transductive** (they also send), **transductive_labels** (training labels are an input feature). A gap counts if it is beyond 2× its seed sd.

**The mechanism exists:** share of a sample's 10 nearest neighbours (CLR, Aitchison) from its own study: HIV 83%, IBD 76%, NAFLD 90%, T1D 93%, asthma 87%, cancer_other 85%.

| protocol | disease | split | G1 AUC | G2 − G1 | G2 filtered − G1 | G3 − G1 | flat logreg | flat gboost |
|---|---|---|---|---|---|---|---|---|
| inductive | HIV | leave project out | 0.516 | +0.001 ± 0.023 | +0.004 ± 0.006 | -0.012 ± 0.025 | 0.542 | 0.532 |
| inductive | HIV | random kfold | 0.760 | -0.000 ± 0.015 | +0.000 ± 0.014 | +0.005 ± 0.008 | 0.680 | 0.785 |
| inductive | IBD | leave project out | 0.694 | -0.010 ± 0.003 | -0.005 ± 0.008 | +0.001 ± 0.014 | 0.701 | 0.730 |
| inductive | IBD | random kfold | 0.795 | -0.001 ± 0.004 | -0.001 ± 0.005 | +0.011 ± 0.008 | 0.747 | 0.819 |
| inductive | NAFLD | leave project out | 0.707 | -0.004 ± 0.014 | -0.008 ± 0.008 | -0.012 ± 0.005 | 0.626 | 0.673 |
| inductive | NAFLD | random kfold | 0.768 | +0.003 ± 0.004 | +0.001 ± 0.008 | -0.005 ± 0.012 | 0.761 | 0.757 |
| inductive | T1D | leave project out | 0.402 | -0.006 ± 0.008 | +0.004 ± 0.005 | -0.002 ± 0.001 | 0.401 | 0.480 |
| inductive | T1D | random kfold | 0.559 | -0.005 ± 0.007 | -0.005 ± 0.020 | -0.024 ± 0.029 | 0.614 | 0.579 |
| inductive | asthma | leave project out | 0.536 | -0.009 ± 0.013 | -0.020 ± 0.017 | +0.000 ± 0.006 | 0.588 | 0.546 |
| inductive | asthma | random kfold | 0.527 | -0.005 ± 0.007 | -0.018 ± 0.008 | -0.017 ± 0.021 | 0.598 | 0.605 |
| inductive | cancer_other | leave project out | 0.522 | +0.000 ± 0.012 | +0.005 ± 0.003 | -0.006 ± 0.008 | 0.595 | 0.566 |
| inductive | cancer_other | random kfold | 0.667 | -0.005 ± 0.006 | -0.008 ± 0.010 | +0.005 ± 0.003 | 0.640 | 0.710 |
| transductive | HIV | leave project out | 0.511 | -0.000 ± 0.039 | +0.000 ± 0.013 | -0.005 ± 0.020 | 0.542 | 0.532 |
| transductive | HIV | random kfold | 0.759 | -0.002 ± 0.007 | +0.002 ± 0.012 | +0.004 ± 0.002 | 0.680 | 0.785 |
| transductive | IBD | leave project out | 0.699 | -0.010 ± 0.001 | -0.014 ± 0.001 | -0.003 ± 0.003 | 0.701 | 0.730 |
| transductive | IBD | random kfold | 0.799 | -0.009 ± 0.003 | -0.006 ± 0.002 | +0.006 ± 0.008 | 0.747 | 0.819 |
| transductive | NAFLD | leave project out | 0.705 | -0.012 ± 0.013 | -0.011 ± 0.020 | -0.012 ± 0.015 | 0.626 | 0.673 |
| transductive | NAFLD | random kfold | 0.761 | +0.006 ± 0.004 | +0.010 ± 0.002 | -0.002 ± 0.014 | 0.761 | 0.757 |
| transductive | T1D | leave project out | 0.403 | +0.006 ± 0.017 | +0.001 ± 0.012 | -0.008 ± 0.005 | 0.401 | 0.480 |
| transductive | T1D | random kfold | 0.554 | -0.006 ± 0.012 | +0.006 ± 0.006 | -0.017 ± 0.009 | 0.614 | 0.579 |
| transductive | asthma | leave project out | 0.524 | -0.005 ± 0.018 | -0.010 ± 0.006 | +0.017 ± 0.011 | 0.588 | 0.546 |
| transductive | asthma | random kfold | 0.516 | -0.015 ± 0.006 | -0.003 ± 0.013 | -0.013 ± 0.011 | 0.598 | 0.605 |
| transductive | cancer_other | leave project out | 0.550 | +0.001 ± 0.008 | +0.007 ± 0.006 | -0.001 ± 0.002 | 0.595 | 0.566 |
| transductive | cancer_other | random kfold | 0.657 | +0.003 ± 0.019 | +0.003 ± 0.010 | +0.018 ± 0.012 | 0.640 | 0.710 |
| transductive_labels | HIV | leave project out | 0.517 | +0.001 ± 0.009 | -0.012 ± 0.022 | -0.007 ± 0.019 | 0.542 | 0.532 |
| transductive_labels | HIV | random kfold | 0.740 | -0.001 ± 0.005 | +0.001 ± 0.004 | +0.002 ± 0.004 | 0.680 | 0.785 |
| transductive_labels | IBD | leave project out | 0.675 | +0.003 ± 0.005 | +0.005 ± 0.006 | +0.008 ± 0.005 | 0.701 | 0.730 |
| transductive_labels | IBD | random kfold | 0.773 | +0.001 ± 0.010 | -0.011 ± 0.003 | +0.028 ± 0.015 | 0.747 | 0.819 |
| transductive_labels | NAFLD | leave project out | 0.700 | +0.003 ± 0.007 | +0.008 ± 0.009 | +0.011 ± 0.006 | 0.626 | 0.673 |
| transductive_labels | NAFLD | random kfold | 0.772 | -0.009 ± 0.016 | -0.004 ± 0.009 | -0.013 ± 0.003 | 0.761 | 0.757 |
| transductive_labels | T1D | leave project out | 0.407 | -0.007 ± 0.010 | -0.004 ± 0.019 | -0.006 ± 0.022 | 0.401 | 0.480 |
| transductive_labels | T1D | random kfold | 0.547 | +0.005 ± 0.018 | +0.002 ± 0.007 | +0.010 ± 0.029 | 0.614 | 0.579 |
| transductive_labels | asthma | leave project out | 0.517 | +0.016 ± 0.005 | +0.017 ± 0.011 | +0.005 ± 0.015 | 0.588 | 0.546 |
| transductive_labels | asthma | random kfold | 0.511 | -0.001 ± 0.013 | -0.007 ± 0.003 | -0.012 ± 0.005 | 0.598 | 0.605 |
| transductive_labels | cancer_other | leave project out | 0.544 | -0.004 ± 0.011 | -0.002 ± 0.015 | -0.006 ± 0.012 | 0.595 | 0.566 |
| transductive_labels | cancer_other | random kfold | 0.652 | +0.001 ± 0.010 | +0.008 ± 0.013 | +0.009 ± 0.004 | 0.640 | 0.710 |

**Why there is no leak, in three checks:**

1. Neighbours' labels say no more than the study's case rate: label agreement between k-NN neighbours sits at the same-study chance level.

| disease | k-NN label agreement | chance (random) | chance (random, same study) |
|---|---|---|---|
| HIV | 0.648 | 0.587 | 0.642 |
| IBD | 0.721 | 0.664 | 0.686 |
| NAFLD | 0.604 | 0.501 | 0.551 |
| T1D | 0.532 | 0.505 | 0.508 |
| asthma | 0.526 | 0.502 | 0.503 |
| cancer_other | 0.588 | 0.568 | 0.588 |

2. The labels-as-input model ignores the label channel: scored with the neighbours' labels true, flipped or withheld (IBD, random k-fold, seed 42), its AUC barely moves.

| graph | labels true | labels flipped | labels withheld |
|---|---|---|---|
| G1 | 0.771 | 0.766 | 0.770 |
| G2 | 0.771 | 0.772 | 0.771 |
| G3 | 0.794 | 0.796 | 0.788 |

3. What a real leak looks like: G2 with its sample edges built from the labels (IBD), trained under the strict inductive protocol. The protocol guards against leaks through messages, not against a graph whose construction used the labels.

| split | G2 from labels (inductive) | G2 from composition (inductive) |
|---|---|---|
| leave project out | 1.000 | 0.683 |
| random kfold | 1.000 | 0.794 |

## Experiment B — disease classification

The flat baselines (logistic regression, gradient boosting), a GCN on one taxonomy tree per sample, the same GCN on a shuffled tree, GraphSAGE and GAT on G1 (inductive), and an MLP size-matched to each graph model.

### LOPO AUC

| disease | logistic regression | gradient boosting | MLP, GCN-sized | GCN, shuffled tree | GCN, per-sample taxonomy | MLP, SAGE-sized | GraphSAGE, G1 | MLP, GAT-sized | GAT, G1 |
|---|---|---|---|---|---|---|---|---|---|
| HIV | 0.542 | 0.532 | 0.519 | 0.517 | 0.477 | 0.549 | 0.516 | 0.545 | 0.490 |
| IBD | 0.701 | 0.728 | 0.740 | 0.741 | 0.741 | 0.739 | 0.693 | 0.739 | 0.687 |
| NAFLD | 0.626 | 0.673 | 0.706 | 0.679 | 0.636 | 0.694 | 0.707 | 0.698 | 0.637 |
| T1D | 0.401 | 0.480 | 0.406 | 0.440 | 0.522 | 0.400 | 0.402 | 0.399 | 0.402 |
| asthma | 0.588 | 0.545 | 0.630 | 0.528 | 0.531 | 0.628 | 0.536 | 0.638 | 0.528 |
| cancer_other | 0.595 | 0.577 | 0.574 | 0.544 | 0.578 | 0.576 | 0.522 | 0.568 | 0.513 |

### Random k-fold AUC (studies shared; for the inflation only)

| disease | logistic regression | gradient boosting | MLP, GCN-sized | GCN, shuffled tree | GCN, per-sample taxonomy | MLP, SAGE-sized | GraphSAGE, G1 | MLP, GAT-sized | GAT, G1 |
|---|---|---|---|---|---|---|---|---|---|
| HIV | 0.680 | 0.775 | 0.726 | 0.748 | 0.746 | 0.721 | 0.760 | 0.728 | 0.755 |
| IBD | 0.747 | 0.820 | 0.839 | 0.768 | 0.768 | 0.835 | 0.795 | 0.836 | 0.771 |
| NAFLD | 0.761 | 0.757 | 0.795 | 0.760 | 0.728 | 0.790 | 0.768 | 0.796 | 0.781 |
| T1D | 0.614 | 0.579 | 0.648 | 0.548 | 0.567 | 0.641 | 0.559 | 0.634 | 0.568 |
| asthma | 0.598 | 0.606 | 0.627 | 0.560 | 0.543 | 0.634 | 0.528 | 0.624 | 0.532 |
| cancer_other | 0.640 | 0.718 | 0.678 | 0.658 | 0.661 | 0.676 | 0.667 | 0.672 | 0.657 |

### The pre-stated test

A graph model wins only if its LOPO AUC beats the best of logistic regression, gradient boosting and its own size-matched MLP by more than its fold-to-fold sd.

| disease | graph model | LOPO AUC | best rival | rival AUC | margin | fold sd | verdict |
|---|---|---|---|---|---|---|---|
| HIV | GCN, per-sample taxonomy | 0.477 | logistic regression | 0.542 | -0.065 | 0.078 | within fold noise |
| HIV | GraphSAGE, G1 | 0.516 | MLP, SAGE-sized | 0.549 | -0.033 | 0.111 | within fold noise |
| HIV | GAT, G1 | 0.490 | MLP, GAT-sized | 0.545 | -0.055 | 0.105 | within fold noise |
| IBD | GCN, per-sample taxonomy | 0.741 | MLP, GCN-sized | 0.740 | +0.001 | 0.068 | within fold noise |
| IBD | GraphSAGE, G1 | 0.693 | MLP, SAGE-sized | 0.739 | -0.046 | 0.147 | within fold noise |
| IBD | GAT, G1 | 0.687 | MLP, GAT-sized | 0.739 | -0.051 | 0.095 | within fold noise |
| NAFLD | GCN, per-sample taxonomy | 0.636 | MLP, GCN-sized | 0.706 | -0.071 | 0.073 | within fold noise |
| NAFLD | GraphSAGE, G1 | 0.707 | MLP, SAGE-sized | 0.694 | +0.013 | 0.075 | within fold noise |
| NAFLD | GAT, G1 | 0.637 | MLP, GAT-sized | 0.698 | -0.061 | 0.045 | loss |
| T1D | GCN, per-sample taxonomy | 0.522 | gradient boosting | 0.480 | +0.043 | 0.105 | within fold noise |
| T1D | GraphSAGE, G1 | 0.402 | gradient boosting | 0.480 | -0.078 | 0.058 | loss |
| T1D | GAT, G1 | 0.402 | gradient boosting | 0.480 | -0.077 | 0.051 | loss |
| asthma | GCN, per-sample taxonomy | 0.531 | MLP, GCN-sized | 0.630 | -0.099 | 0.081 | loss |
| asthma | GraphSAGE, G1 | 0.536 | MLP, SAGE-sized | 0.628 | -0.091 | 0.026 | loss |
| asthma | GAT, G1 | 0.528 | MLP, GAT-sized | 0.638 | -0.109 | 0.073 | loss |
| cancer_other | GCN, per-sample taxonomy | 0.578 | logistic regression | 0.595 | -0.017 | 0.096 | within fold noise |
| cancer_other | GraphSAGE, G1 | 0.522 | logistic regression | 0.595 | -0.073 | 0.078 | within fold noise |
| cancer_other | GAT, G1 | 0.513 | logistic regression | 0.595 | -0.082 | 0.074 | loss |

### Does the real taxonomy matter?

GCN on the real tree minus GCN on a shuffled tree of the same shape (a different random tree per seed), paired by seed; counted if beyond 2× its seed sd.

| disease | split | real − shuffled | reading |
|---|---|---|---|
| HIV | leave project out | -0.040 ± 0.033 | within seed noise |
| HIV | random kfold | -0.001 ± 0.017 | within seed noise |
| IBD | leave project out | -0.000 ± 0.023 | within seed noise |
| IBD | random kfold | +0.000 ± 0.028 | within seed noise |
| NAFLD | leave project out | -0.043 ± 0.057 | within seed noise |
| NAFLD | random kfold | -0.032 ± 0.006 | shuffled tree better |
| T1D | leave project out | +0.082 ± 0.033 | real tree better |
| T1D | random kfold | +0.019 ± 0.057 | within seed noise |
| asthma | leave project out | +0.002 ± 0.036 | within seed noise |
| asthma | random kfold | -0.017 ± 0.031 | within seed noise |
| cancer_other | leave project out | +0.034 ± 0.029 | within seed noise |
| cancer_other | random kfold | +0.003 ± 0.008 | within seed noise |

## Pretraining — what the unlabelled samples add

GraphSAGE pretrained without labels by masked edge reconstruction on every QC-passed sample outside the disease cohorts, so no sample later tested on was seen; 20 epochs. Then fine-tuned whole, or frozen (only the final layer learns), and compared with the random-start GraphSAGE of Experiment B on the same folds and seeds.

**Reconstruction on held-out samples' hidden edges**, against a baseline that ignores the sample (a taxon's prevalence; its mean weight):

| measure | pretrained encoder | sample-blind baseline | encoder better |
|---|---|---|---|
| existence AUC (higher is better) | 0.927 | 0.868 | yes |
| weight MSE (lower is better) | 1.624 | 2.490 | yes |

**Study identity:** share of a sample's 10 nearest neighbours from its own study (5,000 pretraining samples).

| space | same-study share |
|---|---|
| CLR composition | 0.300 |
| encoder, random weights | 0.239 |
| encoder, pretrained | 0.255 |
| chance | 0.011 |

**Disease, LOPO:** pretrained minus random start, counted if beyond the random-start model's fold-to-fold sd.

| disease | variant | random start | pretrained | Δ | fold sd | verdict |
|---|---|---|---|---|---|---|
| HIV | GraphSAGE, pretrained, fine-tuned | 0.516 ± 0.023 | 0.545 ± 0.008 | +0.029 | 0.111 | within fold noise |
| HIV | GraphSAGE, pretrained, frozen | 0.516 ± 0.023 | 0.540 ± 0.009 | +0.024 | 0.111 | within fold noise |
| IBD | GraphSAGE, pretrained, fine-tuned | 0.693 ± 0.009 | 0.736 ± 0.002 | +0.043 | 0.147 | within fold noise |
| IBD | GraphSAGE, pretrained, frozen | 0.693 ± 0.009 | 0.720 ± 0.006 | +0.027 | 0.147 | within fold noise |
| NAFLD | GraphSAGE, pretrained, fine-tuned | 0.707 ± 0.004 | 0.653 ± 0.005 | -0.054 | 0.075 | within fold noise |
| NAFLD | GraphSAGE, pretrained, frozen | 0.707 ± 0.004 | 0.651 ± 0.002 | -0.056 | 0.075 | within fold noise |
| T1D | GraphSAGE, pretrained, fine-tuned | 0.402 ± 0.008 | 0.408 ± 0.011 | +0.006 | 0.058 | within fold noise |
| T1D | GraphSAGE, pretrained, frozen | 0.402 ± 0.008 | 0.452 ± 0.009 | +0.050 | 0.058 | within fold noise |
| asthma | GraphSAGE, pretrained, fine-tuned | 0.536 ± 0.013 | 0.591 ± 0.011 | +0.054 | 0.026 | helps |
| asthma | GraphSAGE, pretrained, frozen | 0.536 ± 0.013 | 0.527 ± 0.024 | -0.009 | 0.026 | within fold noise |
| cancer_other | GraphSAGE, pretrained, fine-tuned | 0.522 ± 0.032 | 0.592 ± 0.006 | +0.070 | 0.078 | within fold noise |
| cancer_other | GraphSAGE, pretrained, frozen | 0.522 ± 0.032 | 0.561 ± 0.008 | +0.039 | 0.078 | within fold noise |

**Random k-fold minus LOPO** (how much a shared-study split flatters each model):

| disease | logistic regression | gradient boosting | GraphSAGE, G1 | GraphSAGE, pretrained, fine-tuned | GraphSAGE, pretrained, frozen |
|---|---|---|---|---|---|
| HIV | +0.138 | +0.243 | +0.244 | +0.182 | +0.192 |
| IBD | +0.047 | +0.092 | +0.102 | +0.075 | +0.034 |
| NAFLD | +0.134 | +0.084 | +0.061 | +0.086 | +0.085 |
| T1D | +0.213 | +0.100 | +0.157 | +0.223 | +0.097 |
| asthma | +0.010 | +0.061 | -0.008 | -0.026 | -0.032 |
| cancer_other | +0.044 | +0.141 | +0.145 | +0.082 | +0.075 |

## Figures

- `figures/phase7_project_compare.png` — Experiment A
- `figures/phase7_leakage_gap.png` — Experiment C
- `figures/phase7_disease_lopo.png` — Experiment B
- `figures/phase7_pretrain_curve.png`, `figures/phase7_pretrain_lopo.png` — pretraining
