# Phase 4 -- technical variance (Goal 2)

## Marginal PERMANOVA (each factor alone)

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

## Sequential decomposition (term order matters)

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

## Project-predictability classifier (PyTorch)

- 38,645 samples, 267 projects, 419 CLR features, device `cpu`
- majority-class baseline: **0.0404**
- dropped 74 projects (< 25 samples, 1,355 samples)

| model | accuracy | balanced acc. | top-5 acc. |
|---|---|---|---|
| linear | 0.7576 ± 0.0045 | 0.6836 ± 0.0079 | 0.9380 ± 0.0016 |
| mlp | 0.7421 ± 0.0034 | 0.6430 ± 0.0047 | 0.9404 ± 0.0024 |

## Depth vs alpha diversity

| metric | column | depth | Spearman rho | p | n |
|---|---|---|---|---|---|
| observed richness | richness_raw_counts | raw | 0.316 | 0 | 29,995 |
| observed richness | richness_relative | raw | 0.316 | 0 | 29,995 |
| Shannon | shannon_relative | raw | 0.100 | 7.62e-68 | 29,995 |
| observed richness | richness_rarefied | rarefied | 0.250 | 0 | 29,995 |
| Shannon | shannon_rarefied | rarefied | 0.099 | 3.36e-66 | 29,995 |
