# Phase 2 cohort flow

## QC exclusions (fixed order)

| step | n in | n removed | n out | note |
|---|---|---|---|---|
| exclude confirmed non-human host_species | 168,464 | 503 | 167,961 | 46,005 samples with unresolved host_species (<NA>) kept, not dropped |
| exclude non-stool / missing sample_type | 167,961 | 40,800 | 127,161 | kept only sample_type == "stool" exactly; mixed body sites and missing values dropped |
| exclude depth < 10,000 reads | 127,161 | 10,504 | 116,657 | depth = row sum from sample_depth.parquet (Phase 0), not sample_metadata.total_bases |

## Cohorts (built from the QC-passed population)

| cohort | n samples | n projects |
|---|---|---|
| labeled_all | 14,954 | 48 |
| within_project | 14,033 | 41 |
| healthy_baseline | 921 | 7 |
