"""Phase 0b — profiles `Data/tags.tsv`, `Data/sample_metadata.tsv` and
`Data/projects.csv` to find every health-status label hiding in the
compendium's metadata, before Phase 1 tries to harmonise it.

Data understanding only: nothing here writes to `data/`, and nothing here
decides a sample's final label. It produces candidates and counts; the
human-curated bucket assignment lives in `config/within_study_review.csv`
(see IMPLEMENTATION.md Phase 0b step 4 -- "candidate generation and
counting are scripted", bucket assignment is by hand).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_ID_COL = "srr"


def _lower(s):
    return s.astype(str).str.strip().str.lower()


def field_census(tags_df):
    """One row per distinct tag name: how many rows/samples/projects/distinct
    values it has. Sorted by sample count, descending -- this is the map of
    what exists in tags.tsv (Phase 0b step 1)."""
    g = tags_df.groupby("tag", sort=False)
    census = g.agg(
        n_rows=("value", "size"),
        n_samples=(SAMPLE_ID_COL, "nunique"),
        n_projects=("project", "nunique"),
        n_distinct_values=("value", "nunique"),
    ).reset_index()
    return census.sort_values("n_samples", ascending=False, ignore_index=True)


def apply_null_rules(values, field_name, null_cfg):
    """Null out values.tsv's missing-data vocabulary, case-insensitively,
    with the per-field exemption in `null_cfg['conditional']` respected --
    e.g. "none" in gastrointest_disord is a control, not an absence of
    data, so it is NOT nulled there. `values` is a pandas Series of raw
    strings; returns a Series with matched entries replaced by NaN."""
    lowered = _lower(values)
    global_null = {v.lower() for v in null_cfg["global"]}
    is_null = lowered.isin(global_null)

    for token, rule in null_cfg["conditional"].items():
        token = token.lower()
        exempt = {f.lower() for f in rule["except"]}
        matches_token = lowered == token
        if field_name.lower() not in exempt:
            is_null = is_null | matches_token
        # else: this field is exempt -- the token is a real value, leave it

    out = values.copy()
    out[is_null] = np.nan
    return out


def categorical_value_profile(tags_df, projects_df, min_samples=30, value_range=(2, 25)):
    """Full value distribution for every field with `value_range` distinct
    non-null values and >= min_samples samples (~1,300 fields), each field's
    value counts alongside the projects.csv.condition of every project that
    uses it. Returns the report as one big string (Phase 0b step 2) -- this
    is what makes human review of the candidate catalogue tractable without
    grepping the raw 216 MB file."""
    lo, hi = value_range
    counts = tags_df.groupby("tag")["value"].nunique()
    eligible = set(counts[(counts >= lo) & (counts <= hi)].index)

    condition_by_project = projects_df.set_index("project")["condition"]

    # Slice to the eligible rows once, then one groupby pass. The previous
    # version re-scanned all 3.5M rows with a boolean mask
    # (`tags_df[tags_df["tag"] == tag]`) for every eligible field, which is
    # what made this run for minutes. groupby default sort keeps the same
    # alphabetical tag order as `counts.index` did.
    subset = tags_df[tags_df["tag"].isin(eligible)]

    lines = []
    for tag, sub in subset.groupby("tag", sort=True):
        n_samples = sub[SAMPLE_ID_COL].nunique()
        if n_samples < min_samples:
            continue
        value_counts = sub["value"].value_counts()
        projects_used = sub["project"].unique()
        lines.append(f"### {tag}  (n_samples={n_samples}, n_projects={len(projects_used)})")
        for val, n in value_counts.items():
            lines.append(f"  {val!r}: {n}")
        conditions = sorted({
            f"{p}: {condition_by_project.get(p, '')}" for p in projects_used
        })
        lines.append("  projects.csv.condition:")
        for c in conditions:
            lines.append(f"    {c}")
        lines.append("")

    return "\n".join(lines)


def detect_label_candidates(tags_df, projects_df, keywords_cfg, review_table=None):
    """Three detection routes, unioned, at (project, tag) granularity
    (Phase 0b step 3 / Phase 1 step 8):

    (a) generic field-NAME patterns
    (b) that project's own projects.csv.condition, abbreviation-expanded,
        matched against that project's own tag names
    (c) value-vocabulary matching, ignoring the field name entirely

    Restricted to fields with `candidate_value_range` distinct non-null
    values. The five known negative-test patterns are dropped before
    returning.

    If `review_table` (the human-curated bucket assignments) is given, its
    (project, field) pairs are unioned in unconditionally -- already-verified
    ground truth must never be silently dropped just because the automated
    heuristics don't happen to reproduce a human reviewer's judgment (e.g. a
    disease named only by a value like "carcinoma", with no generic case
    keyword anywhere near it). The `route_*` columns still report whether
    each route found it on its own, which is the signal for improving the
    heuristics over time.
    """
    lo, hi = keywords_cfg["candidate_value_range"]

    # The 2-25 distinct-value restriction is evaluated PER (project, tag),
    # not per tag across the whole compendium. A field used consistently
    # across dozens of projects (host_disease: 142 distinct strings
    # compendium-wide, gastrointest_disord: 39) naturally accumulates more
    # distinct values globally even though any single project only reports
    # a handful -- IMPLEMENTATION.md names both fields explicitly as ones
    # the restriction must not exclude, which only holds at project grain.
    pair_n_distinct = tags_df.groupby(["project", "tag"])["value"].nunique()
    in_range_index = pair_n_distinct[(pair_n_distinct >= lo) & (pair_n_distinct <= hi)].index
    pairs = pd.DataFrame(list(in_range_index), columns=["project", "tag"])

    reviewed_pairs = set()
    if review_table is not None:
        # FALSE_POSITIVE rows are deliberately-excluded known-bad pairs --
        # union in everything else (CONFIRMED, COLONISATION, NEEDS_CHECK),
        # not those.
        reviewed = review_table.loc[review_table["bucket"] != "FALSE_POSITIVE", ["project", "field"]]
        reviewed = reviewed.rename(columns={"field": "tag"})
        reviewed_pairs = set(zip(reviewed["project"], reviewed["tag"]))
        pairs = pd.concat([pairs, reviewed], ignore_index=True).drop_duplicates()

    scoped = tags_df.merge(pairs, on=["project", "tag"])

    # route (a): field-name pattern, applies to every project using that tag
    name_pattern = re.compile("|".join(keywords_cfg["generic_name_patterns"]), re.IGNORECASE)
    route_a_tags = {t for t in pairs["tag"].unique() if name_pattern.search(t)}

    # route (c): value vocabulary, ignoring field name -- scoped per
    # (project, tag), not pooled across unrelated projects that happen to
    # share a field name (two different projects' values must never
    # combine to fake a control+case match neither one has on its own).
    control_tokens = [t.lower() for t in keywords_cfg["control_value_tokens"]]
    case_tokens = [t.lower() for t in keywords_cfg["case_value_tokens"]]
    route_c_pairs = set()
    for (project, tag), vals in scoped.groupby(["project", "tag"])["value"]:
        lowered_vals = vals.dropna().astype(str).str.lower()
        has_control = lowered_vals.apply(lambda v: any(tok in v for tok in control_tokens)).any()
        has_case = lowered_vals.apply(lambda v: any(tok in v for tok in case_tokens)).any()
        if has_control and has_case:
            route_c_pairs.add((project, tag))

    # route (b): project condition -> abbreviation -> that project's tag names
    condition_by_project = projects_df.set_index("project")["condition"].fillna("")
    abbrev_map = keywords_cfg["condition_abbreviations"]
    route_b_pairs = set()
    tags_by_project = pairs.groupby("project")["tag"].apply(set)
    for project, cond in condition_by_project.items():
        cond_lower = cond.lower()
        project_tags = tags_by_project.get(project, set())
        for phrase, abbrevs in abbrev_map.items():
            if phrase in cond_lower:
                for tag in project_tags:
                    # Split on any non-alphanumeric character (including "_")
                    # rather than using \b -- a strict word boundary never
                    # fires inside "asd_numeric" because "_" counts as a
                    # word character, so \basd\b can't match the "asd" in it.
                    tag_tokens = re.split(r"[^a-z0-9]+", tag.lower())
                    if any(a in tag_tokens for a in abbrevs):
                        route_b_pairs.add((project, tag))

    # Explicit loop, not groupby().apply(...) -- when every group's result is
    # itself dict-like, pandas silently reshapes the apply() output into a
    # DataFrame instead of a Series of dicts, which breaks the plain dict
    # lookup below.
    top_values_by_pair = {}
    n_samples_by_pair = {}
    for (project, tag), grp in scoped.groupby(["project", "tag"]):
        top_values_by_pair[(project, tag)] = grp["value"].value_counts().head(5).to_dict()
        n_samples_by_pair[(project, tag)] = int(grp[SAMPLE_ID_COL].nunique())

    rows = []
    for project, tag in pairs.itertuples(index=False):
        key = (project, tag)
        ra = tag in route_a_tags
        rb = key in route_b_pairs
        rc = key in route_c_pairs
        is_reviewed = key in reviewed_pairs
        if not (ra or rb or rc or is_reviewed):
            continue
        rows.append({
            "project": project,
            "tag": tag,
            "n_samples": n_samples_by_pair.get(key, 0),
            "route_a_name_pattern": ra,
            "route_b_condition_match": rb,
            "route_c_value_vocab": rc,
            "route_reviewed_ground_truth": is_reviewed,
            "top_values": top_values_by_pair.get(key, {}),
        })

    catalogue = pd.DataFrame(rows)
    if catalogue.empty:
        return catalogue
    return _drop_negative_tests(catalogue, keywords_cfg["negative_test_patterns"])


def _drop_negative_tests(catalogue, negative_patterns):
    keep = pd.Series(True, index=catalogue.index)
    for pat in negative_patterns:
        field = pat["field"]
        project = pat.get("project")
        if project is None:
            keep &= ~(catalogue["tag"] == field)
        else:
            keep &= ~((catalogue["tag"] == field) & (catalogue["project"] == project))
    return catalogue[keep].reset_index(drop=True)


def within_study_contrasts(tags_df, candidates_df, null_cfg, review_table, min_per_side=15):
    """For each (project, tag) candidate, compute the per-field, null-aware
    value counts (mechanical), then attach the human-curated bucket +
    verified control/case counts from `review_table` where curated, or the
    literal top values with bucket=NEEDS_REVIEW where not yet reviewed
    (Phase 0b step 4). This is the union of "scripted counting" and "by
    hand" bucket assignment the spec calls for -- it does not invent a
    case/control judgment the humans haven't already made.
    """
    if candidates_df.empty:
        return candidates_df

    review = review_table.set_index(["project", "field"])
    rows = []
    for _, cand in candidates_df.iterrows():
        project, tag = cand["project"], cand["tag"]
        sub = tags_df[(tags_df["project"] == project) & (tags_df["tag"] == tag)]
        cleaned = apply_null_rules(sub["value"], tag, null_cfg)
        value_counts = cleaned.dropna().value_counts()

        key = (project, tag)
        if key in review.index:
            r = review.loc[key]
            bucket = r["bucket"]
            controls = r["controls"] if pd.notna(r["controls"]) else None
            cases = r["cases"] if pd.notna(r["cases"]) else None
            study_n = r["study_n"] if pd.notna(r.get("study_n")) else None
            note = r["note"]
        else:
            bucket, controls, cases, study_n, note = "NEEDS_REVIEW", None, None, None, ""

        rows.append({
            "project": project,
            "tag": tag,
            "bucket": bucket,
            "controls": controls,
            "cases": cases,
            "study_n": study_n,
            "n_distinct_values_after_null": len(value_counts),
            "top_values": value_counts.head(5).to_dict(),
            "note": note,
        })

    out = pd.DataFrame(rows)
    confirmed = out["bucket"] == "CONFIRMED"
    passes_threshold = (out["controls"] >= min_per_side) & (out["cases"] >= min_per_side)
    out = out[~confirmed | passes_threshold].reset_index(drop=True)
    return out


NON_HUMAN_HOST_MARKERS = [
    "mus musculus", "rhesus macaque", "simulated gastrointestinal", "labcontrol test",
]
HUMAN_HOST_VARIANTS = [
    "homo sapiens", "homo_sapiens", "homo sapiens sapiens", "homo", "homosapiens",
    "human beings", "human male adult", "infant",
]


def corruption_census(tags_df):
    """Quantify the known metadata errors (Phase 0b step 5): numeric values
    leaked into sex/age_unit, non-numeric ages, non-human host records,
    unparseable collection dates. Counts, not just examples."""
    census = {}

    for field in ("sex", "host_sex"):
        sub = tags_df[tags_df["tag"] == field]
        numeric = sub["value"].astype(str).str.match(r"^\d+$")
        census[f"{field}_numeric_leak"] = int(numeric.sum())

    for field in ("age_unit", "age units", "host_age_units"):
        sub = tags_df[tags_df["tag"] == field]
        if sub.empty:
            continue
        numeric = sub["value"].astype(str).str.match(r"^\d+(\.\d+)?$")
        census[f"{field}_numeric_leak"] = int(numeric.sum())

    for field in ("age", "host_age"):
        sub = tags_df[tags_df["tag"] == field]
        if sub.empty:
            continue
        numeric = sub["value"].astype(str).str.match(r"^\d+(\.\d+)?$")
        census[f"{field}_total"] = len(sub)
        census[f"{field}_numeric"] = int(numeric.sum())
        census[f"{field}_non_numeric"] = int((~numeric).sum())

    host = tags_df[tags_df["tag"] == "host"]
    if not host.empty:
        vc = host["value"].str.lower().value_counts()
        census["host_value_counts"] = vc.to_dict()
        non_human_mask = host["value"].str.lower().isin(NON_HUMAN_HOST_MARKERS)
        census["host_non_human_total"] = int(non_human_mask.sum())

    date_field = tags_df[tags_df["tag"] == "collection_date"]
    if not date_field.empty:
        parsed = pd.to_datetime(date_field["value"], errors="coerce")
        census["collection_date_total"] = len(date_field)
        census["collection_date_unparseable"] = int(parsed.isna().sum())

    return census


def matrix_depth_profile(interim_dir):
    """Non-zero taxa per sample, kingdom split, and depth quantiles --
    read from Phase 0's already-computed `sample_depth.parquet` and
    `taxon_table.parquet` rather than re-scanning the taxa matrix
    (Phase 0b step 6)."""
    interim_dir = Path(interim_dir)
    depth_df = pd.read_parquet(interim_dir / "sample_depth.parquet")
    taxon_df = pd.read_parquet(interim_dir / "taxon_table.parquet")

    n_nonzero = depth_df["n_nonzero"]
    depth = depth_df["depth"]
    return {
        "n_samples": len(depth_df),
        "n_nonzero_quantiles": {
            "min": int(n_nonzero.min()),
            "p10": float(n_nonzero.quantile(0.10)),
            "median": float(n_nonzero.median()),
            "p90": float(n_nonzero.quantile(0.90)),
            "max": int(n_nonzero.max()),
        },
        "depth_quantiles": {
            "min": int(depth.min()),
            "p10": float(depth.quantile(0.10)),
            "median": float(depth.median()),
            "p90": float(depth.quantile(0.90)),
            "max": int(depth.max()),
        },
        "kingdom_split": taxon_df["kingdom"].value_counts(dropna=False).to_dict(),
    }
