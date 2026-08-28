"""Phase 1 -- turns `Data/tags.tsv` + `Data/sample_metadata.tsv` +
`Data/projects.csv` into one tidy row per sample with trustworthy columns.

Consumes Phase 0b's outputs (`audit/`, `config/within_study_review.csv`)
rather than rediscovering them: candidate label fields are detected the
same way `src/profile.py` detects them for the audit, and the within-study
human curation is the source of truth for *which* field is the real label
in a project that has more than one plausible candidate (LABEL_AUDIT.md's
"wrong-tag selection" problem, e.g. PRJEB4335 has both `hiv` and `cohort`).

Nothing here decides a value's meaning implicitly -- every classification
rule traces back to a `config/*.yaml` file so it can be reviewed and
extended without touching this code.
"""
import re

import numpy as np
import pandas as pd

from src import profile as prof

SAMPLE_ID_COL = prof.SAMPLE_ID_COL  # "srr"


# ---------------------------------------------------------------------------
# Step 2: pivot the concepts in tag_map.yaml, long -> wide
# ---------------------------------------------------------------------------

def pivot_concepts(tags_df, tag_map, null_cfg):
    """One row per srr, one column per canonical concept in `tag_map`, plus
    `<concept>_source_tag` recording which raw tag won. Null handling is
    applied per source tag before coalescing, so a null value in a
    higher-priority tag correctly falls through to the next one instead of
    winning as an empty string.

    Only pivots the tags actually listed in tag_map.yaml -- pivoting all
    2,608 tags would be unusably sparse (IMPLEMENTATION.md Phase 1 step 2).
    """
    wanted_tags = {t for tags in tag_map.values() for t in tags}
    scoped = tags_df[tags_df["tag"].isin(wanted_tags)]

    out = pd.DataFrame({SAMPLE_ID_COL: tags_df[SAMPLE_ID_COL].unique()}).set_index(SAMPLE_ID_COL)
    for concept, source_tags in tag_map.items():
        value_col = pd.Series(np.nan, index=out.index, dtype=object)
        source_col = pd.Series(np.nan, index=out.index, dtype=object)
        for tag in source_tags:
            sub = scoped[scoped["tag"] == tag]
            if sub.empty:
                continue
            cleaned = prof.apply_null_rules(sub["value"], tag, null_cfg)
            cleaned = cleaned.dropna()
            # keep first occurrence per sample (duplicates are rare and not
            # meaningfully orderable -- first-seen is deterministic given a
            # stable read)
            cleaned = cleaned.groupby(sub.loc[cleaned.index, SAMPLE_ID_COL]).first()
            still_missing = value_col.index.difference(value_col.dropna().index)
            fillable = cleaned.index.intersection(still_missing)
            value_col.loc[fillable] = cleaned.loc[fillable]
            source_col.loc[fillable] = tag
        out[concept] = value_col
        out[f"{concept}_source_tag"] = source_col
    return out.reset_index()


# ---------------------------------------------------------------------------
# Step 4: repair known corruptions
# ---------------------------------------------------------------------------

_REPEATED_NOT_PROVIDED = re.compile(r"^(not provided)+$", re.IGNORECASE)
_PURE_INT = re.compile(r"^\d+$")
_PURE_NUMBER = re.compile(r"^\d+(\.\d+)?$")
_SEX_TYPOS = {"famale": "female"}


def repair_sex(sex_series):
    """Numeric values in sex are leaked ages (`"47"` x2,794, `"48"` x1,501)
    -- null them rather than guess. Also nulls the `"not providednot
    provided"` string-concatenation bug, and fixes the one observed typo."""
    s = sex_series.astype(str).str.strip()
    s = s.where(sex_series.notna())
    leaked_age = s.str.match(_PURE_INT, na=False)
    concat_bug = s.str.match(_REPEATED_NOT_PROVIDED, na=False)
    cleaned = s.mask(leaked_age | concat_bug)
    lowered = cleaned.str.lower()
    cleaned = lowered.replace(_SEX_TYPOS)
    n_leaked = int(leaked_age.sum())
    return cleaned.where(cleaned.isin(["male", "female"])), n_leaked


def repair_age_unit(age_unit_series):
    """Purely numeric age_unit values (`78` x1,663, `75` x276) are invalid
    -- null them rather than guess which unit was meant."""
    s = age_unit_series.astype(str).str.strip()
    numeric = s.str.match(_PURE_NUMBER, na=False) & age_unit_series.notna()
    cleaned = age_unit_series.mask(numeric)
    return cleaned, int(numeric.sum())


# ---------------------------------------------------------------------------
# Step 5: age -> age_years + age_confidence
# ---------------------------------------------------------------------------

_UNIT_TO_YEARS = {
    "day": 1 / 365.25, "days": 1 / 365.25,
    "week": 1 / 52.1775, "weeks": 1 / 52.1775,
    "month": 1 / 12, "months": 1 / 12,
    "year": 1.0, "years": 1.0, "yo": 1.0, "y": 1.0,
}
_EMBEDDED_UNIT_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|years|year|yo|y)\b",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|years|year|yo|y)?\b",
    re.IGNORECASE,
)
_BOUND_RE = re.compile(r"^[<>]?=?\s*(\d+(?:\.\d+)?)")
_BOUND_MARKER_RE = re.compile(r"^[<>]=?")


def _unit_years_factor(unit_text):
    if unit_text is None:
        return None
    return _UNIT_TO_YEARS.get(str(unit_text).strip().lower())


def parse_age_value(raw, unit_hint=None, inferred_unit_years=None):
    """Parse one age string into (years, confidence). `unit_hint` is the
    project's own age_unit tag if present; `inferred_unit_years` is the
    per-project fallback conversion factor used only for bare numerics with
    no unit anywhere (IMPLEMENTATION.md Phase 1 step 5). Never silently
    assumes years -- a bare numeric with no hint and no inferred factor is
    left unparsed rather than guessed.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan, None
    text = str(raw).strip()
    if not text:
        return np.nan, None

    m = _RANGE_RE.match(text)
    if m:
        lo, hi, unit = float(m.group(1)), float(m.group(2)), m.group(3)
        factor = _unit_years_factor(unit) or _unit_years_factor(unit_hint) or 1.0
        return (lo + hi) / 2 * factor, "range"

    m = _BOUND_MARKER_RE.match(text)
    if m:
        m2 = _BOUND_RE.match(text)
        factor = _unit_years_factor(unit_hint) or 1.0
        return float(m2.group(1)) * factor, "bound"

    m = _EMBEDDED_UNIT_RE.match(text)
    if m:
        value, unit = float(m.group(1)), m.group(2)
        return value * _unit_years_factor(unit), "exact"

    if _PURE_NUMBER.match(text):
        value = float(text)
        factor = _unit_years_factor(unit_hint)
        if factor is not None:
            return value * factor, "exact"
        if inferred_unit_years is not None:
            return value * inferred_unit_years, "inferred"
        return np.nan, None  # no unit anywhere -- do not guess

    return np.nan, None  # unparseable free text (e.g. "child 2-year visit")


def clean_bmi(bmi_series):
    """Numeric coercion; non-numeric garbage becomes null rather than
    guessed at."""
    return pd.to_numeric(bmi_series, errors="coerce")


def infer_project_age_unit(subjects_text):
    """Per-project fallback unit when a bare numeric age has no explicit
    unit tag: `subjects` mentioning infants implies months, else years."""
    if not isinstance(subjects_text, str):
        return 1.0
    if re.search(r"infant|neonat|newborn", subjects_text, re.IGNORECASE):
        return 1 / 12
    return 1.0


def parse_ages(wide_df, projects_df):
    """Vectorised wrapper around `parse_age_value` for the whole spine.
    Returns (age_years, age_confidence) Series aligned to wide_df's index.
    """
    unit_factor_by_project = (
        projects_df.set_index("project")["subjects"].apply(infer_project_age_unit)
    )
    inferred_factor = wide_df["project"].map(unit_factor_by_project)

    years, conf = [], []
    for raw, unit_hint, factor in zip(wide_df["age"], wide_df["age_unit"], inferred_factor):
        y, c = parse_age_value(raw, unit_hint, factor)
        years.append(y)
        conf.append(c)
    years = pd.Series(years, index=wide_df.index)
    conf = pd.Series(conf, index=wide_df.index)
    # sanity bound (IMPLEMENTATION.md Phase 1 acceptance: 0 <= age_years <= 120)
    out_of_range = years.notna() & ((years < 0) | (years > 120))
    years = years.mask(out_of_range)
    conf = conf.mask(out_of_range)
    return years, conf, int(out_of_range.sum())


# ---------------------------------------------------------------------------
# Step 6: host_species normalisation
# ---------------------------------------------------------------------------

_HUMAN_PREFIXES = ("homo sapiens", "homo_sapiens", "human")
_HUMAN_EXACT = {"homo", "homosapiens", "infant"}
_NON_HUMAN_SUBSTRINGS = (
    "mus musculus", "mice", "mouse", "rhesus macaque", "macaque",
    "simulated gastrointestinal", "labcontrol test", "triticum aestivum",
)


def classify_host_species(host_series):
    """True = human, False = non-human, None = free text that names neither
    (e.g. a demographic description with no species word) -- left for
    review rather than guessed either way. `host` is genuinely free text
    (204 distinct values including plant names and per-subject codes like
    "human a2"), so this is prefix/substring matching, not an exact-value
    lookup like the sex/host corruption checks above.
    """
    lowered = host_series.astype(str).str.strip().str.lower()
    lowered = lowered.where(host_series.notna())

    is_non_human = pd.Series(False, index=host_series.index)
    for marker in _NON_HUMAN_SUBSTRINGS:
        is_non_human |= lowered.str.contains(marker, na=False, regex=False)

    is_human = lowered.isin(_HUMAN_EXACT)
    for prefix in _HUMAN_PREFIXES:
        is_human |= lowered.str.startswith(prefix, na=False)
    is_human &= ~is_non_human

    result = pd.Series(pd.NA, index=host_series.index, dtype=object)
    result[is_human] = True
    result[is_non_human] = False
    return result


# ---------------------------------------------------------------------------
# Step 7: collection_date -> collection_year
# ---------------------------------------------------------------------------

_LEADING_YEAR_RE = re.compile(r"(\d{4})")
_YEAR_BOUNDS = (1990, 2022)  # matches the compendium's known collection window


def parse_collection_year(date_series, tag_name, null_cfg):
    """Year extraction with a fallback for MIxS date-RANGE values
    (`"2011-01-01/2014-12-31"`, `"2013-07/2015-07"`, `"2010/2012"`) that
    `pd.to_datetime` can't parse as a single date: take the first four-digit
    year token, i.e. the start of the range.

    Applies the null vocabulary first -- IMPLEMENTATION.md's ~45,402
    "unparseable" figure turns out to conflate genuinely-missing values
    ("na", "missing", "not applicable", ...) with actual parse failures;
    once those are stripped, only a handful of true garbage values
    (`"-"`, `"0000-00-00"`) remain unparseable.
    """
    cleaned = prof.apply_null_rules(date_series, tag_name, null_cfg)
    parsed = pd.to_datetime(cleaned, errors="coerce", format="mixed", utc=True)
    years = parsed.dt.year.astype("Float64")

    still_missing = years.isna() & cleaned.notna()
    leading_year = pd.to_numeric(
        cleaned[still_missing].str.extract(_LEADING_YEAR_RE, expand=False), errors="coerce"
    )
    in_bounds = leading_year.between(*_YEAR_BOUNDS)
    years.loc[still_missing] = leading_year.where(in_bounds)
    return years


# ---------------------------------------------------------------------------
# Step 8: disease_label
# ---------------------------------------------------------------------------

def _subset_sum_search(items, target):
    """`items`: list of (value, count). Returns the sub-list summing exactly
    to `target`, or None if no such sub-list exists. Plain DFS with sum
    pruning (descending order, stop at the first match, abandon a branch
    once the running sum would over/undershoot) -- confirmed label fields
    have at most a handful of distinct values (LABEL_AUDIT.md's 2-25 range
    restriction, and CONFIRMED ones in practice are far smaller), so this
    is fast despite being worst-case exponential.
    """
    if target == 0:
        return []
    items = sorted(items, key=lambda kv: -kv[1])
    result = None

    def dfs(i, remaining, chosen):
        nonlocal result
        if result is not None or remaining < 0:
            return
        if remaining == 0:
            result = list(chosen)
            return
        if i >= len(items):
            return
        chosen.append(items[i])
        dfs(i + 1, remaining - items[i][1], chosen)
        chosen.pop()
        if result is None:
            dfs(i + 1, remaining, chosen)

    dfs(0, target, [])
    return result


def resolve_value_labels(value_counts, controls, cases):
    """Reconstructs which raw values are controls and which are cases for
    one confirmed (project, tag) field, using the human-verified totals in
    `config/within_study_review.csv` as ground truth rather than guessing
    from keyword vocabulary.

    This matters because most confirmed fields do NOT use recognisable
    control/case words -- they use numeric codes (`"0"`/`"1"`), abbreviations
    (`"hc"`, `"nc"`, `"cd"`, `"uc"`), booleans (`"true"`/`"false"`), or a
    study-specific split whose direction isn't guessable from the value
    text at all (e.g. PRJNA307992's C. diff `patient_group`: `nonrecurrent`
    is the confirmed control side, `reinfection` the confirmed case side --
    no generic vocabulary rule could know that without the human count).

    Finds a partition of the field's value distribution into a
    controls-summing subset and a (disjoint) cases-summing subset by exact
    subset-sum against the counts LABEL_AUDIT.md's review already verified.
    Leftover values (e.g. a third non-binary category) map to neither --
    this is the documented gap between a field's full population and its
    strict binary contrast. Returns None if no exact partition exists,
    which the caller treats as "investigate", not "guess".
    """
    items = list(value_counts.items())
    control_items = _subset_sum_search(items, int(controls))
    if control_items is None:
        return None
    used_values = {v for v, _ in control_items}
    remaining = [(v, c) for v, c in items if v not in used_values]
    case_items = _subset_sum_search(remaining, int(cases))
    if case_items is None:
        return None
    mapping = {v: "healthy" for v, _ in control_items}
    mapping.update({v: "case" for v, _ in case_items})
    return mapping


def categorize_disease_value(value, condition_fallback, condition_cfg):
    """Disease category for a case value: use the raw value's own words
    when they match a controlled category, else fall back to the study's
    normalized `condition` (from `normalize_condition`)."""
    v = str(value).strip().lower()
    for rule in condition_cfg["rules"]:
        if any(m in v for m in rule["match"]):
            return rule["category"]
    return condition_fallback


def resolve_healthy_only_labels(value_counts, keywords_cfg, tag):
    """All-or-nothing counterpart to `resolve_value_labels` for a
    HEALTHY_ONLY field: every non-null value must be a control-token /
    disorder-family-control value (the same vocabulary `resolve_value_labels`
    treats as a hint, used here as the actual check since a HEALTHY_ONLY
    field has no case side to reconstruct). Returns a `{value: "healthy"}`
    mapping only if ALL values pass; otherwise None, so a field with a
    hidden non-control value (data drift, or a curation mistake) is
    rejected rather than partially trusted.
    """
    control_tokens = [t.lower() for t in keywords_cfg["control_value_tokens"]]
    disorder_controls = {t.lower() for t in keywords_cfg["disorder_family_control_values"]}
    is_disorder_field = tag in keywords_cfg["disorder_family_fields"]

    for value in value_counts:
        v = str(value).strip().lower()
        ok = v in disorder_controls if is_disorder_field else _substring_classify(v, control_tokens)
        if not ok:
            return None
    return {v: "healthy" for v in value_counts}


def _substring_classify(value_lower, tokens):
    return any(tok in value_lower for tok in tokens)


def derive_disease_label(tags_df, null_cfg, keywords_cfg, review_df,
                          condition_cfg, condition_normalized):
    """Per-sample disease_label ("healthy"/"case"), disease_category and
    disease_field, for every sample in a project with a human-curated
    label field.

    Two buckets from `config/within_study_review.csv` feed this:

    - **CONFIRMED** (53 projects): within-study case/control contrasts --
      `within_project`, Phase 2's Way A cohort. Value-to-label resolution
      uses `resolve_value_labels` (exact subset-sum against the review's
      verified controls/cases counts), not keyword matching: most of these
      fields code the contrast as numbers, abbreviations or study-specific
      terms with no generic control/case vocabulary in them at all -- see
      that function's docstring for a concrete example.
    - **HEALTHY_ONLY** (9 projects): studies with `projects.csv.condition`
      exactly `"healthy"` and a per-sample field that is 100% a control
      value after null handling (no case side exists to reconstruct, so
      `resolve_healthy_only_labels`'s all-or-nothing check is used instead).
      Feeds Phase 2's `healthy_baseline`, in place of the unreliable
      study-level `condition` field.

    Scope decision: only these 62 curated (project, field) pairs are used.
    The remaining ~63 candidate fields in `audit/health_field_catalogue.csv`
    are still bucketed `NEEDS_REVIEW`: auto-assigning labels there without
    the same value-level check re-creates the "wrong tag" / "confident
    false positive" failure modes LABEL_AUDIT.md Section 6 documents.
    Extending `within_study_review.csv` to cover them is the natural next
    step to reach the full `labeled_all` / Way B cohort (31,239 across 116
    projects) -- flagged here rather than approximated.

    A field whose current value distribution can't be resolved against its
    reviewed bucket (data drift since curation) is skipped with a warning
    rather than guessed. Samples outside these projects, or whose value
    didn't survive null handling, get all-null disease columns.
    """
    condition_by_project = condition_normalized.set_index("project")["condition_category"]

    rows = []
    unresolved = []
    curated = review_df[review_df["bucket"].isin(["CONFIRMED", "HEALTHY_ONLY"])]
    for row in curated.itertuples(index=False):
        project, tag, bucket = row.project, row.field, row.bucket
        sub = tags_df[(tags_df["project"] == project) & (tags_df["tag"] == tag)]
        cleaned = prof.apply_null_rules(sub["value"], tag, null_cfg)
        value_counts = cleaned.value_counts().to_dict()

        if bucket == "CONFIRMED":
            mapping = resolve_value_labels(value_counts, row.controls, row.cases)
        else:
            mapping = resolve_healthy_only_labels(value_counts, keywords_cfg, tag)
        if mapping is None:
            unresolved.append((project, tag, bucket))
            continue

        fallback_cat = condition_by_project.get(project, "other")
        for srr, value in zip(sub[SAMPLE_ID_COL], cleaned):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            label = mapping.get(value)
            if label is None:
                continue
            category = categorize_disease_value(value, fallback_cat, condition_cfg) \
                if label == "case" else None
            rows.append((srr, project, label, category, tag))

    out = pd.DataFrame(
        rows, columns=[SAMPLE_ID_COL, "project", "disease_label", "disease_category",
                        "disease_field"],
    )
    return out, unresolved


# ---------------------------------------------------------------------------
# Step 8 (self-report battery): wide <field>_status / <field>_evidence pairs
# ---------------------------------------------------------------------------

def _sanitize(tag):
    return re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")


def derive_self_report_columns(tags_df, keywords_cfg):
    """`<field>_evidence` (ordinal 0-3, NA = unqualified positive) and
    `<field>_status` (binary; 1 only for evidence==3) per self-report field,
    wide-format, one pair per condition (LABEL_AUDIT.md 5.1). Populated
    only for the ~4,842 American Gut samples that carry these tags at all.
    `fungal_overgrowth` is skipped entirely per the exclusion decision.
    """
    vocab = {k.lower(): v for k, v in keywords_cfg["self_report_vocab"].items()}
    binary_merge = keywords_cfg.get("self_report_binary_merge", {})
    excluded = set(keywords_cfg.get("self_report_excluded_fields", []))
    fields = [f for f in keywords_cfg["self_report_fields"] if f not in excluded]

    out = pd.DataFrame({SAMPLE_ID_COL: tags_df[SAMPLE_ID_COL].unique()}).set_index(SAMPLE_ID_COL)
    for field in fields:
        sub = tags_df[tags_df["tag"] == field]
        if sub.empty:
            continue
        lowered = sub["value"].astype(str).str.strip().str.lower()
        merge_map = {k.lower(): v for k, v in binary_merge.get(field, {}).items()}
        evidence = lowered.map(lambda v: merge_map[v] if v in merge_map else vocab.get(v, np.nan))
        evidence = pd.Series(evidence.to_numpy(), index=sub[SAMPLE_ID_COL].to_numpy())
        evidence = evidence.groupby(level=0).first()

        name = _sanitize(field)
        out[f"sr_{name}_evidence"] = evidence.reindex(out.index)
        status = pd.Series(np.nan, index=out.index)
        status[out[f"sr_{name}_evidence"] == 0] = 0
        status[out[f"sr_{name}_evidence"] == 3] = 1
        out[f"sr_{name}_status"] = status
    return out.reset_index()


# ---------------------------------------------------------------------------
# Step 9: projects.csv normalisation
# ---------------------------------------------------------------------------

def _apply_rules(value, rules_cfg, key="category"):
    if not isinstance(value, str) or not value.strip():
        return rules_cfg["default"]
    v = value.strip().lower()
    for rule in rules_cfg["rules"]:
        if any(m in v for m in rule["match"]):
            return rule[key]
    return rules_cfg["default"]


def normalize_condition(projects_df, condition_cfg):
    out = projects_df[["project", "condition"]].copy()
    out["condition_category"] = out["condition"].apply(lambda v: _apply_rules(v, condition_cfg))
    return out


def normalize_kit(projects_df, kit_cfg):
    out = projects_df[["project", "kit"]].copy()
    out["kit_family"] = out["kit"].apply(lambda v: _apply_rules(v, kit_cfg, key="family"))
    return out


# ---------------------------------------------------------------------------
# Step 10: join everything onto the 168,464-row spine
# ---------------------------------------------------------------------------

def build_harmonized(sample_metadata_df, wide_concepts_df, disease_df, self_report_df,
                      condition_norm_df, kit_norm_df, projects_df):
    """One row per sample (`srr`), joining the sample_metadata spine with
    every harmonised piece built above. Left joins throughout -- a sample
    missing a piece (e.g. no disease label) keeps its row with nulls in
    those columns, it is never dropped here.
    """
    spine = sample_metadata_df.copy()
    n0 = len(spine)

    df = spine.merge(wide_concepts_df, on=SAMPLE_ID_COL, how="left", suffixes=("", "_wide"))
    df = df.merge(
        disease_df.drop(columns=["project"]), on=SAMPLE_ID_COL, how="left",
    )
    df = df.merge(self_report_df, on=SAMPLE_ID_COL, how="left")
    df = df.merge(condition_norm_df.drop(columns=["condition"]), on="project", how="left")
    df = df.merge(kit_norm_df.drop(columns=["kit"]), on="project", how="left")
    df = df.merge(
        projects_df[["project", "amplicon", "avg_length", "bead_beating", "sample_type"]],
        on="project", how="left",
    )
    df["amplicon"] = df["amplicon"].replace("", np.nan).fillna("unknown")

    assert len(df) == n0, f"join changed row count: {n0} -> {len(df)}"
    assert df[SAMPLE_ID_COL].notna().all() and df["project"].notna().all()
    assert not df[SAMPLE_ID_COL].duplicated().any(), "join introduced duplicate samples"
    return df


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def coverage_report(df, concepts):
    """Non-null count and % per canonical concept, before/after repair --
    itself a deliverable (IMPLEMENTATION.md Phase 1 outputs)."""
    n = len(df)
    lines = ["| column | non-null | % |", "|---|---|---|"]
    for col in concepts:
        if col not in df.columns:
            continue
        non_null = int(df[col].notna().sum())
        lines.append(f"| `{col}` | {non_null:,} | {non_null / n:.1%} |")
    return "\n".join(lines)
