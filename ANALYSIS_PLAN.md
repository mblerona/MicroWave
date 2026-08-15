# Analysis Plan — Human Microbiome Compendium

**A plain-language guide to what we have, what we want to find out, and how we intend to do it.**

This document is written to be read by anyone on the team, including people who are not doing the coding. The technical companion — the actual step-by-step build instructions — lives in [IMPLEMENTATION.md](IMPLEMENTATION.md).

Every number in this document was measured directly from the files in `Data/`, not copied from the dataset's own README. Where a figure is an estimate from a sample rather than a full count, it says so.

---

## 1. What this dataset is

The Human Microbiome Compendium is a large collection of gut bacteria measurements, gathered from 482 separate published research studies and re-processed so they can be compared with each other.

**The basic unit is a stool sample.** Someone donated a stool sample, a lab extracted the DNA from it, and a machine counted how many bacteria of each kind were present. Do that 168,464 times and you have this dataset.

There are four files:

| File | Size | What's in it |
|---|---|---|
| `Data/raw_taxa_110.csv` | 1.6 GB | The measurements. 168,464 rows (one per sample) × 4,680 columns (one per bacteria type). Each cell is a count. |
| `Data/sample_metadata.tsv` | 23 MB | One row per sample: which study it came from, which machine sequenced it, which country, when it was published. |
| `Data/projects.csv` | 69 KB | One row per study (482 total): what disease the study was about, what lab kit they used, a link to the paper. |
| `Data/tags.tsv` | 216 MB | Everything else the original researchers recorded about each sample — age, sex, diet, diagnosis, and 2,600 other things. 3.5 million entries. |

**Good news up front:** the files line up perfectly. Every one of the 168,464 samples in the measurement file has exactly one matching row in the metadata file, and vice versa — no orphans in either direction. This is rarer than it sounds and saves us a lot of trouble.

### What a "bacteria type" means here

The 4,680 columns are *genera* — a level of biological grouping roughly comparable to calling something a "dog" rather than a "golden retriever". We can say *Bifidobacterium* was present; we cannot say *which* Bifidobacterium. That is a real limit on how specific our conclusions can be, and we will state it in the final report.

### The measurements are mostly zeros

This surprises people, so it's worth stating clearly: **a typical sample contains only about 45 different bacteria types out of the 4,680 possible.** The other ~4,635 columns are zero for that sample.

- Only **10** bacteria types appear in more than half of all samples
- **421** appear in at least 1% of samples
- Roughly **2,068** of the 4,680 types were never seen at all in a 4,000-sample probe (to be confirmed against the full file)

This is normal for this kind of data — the 4,680 columns cover every organism anyone has ever detected across all 482 studies, and no single gut contains more than a fraction of them. But it shapes everything downstream: most standard statistical methods assume data that is not 99% zeros, so we have to choose methods that handle this properly.

---

## 2. What we want to find out

We are committing to **two goals**, pursued thoroughly, rather than many goals pursued shallowly.

### Goal 1 (the main one) — Which bacteria genuinely differ between sick and healthy people?

Given a disease — inflammatory bowel disease, colorectal cancer, Parkinson's, and others present in this data — which gut bacteria are reliably more common, or less common, in people who have it?

That word **genuinely** is carrying a lot of weight, so it's worth spelling out. Producing a list of bacteria that *look* different is easy — in fact it's too easy. A bacterium can look different for two reasons that have nothing to do with disease.

**Reason 1: luck, because we are testing thousands of things at once.** We have 4,680 bacteria types. If we test every one of them and accept any result that clears the usual "less than a 1 in 20 chance" bar, then roughly **234 of them will clear that bar by chance alone** — even if sick and healthy people had identical gut bacteria. That number is just 4,680 divided by 20. Hand someone a list of 234 bacteria and it looks like a major discovery; it is noise.

The everyday version of this: measure 4,680 things about two school classes — height, shoe size, favourite colour, birth month, distance from home. You are guaranteed to find some differences. That does not mean the two classes are genuinely different. It means you measured a lot of things.

(Filtering down to the few hundred bacteria that are actually common, which we do in Phase 0, shrinks this problem but does not remove it. Proper correction for multiple testing is required regardless, and is built into every phase.)

**Reason 2: the lab, not the person.** This is the bigger problem here, and Section 3 is devoted to it.

So **genuinely** means: survives both objections. The practical test is whether a different research team, studying different patients in a different lab, would find the same bacteria. Goal 1 aims to produce a list that passes that test. It will be shorter and more cautious than a list of everything that merely looks different — and that shortness is the point of the exercise, not a shortcoming of it.

### Goal 2 (the supporting one) — How much of what we see is the lab rather than the person?

Before we can trust any answer to Goal 1, we need to know how much of the variation between samples is caused not by the people who donated them, but by *how the samples were processed*.

Different studies used different chemical kits to extract DNA, targeted different sections of the bacterial genome, and ran different sequencing machines. All of these change the numbers you get out, even if you fed in two identical stool samples.

Goal 2 measures the size of that effect. **We do Goal 2 first**, because its answer tells us how skeptical to be about Goal 1.

---

## 3. The main obstacle, explained plainly

Here is the central problem with this dataset, and most of our methodology exists to deal with it.

**Most studies collected either sick people or healthy people — not both.** A study about Crohn's disease recruited Crohn's patients. A study about healthy ageing recruited healthy volunteers. They were separate studies, run by separate teams, in separate countries, using separate lab kits.

So when we pool everything together and find that some bacterium is higher in Crohn's patients, there are two possible explanations and the data alone cannot distinguish them:

1. Crohn's disease really does raise that bacterium, **or**
2. The Crohn's study happened to use a DNA extraction kit that reports more of that bacterium

This is called **confounding**: the thing we care about (disease) is tangled up with something we don't (which lab did the work). If we ignore it we will publish a list of "disease bacteria" that is partly a list of "lab equipment bacteria".

### How we handle it: run the comparison two ways and see if they agree

**Way A — compare inside a single study.** Some studies *did* recruit both patients and healthy controls, and processed both groups in the same lab, on the same machine, with the same kit. Comparing sick to healthy *within* one of those studies is a fair comparison — the lab differences cancel out because both groups went through the same lab.

The catch: this leaves us with far fewer samples, so we have less power to detect small effects.

**Way B — compare across all studies, with statistical correction.** Use everything, but apply methods designed to account for study-to-study differences. More samples, more power.

The catch: the correction is an estimate. It might remove some real biology along with the lab noise, or fail to remove all the lab noise.

**Then compare A and B.** Bacteria that show up in both are our confident findings. Bacteria that show up only in B are suspect — likely lab artifacts. Bacteria that show up only in A may be real but under-powered.

**This comparison is the intellectual core of the project.** The disagreement between the two approaches is not a problem to be hidden — it is one of the most interesting results we can report, and it's a question a lot of published microbiome research handles badly.

---

## 4. What the data honestly supports

We started with 168,464 samples. We will not be analysing 168,464 samples. Here is the honest accounting, because building the project on the wrong number would guarantee failure later.

### The labels are the bottleneck

To compare sick people to healthy people, we need to know, *for each individual sample*, whether that person was sick. We checked all 3.5 million metadata entries for this.

**31,280 samples (18.6%), spread across 85 studies, have a usable health status recorded.**

Finding them is harder than it sounds, and this is a warning for whoever implements Phase 1. There is no standard field name. Some studies use an obvious one like `host_disease` or `diagnosis`. But many name the field **after the disease itself** — a study of autism has a column called `asd` containing yes/no, a Parkinson's study has one called `parkinson`, an HIV study has `hiv` with positive/negative. A search for sensible words like "disease" or "diagnosis" walks straight past all of these. The detection has to also look for each study's own condition name, taken from `projects.csv`, and then be checked by a human. We know this because our first pass made exactly that mistake and undercounted by about 7,000 samples.

The remaining 81% fall into two groups:

- Studies that recorded no health information at all
- Studies where the health status exists **only in the published paper's supplementary tables**, not in any data file

The second group is genuinely frustrating. Study PRJNA237362 has 1,379 samples and its description says "CD and healthy", so we know it contains both Crohn's patients and healthy controls. But the per-sample records contain only the body site sampled (terminal ileum biopsy, rectum biopsy, stool) and coded IDs like `skbti-0185`. No field anywhere says who had Crohn's. Similarly PRJEB5482 (1,961 samples, "malnutrition + healthy") identifies its samples as `bgtw1.f.m10` — an internal code explained only in the paper.

In total, 42 studies describe themselves as containing both patients and controls, covering 16,892 samples — but **only 5,996 of those (12 studies) actually record which sample is which.** For the remaining 10,896 across 30 studies, the answer is in a PDF.

### Could the missing labels be in one of the other compendium files?

No — and it is worth knowing why, because it saves someone a wasted week.

`Data/tags.tsv` **is** the compendium's complete per-sample metadata layer. The dataset's own [README.md](README.md) describes it as the key–value pairs associated with individual samples, *retrieved from BioSample* — meaning it is a full dump of whatever each research group uploaded to the public NCBI archive. Nothing was summarised or trimmed on the way in; that is precisely why it contains 2,608 different field names and obvious errors like ages sitting in the sex field.

The other files the README lists cannot help, because none of them has a place to put this information: the ASV tables and the Greengenes2 tables are counts for the same samples at finer resolution, `obs_md.txt` maps sequence IDs to DNA sequences, and the SILVA file is a reference library of known bacteria. The one partial exception is `gg2-2022.10-cref99.biom.qza` — the BIOM file format does have an optional slot for sample metadata, so it is worth opening if we obtain it. But it is a re-classification of these same samples, so anything inside it is almost certainly a copy of `sample_metadata.tsv`.

So if a disease status is not in `tags.tsv`, the original researchers never deposited it, and no other file in the compendium can supply what was never uploaded. `projects.csv` does give a DOI link to every paper, so recovering those labels by hand is *possible* — but it is manual work, one study at a time. We treat it as an optional extension, not part of the core plan.

### The working numbers

| Group | Samples | Studies | Used for |
|---|---|---|---|
| Everything | 168,464 | 482 | Goal 2 (lab effects) — needs no health labels |
| Has a health label | ~31,280 | 85 | Goal 1, cross-study comparison (Way B) |
| Sick + healthy in the same study, both labelled | ~5,996 | 12 | Goal 1, fair within-study comparison (Way A) |
| Healthy-only studies | ~54,571 | 125 | Reference picture of a "typical" gut |

These are pre-filtering figures. Quality filtering (Section 6) will reduce them somewhat, and we will publish the exact final numbers as a flow table so nothing is hidden. The labelled figure may also rise slightly during Phase 1, since per-project curation will catch labels that automated detection still misses.

**31,000 samples is a large, comfortable dataset.** This section is not bad news — it is the difference between a project that works and one that collapses in week ten.

---

## 5. Data quality problems we must fix first

While profiling the files we found a substantial number of errors and inconsistencies. These need fixing before any analysis, and several are interesting enough to report as findings in their own right.

**The same thing is recorded under different names.** Age appears as both `age` and `host_age`. Sex appears as both `sex` and `host_sex`. Body mass index appears as both `body_mass_index` and `host_body_mass_index`. Location appears as `lat_lon`, as separate `latitude`/`longitude`, and as `geographic location (latitude)`. If you only read one of each pair you silently lose half your data. In total there are **2,608 distinct field names** across the studies, many of them describing the same handful of concepts.

**The sex field contains ages.** The value `"47"` appears 2,794 times in the sex field, and `"48"` appears 1,501 times. Some study's columns were misaligned before upload. There is also `"neuter"` and `"not providednot provided"` — the latter clearly a string-concatenation bug at the source.

**The age-units field contains numbers.** `age_unit = 78` appears 1,663 times. Same class of error.

**Ages are written every possible way.** Of 42,796 age entries, 32,926 are plain numbers and 9,870 are not — including `"6 months"`, `"3.5 years"`, `"0-100 days"`, `"17-29 yo"`, `"85-89"` and `">=100"`. Worse, a plain number is *ambiguous*: in an adult study `6` means six years, in an infant study it very likely means six months. Getting this wrong would corrupt any age-related analysis, so our parser has to reason about which study a value came from.

**There are eight different ways of writing "no data".** `missing`, `not provided`, `not applicable`, `na`, `not collected`, `none`, `unknown`, and `restricted access` all mean the same thing and all must be caught. Miss one and it becomes a spurious category in your results.

**Not everything is a human stool sample.** The `host` field contains 221 mouse samples, 122 rhesus macaque samples, 96 simulated/artificial gut samples, and 86 laboratory control samples. Three entire studies are labelled as `mice`. There are also samples of saliva, rectal swabs and intestinal biopsies mixed in. All of these must be removed — a laboratory blank is not a person.

Even the word "human" is inconsistent: `homo sapiens`, `homo_sapiens`, `homosapiens`, `homo`, `human beings`, and `human male adult` all appear as separate values.

**Free text where categories belong.** The DNA extraction kit is recorded as free text: 283 distinct strings for 358 studies, including at least seven different spellings of "QIAamp DNA Stool Mini Kit". The disease field has 204 distinct values, with `CRC` and `colorectal cancer` stored as unrelated strings. Both need mapping onto a controlled list before they can be used as variables.

**Some readings are effectively empty.** Sequencing depth — how many bacterial DNA fragments were read from a sample — has a median of about 35,000, but **the minimum is 1**. A sample with a handful of reads tells us nothing and will produce wild, meaningless percentages. These must be filtered out.

**Publication date is not sampling date.** The metadata's `pubdate` runs 2012–2021, but the actual `collection_date` recorded by researchers runs 1998–2021 and peaks around 2015–2017. If we want to study change over time we must use collection date — though 45,402 of those entries are written in formats that don't parse, so that needs work too.

---

## 6. How we'll do it

Seven phases. The technical detail for each is in [IMPLEMENTATION.md](IMPLEMENTATION.md); this is the overview.

**Phase 0 — Get the data into a workable form.** The measurement file is 1.6 GB of text; loading it naively would need roughly 6–10 GB of memory and take a long time every single time. We read it once in chunks, convert it to a compact binary format (Parquet), and drop the columns that are zero everywhere. After this, loading takes seconds instead of minutes and fits comfortably on a laptop.

**Phase 1 — Clean and unify the metadata.** Fix everything in Section 5. Merge the duplicate field names, parse the ages, standardise the "no data" values, map the disease and kit descriptions onto controlled lists. The output is one tidy table with one row per sample and clean, trustworthy columns. We also produce a report showing how many samples we have data for on each variable — that report is itself a useful deliverable.

**Phase 2 — Quality filtering and cohort building.** Remove non-human samples, non-stool samples, and samples with too few reads. Every filter is logged with a before-and-after count, so the final table shows exactly how we got from 168,464 to our working set and anyone can audit it. Then we split out the three groups from Section 4.

**Phase 3 — Put the counts on a comparable scale.** Raw counts can't be compared directly, because one sample might have 10,000 reads and another 200,000 — the second isn't "more bacterial", it was just sequenced more deeply. We convert to proportions and apply a transformation suited to this type of data. We deliberately carry forward more than one method so that "did our choice of method drive the result?" becomes a question we answer rather than assume.

**Phase 4 — Goal 2: measure the lab effect.** Compute how similar every sample is to every other, then ask which factors explain that similarity: the study, the DNA kit, the machine, the genome region targeted, the sequencing depth, the country. We expect *study* to be the single largest factor — bigger than disease. As a deliberate check we also train a model to predict which study a sample came from using only its bacteria. If that model is highly accurate, it proves the lab signature is strong and easily learned — which is exactly the thing that would fool a naive disease model.

**Phase 5 — Goal 1: find the disease-associated bacteria.** Run the comparison inside individual studies, then combine those individual results properly (a meta-analysis, the same technique used to combine drug trials), then separately run the naive pooled version with correction applied. Compare all three. Finally, test a disease-prediction model two ways: once where the test data comes from studies the model has never seen, and once where test samples are drawn randomly from studies it has seen. The gap between those two accuracies is a direct measurement of how much a model can cheat by recognising the lab instead of the disease.

**Phase 6 — Validate and write up.** Report effect sizes and corrected significance throughout, never bare p-values. Re-run key analyses with the labels randomly shuffled — anything still "significant" reveals a flaw in our method. Re-run with different filtering thresholds to confirm conclusions aren't an artifact of one arbitrary cutoff. Check our top findings against the published literature using the paper links in `projects.csv`.

---

## 7. What we'll produce

- **A cleaned, documented sample table** — the harmonised metadata, which is genuinely reusable beyond this project
- **A data quality report** — coverage per variable, the error catalogue from Section 5, the filtering flow table
- **A variance-explained table** — how much of the variation each technical and biological factor accounts for (Goal 2's main answer)
- **A ranked list of disease-associated bacteria**, with effect sizes, corrected significance, and a clear marker for which findings replicated across both analysis designs (Goal 1's main answer)
- **A concordance figure** comparing within-study against cross-study results — the centrepiece
- **Numbered notebooks**, one per phase, that run start to finish
- **A final written report**

---

## 8. Limitations we will state up front

Being explicit about these strengthens the work rather than weakening it.

- **The sample is not the world.** 61% of samples come from Europe and North America, and 36% from the United States alone. Oceania contributes 4 samples. 17% have no usable country recorded. Any claim about "the human gut microbiome" is really a claim about a mostly Western one.
- **Genus-level resolution only.** We can identify broad bacterial groups, not species or strains. Effects that operate at strain level are invisible to us.
- **Observational, so no causal claims.** If a bacterium is elevated in a disease, we cannot say it caused the disease. It may be a consequence of the disease, of its treatment, or of an associated change in diet.
- **Most samples have no health label.** 81% are unlabelled; our disease work rests on 19% of the compendium.
- **Health labels are self-defined by each study.** One study's "healthy" may be another's "mild symptoms". We are harmonising the words, not the underlying clinical assessments.
- **Diet, smoking and antibiotic history come from essentially one cohort** (each covering ~4,842 samples, all the same study). Those variables describe that cohort, not the compendium, and cannot support general claims.

---

## 9. Optional extensions

These are **not** part of the committed plan. They are listed so the team can pick up an additional direction if time allows, roughly easiest first.

| Extension | What it would add | Feasibility |
|---|---|---|
| **What a typical healthy gut looks like** | Use the 125 healthy-only studies (54,571 samples) to describe the normal range — which bacteria are near-universal, how much healthy people vary. Gives disease findings a baseline to be measured against. | **Easy.** No labels needed beyond "healthy", data already sufficient. |
| **Do people fall into distinct "gut types"?** | Test the long-debated enterotype hypothesis — whether guts cluster into a few discrete types or vary continuously — at a scale most published attempts never had. | **Easy to run, hard to interpret.** Clustering will always return clusters; proving they're real needs care. |
| **Geography and industrialisation** | Compare composition across world regions and against an industrialisation gradient. Well-supported by `region` and `iso`. | **Moderate.** The 17% unknown-country rate and severe Western skew both need handling. |
| **Change over time, 1998–2021** | Has the measured gut microbiome shifted across two decades? | **Moderate.** Needs the collection-date parser, and separating real change from changing lab methods is genuinely hard — it depends on Goal 2's results. |
| **Metadata harmonisation as its own contribution** | Publish the cleaning layer and error catalogue from Section 5 as a reusable resource. Phase 1 produces most of this anyway. | **Cheap add-on.** Largely a by-product of work we're already doing. |
| **Recovering labels from the papers** | Manually extract per-sample disease status from supplementary tables for high-value studies, unlocking up to 10,896 additional samples across 30 studies. | **Expensive.** Manual, one study at a time, with no guarantee the tables map cleanly onto sample IDs. Only worth it for a few large studies — PRJEB5482 (1,961 samples) and PRJNA237362 (1,379) are the best candidates. |

---

## Where to go next

- **To start building:** [IMPLEMENTATION.md](IMPLEMENTATION.md) — phase-by-phase technical specification
- **For the dataset's own documentation:** [README.md](README.md) — upstream description of how the compendium was produced
