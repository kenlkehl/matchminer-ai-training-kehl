# Prompt changes

Only the current version (v11, with its layouts file) is kept in the tree; earlier
versions and the dropped editor prompt were removed on 2026-10-28 and remain in
git history. Every run directory under `data/no_phi/synthetic_protocols_from_registry/`
holds a copy of the template it used.

## from_registry.v1

Registry title, brief summary, detailed description and eligibility criteria;
protocol structure, eligibility detail, consistency, no synthetic disclaimer, no
citations. First run: NCT04816214 (GEOMETRY-E), Camus FP8, 2026-10-28.

Findings on NCT04816214 (real protocol available for comparison):
- No disclaimer leaked.
- Written retrospectively: past tense, "Study Status: Enrollment halted", actual
  end-of-trial date and "randomized part not initiated", copied from the
  registry's post-hoc detailed description.
- Dropped the registry exclusion for known druggable molecular alterations.
- Missed criteria implied by the regimen: no non-squamous restriction despite a
  pemetrexed control (real protocol excludes squamous histology).
- Eligibility ~7,100 characters vs ~1,700 in the registry and ~17,600 in the
  real protocol; whole document ~38,000 characters vs ~800,000 real.

## from_registry.v2

- Timing: write as approved before enrollment, present/future tense; omit
  post-hoc registry events and status.
- Eligibility: target 2,000-4,000 words with lettered sub-criteria, assay
  methods and positivity thresholds; map every registry criterion explicitly.
- Add criteria implied by the study treatments and comparator labels.
- Whole-document length target: more than 15,000 words.

First run: NCT04816214, sn4622130540:8000 NVFP4, 2026-10-28 (98.7 s, 21,939
completion tokens including 9,045 reasoning).

Findings:
- Timing fixed: prospective voice, no halt/termination/status content; the run-in
  and randomized parts are both described as planned.
- Registry druggable-alteration exclusion restored (includes KRAS, as in the
  real protocol); CNS, HBV/HCV/HIV and prior-therapy rules now have real-protocol
  nuance. QTcF >470 ms exclusion and MET copy number >=5 match the real protocol.
- Eligibility ~16,400 characters / 2,366 words (real ~17,600 characters).
  Whole document ~58,000 characters / 8,600 words: below the 15,000-word target.
- Still missing the non-squamous restriction implied by pemetrexed, and
  contraception is 4 weeks for every drug despite platinum/pemetrexed labels.
- Invented a capmatinib code ("CBIC450") alongside the real INC280.
- Renal criterion contradicts itself (CrCl >=50 required, then 30-50 allowed for
  carboplatin).
- Sample size: 300 PFS events stated; Schoenfeld gives ~262 for HR 0.67,
  two-sided alpha 0.05, 90% power (conservative, not inconsistent).

## from_registry.v3

- Cohorts: each part/cohort/arm with a distinct population gets its own
  subsection with cohort-specific criteria at full detail; common criteria kept
  separate with stated applicability; no cohorts beyond the record.
- Study-treatment criteria: check each agent's label (histology, renal/hepatic
  limits, interactions, contraception) and apply the strictest.
- Names: only drug names/codes from the record.
- Self-check for contradictions within criteria and across sections.

First run: NCT04259450 (AFM24-101; Part 1 escalation + three Part 2a tumor-type
cohorts), sn4622130540:8000, 2026-10-28 (71.8 s, 15,928 completion tokens
including 3,963 reasoning). Real protocol V10 (26 July 2023) for comparison.

Findings:
- Cohorts: four named subsections (Part 1, Arms A-C) matching the registry's
  four populations and the pipeline's four extracted cohorts; no invented cohorts.
  Each has a population statement and cohort-specific inclusion/exclusion
  (histology, extent, biomarkers with assay/lab, measurability, line counts,
  required prior agents). Cohort-specific text 1,900-2,600 characters per cohort
  vs one ~150-250 character registry bullet and 300-840 characters in the real
  protocol, which keeps cohort rules inside one numbered list.
- Added detail is invented and diverges from the real protocol: EGFR IHC >=10%
  / H-score >=100 central (real >=1%, local acceptable); CRC KRAS-only (real RAS
  wild type, irinotecan required); no NSCLC T790M -> osimertinib rule.
- Follows the registry where the registry differs from the real V10 protocol
  (ccRCC >=2 prior lines vs >=1 after amendment).
- Exclusion lists contain permissive notes ("measurable disease not required",
  "X is not an exclusion criterion"); NSCLC driver-alteration exclusion is
  muddled; prior AFM24 exclusion appears only under Arm A.
- No invented drug codes; contraception 6 months after AFM24.

## from_registry.v4

Generic rules only; no trial-, drug- or biomarker-specific wording.
- Criterion form: inclusions state requirements, exclusions state disqualifying
  conditions; permissions/exceptions attach to the criterion they qualify, never
  stand-alone; one unambiguous reading per criterion.
- Criterion placement: criteria that apply to every cohort go in the common
  section; cohort subsections hold only what differs; each criterion once.

Runs: NCT05008055, NCT05082025, NCT04902885, NCT06331299 concurrently on
sn4622130540:8000, 2026-10-28 (106-123 s each, all finish_reason stop). Real
protocols downloaded for all four; eligibility lengths (characters, registry /
real / synthetic): 5.5k / 16.2k / 11.3k; 7.6k / 9.2k / 17.6k; 3.3k / 4.2k /
18.9k; 4.0k / ~5k / ~26k.

Findings:
- Cohorts explicit in every multi-cohort trial: named subsections with
  population and cohort-specific criteria; registry "cohort 1-3" became named
  ovarian/endometrial/breast cohorts (names taken from the registry text).
- Over-splitting: NCT04902885's randomized arms (drug vs placebo, same
  population) and an optional dose level became separate cohorts, with
  "randomized to arm X" used as an inclusion/exclusion criterion (7 cohorts vs
  the registry's 4).
- Criterion form/placement rules only partly followed: stand-alone non-criteria
  in exclusion lists ("not applicable", "is not an exclusion for Cohort 3",
  "none beyond ..."), common criteria repeated in every cohort, and a common
  exclusion that one cohort then waives. NCT05008055 was clean apart from a
  repeated line-count criterion.
- Leak: NCT06331299 mentions "the registry record" and adds an appendix mapping
  registry criteria to protocol criteria (the v2 mapping instruction surfaced
  in the output).
- Length: where the registry is already close to the real protocol
  (investigator-initiated and smaller trials), the fixed 2,000-4,000-word
  eligibility target pads far beyond the real protocol (up to ~5x).

## from_registry.v5 (+ from_registry.v5.layouts.json, editor.v1)

Goal clarified by the user (2026-10-28): build a corpus of realistic, often
messily specified eligibility criteria with one or more protocol-specified
cohorts/parts/arms; consistency with the seed registry text is not essential.
- Registry is a seed, not a specification; comparison happens in reasoning only.
- Populations defined by differing eligibility; randomized arms, dose levels and
  treatment assignment are never populations or eligibility criteria.
- Eligibility layout varies per trial (four real-world layouts in
  from_registry.v5.layouts.json; stable NCT-ID hash, recorded per run).
- Realistic texture (notes, inline exceptions, cross-references, sponsor habits)
  explicitly wanted; v4's strict form/placement rules removed.
- Eligibility length sized to trial type and sponsor instead of a fixed target.
- The document must never mention the registry/seed or its derivation.
- editor.v1: second call returning JSON find/replace or delete-section edits
  for five artifact types only (provenance mentions, assignment-based criteria,
  no-requirement entries, undecidable contradictions, clinically meaningless
  criteria); edits apply only where the target occurs exactly once.

Runs: NCT05008055 (layout 1), NCT05082025 (2), NCT04902885 (3), NCT06331299 (3,
single population), NCT04259450 (0), concurrently on sn4622130540:8000,
2026-10-28; generation 111-149 s, editor 48-67 s.

Findings:
- Every draft followed its assigned layout; no registry/derivation mentions in
  any draft; no assignment-based criteria; trilaciclib's randomized arms were no
  longer split into populations (core criteria plus Part 1 / Part 2 modules).
- Layout 0 (AFM24) produced "Part 1 only:" / "Part 2a Cohort A only:" inline
  criteria, close to the real AFM24 protocol's own format. Cohort letters were
  reassigned relative to the registry but used consistently.
- Eligibility length now tracks trial type (characters, registry / real / v4 /
  v5): capivasertib 5.5k / 16.2k / 11.3k / 12.8k; trilaciclib 3.3k / 4.2k /
  18.9k / 7.4k; UGN-103 4.0k / ~5k / ~26k / 10.1k; AFM24 2.3k / 15.3k / - /
  10.2k; copanlisib 7.6k / 9.2k / 17.6k / 28.3k (layout 2 repeats shared
  criteria in each of three self-contained lists).
- editor.v1: 3 edits across 4 successful runs. One correct (a "no additional
  exclusion criteria" note); two over-applied defect 2 by deleting trilaciclib's
  "willing to be randomized/receive placebo" criteria, which real protocols use.
  The copanlisib editor call hit the 32,768-token limit while reasoning and
  returned no JSON, so its draft was kept unedited.
- Remaining texture is realistic rather than artifactual, e.g. a copanlisib
  Cohort 3 exclusion that mentions Cohorts 1 and 2 and waives prior fulvestrant
  with a condition.

Editor dropped (user decision, 2026-10-28); generation is v5 alone.

Runs without editor: 8 new seeds sampled from the 5,000-trial training sample
(4 multi-cohort, 4 single-cohort, random_state 20261028; none has a posted
protocol): NCT07346157, NCT04663763, NCT05642780, NCT06257758, NCT04587739,
NCT06678269, NCT05193721, NCT04601831; concurrently on sn4622130540:8000,
107-134 s each, all finish_reason stop, layouts 0-3 all used.
- No provenance, status or assignment-criterion artifacts (two flagged hits were
  "synthetic lethality" and an inline clarification within a real criterion).
- Populations matched the pipeline's registry cohorts for the multi-cohort
  seeds (5 named basket cohorts; MSS/MSI-high; cohorts 1/2), with one added
  split (Phase 2 ovarian vs other solid tumors, supported by the seed text).
- Two of four single-cohort seeds gained dose-escalation/expansion parts that
  the seed does not mention; acceptable under the seed-not-specification goal
  but it shifts the cohort-count distribution.
- Eligibility 6.9k-12.3k characters, including 6.9k from a 574-character seed.

Full batch launched 2026-10-28 20:56 UTC: 4,981 unique NCT IDs from
sample_trial_cohort_lineitems.csv, v5, run_batch.py, servers
sn4622130540:8000 (32 concurrent), 172.24.216.113:8001-8003 and
camus:8002-8003 (16 each); output
data/no_phi/synthetic_protocols_from_registry/batch_v5_sample5000_20261028/.

Batch stopped at user request 2026-10-28 ~22:55 UTC with 3,861 of 4,981
protocols complete (camus:8003 removed at 22:04 via disabled_servers.txt).
User review: most sections are acceptable, but the eligibility division mirrors
what the downstream extraction prompt asks for (named study populations,
cohort-specific vs common criteria) too closely to be realistic. Next prompt
versions target realistic protocol eligibility structure for Nemotron.

## from_registry.v6 (+ from_registry.v6.layouts.json; weighted layouts)

Goal (user, 2026-10-28): eligibility should read like real protocols, not like
the extraction schema. A survey of 59 real multi-cohort protocols from the
sample (pdftotext, heuristic) found mostly one inclusion and one exclusion list,
cohort differences expressed inside criteria (disease-keyed sub-items,
part-keyed thresholds, NOTE exceptions), inline part/cohort qualifiers in about
a third, and only a few separate lists (core + modules, "disease-specific
inclusion criteria", tiers). Cohorts are defined in the design section.
- Removed population/common/cohort-specific vocabulary; such headings banned.
- Study design defines parts/cohorts/arms with a schema.
- Layouts weighted 3:2:1:1 (differences inside criteria; inline qualifiers;
  disease-specific subsection; core + module sections).
- Texture: sponsor intro sentence, numbering/nesting, NOTE lines, references to
  sections/appendices that must exist; synopsis lists key criteria only.
- choose_layout accepts {"weight", "text"} entries (v5 hashing unchanged).

Runs (9 seeds, sn4622130540:8000, 106-167 s, all stop): extraction-vocabulary
headings 0/9 (v5: 6/9); intro sentence 9/9 (v5: 5/9); NOTE lines in 5/9 (v5:
0/9); no dangling section/appendix references. Problems: the treatment-
assignment rule was echoed as protocol text in 5/9 ("cohort assignment is a
treatment-group designation, not a randomisation outcome"); layout 1 tagged
every criterion "[All participants]".

## from_registry.v7

- Assignment rule reworded ("assigned after enrollment, so eligibility criteria
  do not depend on them"); new closing instruction not to restate or explain
  any requirement in the document.
- Layout 1: only criteria limited to some parts/cohorts/arms carry a qualifier.

Runs: same 9 seeds plus NCT04599634, NCT04659629, NCT04455620 (12 total,
sn4622130540:8000, all stop; two reruns after ClinicalTrials.gov 429s, now
retried in registry_record). Extraction-vocabulary headings 0/12; blanket
"all participants" labels 0/12 (v6: 29 in one protocol); intro sentence 12/12;
NOTE lines 8/12; no dangling references. Remaining: the study-design sentence
was still echoed in 5/12 ("dose levels are assigned after enrollment;
eligibility criteria do not depend on treatment assignment").

## from_registry.v8

- Dropped the assignment sentence from the study-design requirement and the
  example from the no-restate instruction; checked for assignment-based
  criteria returning.

Runs: same 12 seeds (sn4622130540:8000, all stop). Echo 0/12 (one false
positive: "participants are not re-assigned after enrollment"); assignment-based
eligibility criteria 0/12 without the rule; extraction-vocabulary headings
0/12; intro 12/12. Remaining: layout 1 tagged 33 criteria "(All parts)" in
NCT04659629 (v7: none for that seed); one protocol referenced a nonexistent
Section 4.5 (left as realistic messiness).

## from_registry.v9

- Prompt text identical to v8; layout 1 now says most criteria carry no label
  and must never be tagged "(All parts)" or "[All participants]".

Runs: the 5 layout-1 seeds plus 6 unseen multi-cohort seeds (two each for
layouts 0, 2, 3; sn4622130540:8000, all stop). Blanket labels 0/11 apart from
one layout-2 heading "Other Inclusion Criteria (All Parts)"; echo 0/11;
assignment-based criteria 0/11; extraction-vocabulary headings 0/11; intro
sentence in every protocol whose eligibility section the checker located (10/11;
the eleventh is headed "Section 6. Study Population"). Synopses give abbreviated
key criteria and point to the full section; layout 3 produced core criteria
plus "Additional inclusion/exclusion criteria for Part A/B/C" in a later
section. v9 is the current candidate for a new batch.

User review of a v9 protocol (NCT07134088, layout 3): eligibility rules leaked
into the background/rationale (excluded/permitted groups with justifications),
and the study population intro described how core and part-specific lists
combine ("...none of the core exclusion criteria apply. Part-specific
additional criteria are provided in Section 7 and must be reviewed in
conjunction with this section"). Core-plus-module and disease-specific
subsection layouts are real but rare (about 2 of 59 surveyed multi-cohort
protocols each), so they were over-weighted at 1/7.

## from_registry.v10 (+ layouts reweighted 20:9:1:1)

- Outside the synopsis and study population section, who enrolls is described
  only at study-design level; no listing, previewing or justifying of
  individual eligibility rules elsewhere.
- List openings: one short sponsor-style sentence, without explaining how
  lists in different sections combine; module sections open with "In addition
  to the inclusion criteria in Section 5.1, patients in Part B must meet...".
- Layout weights approximate the survey: ~65% differences inside criteria,
  ~29% inline qualifiers, ~3% disease-specific subsection, ~3% core + modules.

Runs: 11 seeds (sn4622130540:8000, all stop). Layouts now 0/1 apart from one
layout 2; vocabulary headings, echoes, assignment criteria and blanket labels
0/11; NCT07134088's population intro is a plain sponsor sentence with part/arm
differences as sub-items of one disease-status criterion. Background leakage
fell but persisted where the seed's own description states eligibility
(NCT07134088: prior CD19 therapy permitted, Ph+ TKI-intolerant eligible).

## from_registry.v11

- Seed bullet: eligibility details in the record's title, summary or
  description go only in the study population section and synopsis.

Runs: same 11 seeds (sn4622130540:8000, all stop). NCT07134088's background
no longer restates eligibility (its leaked rules had come from the seed's
eligibility criteria; the seed has a 125-character summary and no detailed
description). Background leakage 1/11 (a sentence justifying "disease-specific
inclusion criteria" in NCT06257758); vocabulary headings, echoes, assignment
criteria and blanket labels 0/11; intro sentence in every located section.
