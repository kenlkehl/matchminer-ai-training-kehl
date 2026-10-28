# Prompt changes

Each version is kept unchanged once it has been run; every run directory under
`data/no_phi/synthetic_protocols_from_registry/` holds a copy of its template.

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
