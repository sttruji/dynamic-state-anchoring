# Checkpoint label audit — 19 October 2026

The primary assistant personally read and independently solved 76 training
questions, inspecting their marked rationales and all 155 accepted checkpoints.
This includes every question in the local 32-question pilot, 32 additional
risk-stratified historical questions, and 12 supplemental cases completing coverage
of all equal-final-value and accepted-after-rejected-arithmetic flags. These are
assistant judgments, not human adjudications or an independent second review.
The purposive sample does not estimate a corpus-wide semantic error rate.

The mechanical audit covers the actual 512 historical training records, including
all 32 local records. No new model training or validation answer evaluation
occurred, and no held-out test examples were inspected. Original source data,
prepared labels and model code are unchanged.

## What is mechanically correct

- All 512 records reconstruct against the pinned original source rows. Every
  model-input and supervision field returned by tokenization equals the saved one.
- All 821 retained equations pass a separate AST/Fraction arithmetic checker.
  This checker does not call the production arithmetic evaluator or execute text.
- All marker counts, teacher indices, and reconstructed loop schedules agree.
  No event required collision adjustment from its proportional ideal loop.
- Each teacher target is the causal hidden state at the `>>` token immediately
  before `<SA>`. All 821 target tokens are `>>`. This is the intended boundary,
  not an off-by-one error: the state has seen the question and equation result.
  It is not a direct numeric-value target or an explicit value-and-role label.
- 426/512 records receive auxiliary supervision. The other 86 have no accepted
  checkpoint and retain answer/CoT losses, with auxiliary losses masked. They are
  not trained as eight explicit negative gate labels.
- Among the 3,408 supervised loop decisions there are 821 positives (24.09%)
  and 2,587 negatives. The local 32 have 29 supervised records and 62 positives
  among 232 supervised decisions.

These checks rule out specific serialization/indexing explanations for the earlier
results. Reconstruction parity does not validate the semantic target or prove
that a token-relative loop is the right time to write an intermediate conclusion.

## Personal judgments

| Verdict on the question, trace and answer | Reviewed records |
| --- | ---: |
| Supported under the recorded interpretation | 56 |
| Supported with currency rounding or a justified whole-item/day conversion | 7 |
| Inconsistent trace, answer or requested output | 7 |
| Contradictory premises | 2 |
| Ambiguous wording | 3 |
| Underdetermined placement/boundary convention | 1 |

In the local 32 alone, 29 are supported, one is ambiguous and two are inconsistent.
The supported group still contains one uninformative given-restatement checkpoint.
The complete written solutions, assumptions, roles and verdicts are recorded in
`provenance/checkpoint_label_personal_review.jsonl`, keyed by 1-based source row,
sample ID, question hash and original-row hash.

### Concrete findings in the local pilot

| Source row | Finding | Consequence |
| --- | --- | --- |
| 288,908 | Bike distance: 5 km daily plus 4 extra on each weekend day means `7*5+2*4=43`. The trace/gold give 33. | Every retained equality is locally true, but the overall trace omits 10 weekend-base kilometers. Wrong gold still trains both main losses. |
| 305,659 | Candy trace ends at 21.25, while gold is 21 without an authorized rounding rule. | All three equations remain checkpoints, including the final quantity at loop 8, because 21.25 differs from gold. |
| 207,002 | Dinner is “twice as much,” with breakfast or lunch as possible antecedents. | Source gives 315 eggs monthly; the other reading gives 225. Arithmetic labels cannot resolve the referent. |
| 112,564 | `120=120` repeats a supplied shoelace length. | A true equality is labeled as a completed intermediate even though no new conclusion was derived. |

### Other important cases

- Row 145,668: the correct five-year salary is 537,600; the source claims
  480,000. The parser rejects the wrong terminal equation as a checkpoint, but
  answer and explicit-CoT losses still train its wrong content.
- Row 287,176: `160-107=53` is locally exact, but 107 comes from an unlicensed
  rounding of two thirds of 160. Local checkpoint verification does not establish
  that its input quantities are justified by the question.
- Row 350,567: a tax equality uses `0.34` for an exact `0.336`. Ordinary cent
  rounding makes the 6.06-dollar final answer reasonable. The 24 rejected
  arithmetic candidates across the 512 are not 24 wrong solutions.
- Row 315,768: `5*30=150` answers the question and is marked; the following
  `150=150` is the candidate removed by the terminal rule. Removing only the
  last matching equality can leave direct final-answer computation supervised
  as an intermediate checkpoint.
- Row 14,677 requires subtracting 40 from 400 and dividing by two, but the supplied trace
  includes only `360/2=180`. The terminal-only trace gives no checkpoint despite
  a multistep word problem. No-event count is not equivalent to one-step count.

## Equal numbers are not necessarily equal conclusions

Sixteen accepted events in 15 historical questions have the same numeric value
as the final answer. I reviewed all 15 questions; blanket removal would discard
legitimate intermediates. For example, row 183,715 has `24/4=6` cupcakes per cup
followed by `36/6=6` cups required. The values match while their units and roles
differ. Twelve reviewed records include a legitimate equal-valued intermediate;
this includes a record whose supplied final output is nevertheless incomplete.
Two records have an answer-equivalent computation followed by an identity or
multiply-by-one step. The remaining flagged record has a separate promotion-time
horizon error. One of these records contributes two equal-valued events.

Do not repair this by excluding every event whose value equals gold. A separate
experimental policy may remove identity tails and given restatements, but it must
preserve legitimate equal-valued roles and leave the primary source untouched.

## Timing labels have a strong position prior

The historical positive-event histogram by loop 1 through 8 is
`[2, 107, 140, 254, 90, 170, 45, 13]`. These timings derive from token position,
not observed latent progress.

I fitted eight smoothed loop-position probabilities on the other 375 supervised
training records, excluding the fixed 64-question historical diagnostic subset.
The 51 supervised diagnostic questions provide the same 108 positive and 300
negative decisions used by the old controller AUC report. Their saved sample IDs
were checked against the seeded selection.

| Timing model on those same decisions | AUC |
| --- | ---: |
| Previously trained DSA controller | 0.51349 |
| Loop-position frequencies, no question or latent features | 0.78176 |

At threshold 0.5 the position baseline writes only on loop 4: precision 0.6863,
recall 0.3241, and a 12.5% write rate. This is an offline label diagnostic, not an
answer-accuracy result or a newly trained DSA policy. The previous controller
was trained on all 512 records, while the position baseline excludes the scored
subset from fitting; training exposures are not identical. Both use the same
scored decisions. Even a gate that learns these proxy labels may learn a clock
rather than when a useful semantic conclusion is ready.

## Disposition and reproducibility

Keep the original historical experiment and labels intact. The proposed next
mechanism-development fixture retains the 29 supported local questions, replaces
rows 207,002 / 288,908 / 305,659 with three independently reviewed supported rows
99,935 / 186,418 / 219,649, and proposes removal of the `120=120` checkpoint in a
new derivative. Hash-ordering, rather than model performance, chose replacements.
No derivative training dataset has been emitted or used. This is a curated small
fixture, not a correction to the official GSM8K benchmark or a label-only causal
ablation against the old membership.

The next experiments are specified in `MPS_CONTROLLED_EXPERIMENTS_20261019.md`.

Artifacts:

- `provenance/checkpoint_label_audit_20261019.json`: mechanical report, verdict
  aggregates, hashes, prior baseline and frozen proposed fixture membership.
- `provenance/checkpoint_label_personal_review.jsonl`: all 76 individual judgments.
- `artifacts/checkpoint-label-audit-20261019-final/`: local source-linked review
  packets and all 512 reconstructed training rows, including dataset text.

Reproduce the mechanical checks and validate the written ledger without training:

```bash
.venv/bin/python scripts/audit_checkpoint_labels.py \
  --output artifacts/checkpoint-label-audit-rerun \
  --review-ledger provenance/checkpoint_label_personal_review.jsonl \
  --summary-output /tmp/checkpoint-label-audit-rerun.json

.venv/bin/python -m unittest tests.test_checkpoint_label_audit -v
```

The script packages the prewritten semantic judgments; it does not generate them.
