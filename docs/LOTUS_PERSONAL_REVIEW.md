# Personal semantic review of LOTUS source issues

I reviewed every one of the 194 unusual-answer flags, every representative of the
52 exact-duplicate groups (112 source occurrences, including 60 extras), and both
records in the conflicting-answer pair. This review reads the question and trace,
solves the requested problem independently, and distinguishes answer formatting
from reasoning or premise errors. Contentious decisions were reread by the
primary reviewer. It is the assistant review requested by the user; no human
identity or signature is implied.

The review is complete for those identified issues. It does not establish the
semantic correctness of all 385,620 training records or a corpus-wide error rate.

## Findings across all 194 unusual answers

| Verdict | Rows | Interpretation |
| --- | ---: | --- |
| Correct | 123 | Valid requested answers expressed as fractions, currency, times, ratios, percentages or explicitly symbolic quantities |
| Acceptable marked approximation | 24 | Correct method and appropriate magnitude; rounding limitations are recorded |
| Incorrect | 16 | Wrong arithmetic, requested quantity, day count, or required whole-package rounding |
| Ambiguous | 17 | Multiple materially different interpretations or contradictory premises |
| Underdetermined | 12 | Missing counts, durations, rates, schedule phase or other required information |
| Malformed/non-answer | 2 | Output fails to answer the requested quantity/person or leaves an unsolved expression |

Thus 147 of the 194 formatting flags are retained as valid or explicitly
approximate under the documented interpretation. The other 47 are specifically
flagged for a separate clean sensitivity dataset. These are reviewer verdicts,
not output from an automatic numeric-format classifier.

Approximation policy is conservative: an explicit `~` can indicate a useful
approximation when no precision is demanded. Premature rounding is noted rather
than automatically treated as a substantive wrong method. Exact-fraction requests
and whole-object requirements are checked more strictly. For instance, the source
`~16.04` pounds in the cat/kitten problem uses the intended method with early
rounding; the exact result is 16. An unmarked `3.17/10` for a requested fraction
instead differs from the exact `19/60` and is flagged.

## Concrete errors found

Source row numbers below are **1-based positions in the original training array**.

| Row | Source answer | Review result | Reason |
| --- | --- | --- | --- |
| 41,385 | `1/11` | `2/11` | An unordered green/red pair includes both draw orders |
| 43,194 | `11/20` | `7/20` | `1/4 + 2/5 = 13/20`, so the remaining jar fraction is `7/20` |
| 79,733 | `~533.33` | `0` additional dollars | The question already says she saves $1,500 monthly; the trace substitutes income for savings |
| 85,132 | `Tuesday` | Following Monday | Eight play days starting Monday include Monday as day one |
| 119,935 | `1/35` | `16/35` | `1 - 2/5 - 1/7 = 16/35` |
| 129,612 | `~7.2` bundles | `40` bundles | Each bundle has three balls; `120/3 = 40` |
| 178,590 | `$150` | `$187.50` | Three years of simple interest: `2500 * 0.025 * 3` |
| 286,438 | `73/100` | `19/26` | The requested simplest exact fraction cannot be replaced with a rounded percentage |
| 341,589 | `9/20` | `7/20` | `1 - 2/5 - 1/4 = 7/20` |
| 363,644 | `10/119` | `30/119` | The stated no-replacement calculation was multiplied incorrectly |

Every reviewed unusual-answer row, including all 16 incorrect cases and all
conditional alternatives, is in `provenance/lotus_personal_review_rows.jsonl`.

## Duplicates

The 60 extra occurrences are repeats of 52 exact records entirely within the
training split; the stored questions, traces and gold answers agree within each
group. I solved all 52 representative word problems. Their supplied answers follow
the intended arithmetic interpretation. One body-part-length question has awkward
additive-model wording, which is explicitly noted rather than presented as a
rigorous physical model.

The extra occurrences are about 0.01556% of the training row count. That measures
exposure, not the effect on learning. Retain them in the primary LOTUS mirror so
the baseline and DSA see the same supplied examples. A separately named
sensitivity experiment can deduplicate by source hash, keeping the first occurrence
and using the identical policy across compared arms. There is no new evidence of
cross-partition exact-duplicate leakage.

## Conflicting bus question

Rows **44,776 and 198,445** ask the same question about wheels on seven motorcycles
and twelve buses. One source assumes four wheels per bus and answers 62; the other
assumes six and answers 86. With the ordinary two-wheel motorcycle assumption,
the answer is `14 + 12*w`, where `w` is the unspecified bus wheel count.

Both traces contain correct local arithmetic. The wording does not establish a
unique gold answer. I therefore adjudicate this as **underdetermined**, and do not
guess a replacement. Preserve both source rows in the primary mirror with this
limitation disclosed; exclude both from any explicitly cleaned sensitivity set.

## Recommended disposition

- Keep the original corpus and primary matched benchmark membership unchanged.
  Disclose the reviewed noise; all arms receive the same original exposure.
- Keep the 147 valid/marked-approximation unusual-answer rows. Avoid collapsing
  times, ratios, fractions or units into bare scalar numbers without a declared
  task-aware normalization rule. The pinned baseline scorer remains unchanged.
- For a clean sensitivity experiment, exclude the 47 adjudicated problematic
  unusual-answer rows and both ambiguous bus rows: **49 unique training rows**.
  This is a proposal manifest, not an applied corpus edit. An optional further
  deduplication would remove the 60 extra training occurrences separately.
- Proceed to a small checkpoint/implementation pilot under a documented unchanged-
  source protocol. The user-authorized source-issue review is finished; additional
  research claims still require the planned controlled and matched experiments.

There are also 89 empty training rationales. Their structural effect is already
known: they supply answer/explicit losses and no milestone targets. They are not
included in this review's 194-row semantic census unless their answer was one of
the flagged formats. An empty rationale alone is not evidence that its final
answer is wrong.

The previous 32-record milestone sample was already manually read by the
assistant and exposed semantic risks such as rectangle formulas applied to an
oval and reuse of an upstream rounded value. That evidence remains recorded;
local arithmetic verification is still weak supervision rather than a semantic
verifier.

## Evidence and reproducibility

- Summary and scope: `provenance/lotus_personal_review.json`
- All 194 source-linked adjudications: `provenance/lotus_personal_review_rows.jsonl`
- All 52 duplicate-group adjudications: `provenance/lotus_personal_review_duplicates.jsonl`
- Proposed row exclusions, not applied: `provenance/lotus_personal_review_sensitivity_proposal.json`

The summary records source and artifact hashes. The original training SHA256 is
`93a266b1f7425a00463bb775de2946f9b30f76514944b5456bbffceb50b5de30`.
No source questions, answers, annotations, train/validation/test membership,
scoring code, or model weights were changed for this review.

`assemble_lotus_personal_review.py` packages the written verdicts, verifies exact
coverage of every unusual-answer flag, derives record IDs from authoritative
source rows, rejects conflicting identities and duplicate/missing reviews, and
checks that all duplicate occurrences match their hashes. It does not generate
semantic judgments or repair data.
