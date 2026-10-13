# LOTUS corpus audit

The pinned LOTUS corpus has been downloaded and reproduced with its unmodified
converter. Exact/format-normalized split integrity and ordered official-test
membership pass. Frozen V1 annotation has useful coverage. The source-data quality
findings now have an individual semantic review and a documented unchanged-source
pilot policy; see `LOTUS_PERSONAL_REVIEW.md`.
The full tokenizer pass also completed with zero mapping or source-text roundtrip
failures. The aggregate result is in `provenance/lotus_corpus_audit.json`.

## Provenance and split integrity

- LOTUS checkout: `eb77e2f7909c5006f58ff0ad7cd6629b942caa9e`.
- Data source: Internalize_CoT_Step_by_Step
  `e06a32ee5e4cd117171daeb4755d2a97ece62761`.
- Raw train bytes match the pinned Git LFS SHA256 and byte count. Validation and
  test bytes match their actual pinned Git blobs. Original `.txt` files are kept.
- The upstream `preprocessing/gsm_icot.py` produced the JSON arrays without
  modification. Arrays, row ordering, duplicate rows, questions, steps and answers
  are preserved; rationales are reconstructed with `steps_join=space`.

| Split | Rows | Unique normalized questions | No-event rows | Accepted events | Eligible at K=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 385,620 | 385,533 | 63,665 (16.51%) | 613,547 | 321,955 (83.49%) |
| Validation | 500 | 500 | 20 (4.00%) | 1,107 | 480 (96.00%) |
| Test | 1,319 | 1,319 | 48 (3.64%) | 3,074 | Descriptive labels only |

All three split pairs have zero overlapping normalized question hashes or
fallback problem groups. This establishes exact/format duplicate separation;
source records have no original problem IDs, so paraphrased/semantic duplicates
cannot be ruled out by this audit.

LOTUS test questions and normalized gold answers match all 1,319 official GSM8K
test rows **in the same order**. All 7,473 official training questions are present
across LOTUS train/validation: 6,973 in train and 500 in validation. Augmentation
also contributes many additional question strings, so the LOTUS training count
must not be represented as 385,620 original GSM8K problems.

## Annotation findings

`conclusive` and `verified_only` are identical on every split: all retained events
are R1 calculator equalities; R2/R3 have zero retained events. The frozen V1 parser
was not expanded in response to the corpus. All source/annotation/span and JSON
roundtrip checks pass.

At loop budgets 4/8/16/32, training controller eligibility is
313,484 / 321,903 / 321,955 / 321,955 rows. Event overflow masks another
8,471 / 52 / 0 / 0 rows, respectively. Every no-event/overflow row is retained for
answer and explicit supervision. Coverage does not change test denominators.

Training diagnostics include 24,466 locally inconsistent exact equalities. That
category includes finite decimal approximations of nonterminating fractions as
well as actual arithmetic mistakes, so it is not a count of incorrect solutions.
The terminal policy withholds 362,407 final-value-matching candidates. Unsupported
functions, powers, malformed/chained expressions and two division-by-zero
candidates remain rejected under V1.

A reproducible seed-7 stratified sample of 32 training question groups was manually
read by the AI agent. Sampled annotation decisions follow V1, but the sample is
not an unbiased precision estimate and no human precision audit has occurred.
Examples include an oval pool treated as a rectangle, and a locally exact
multiplication that reuses an earlier rounded intermediate. Local arithmetic
verification therefore does not establish semantic checkpoint correctness.

## Data-quality findings

- **60 repeated exact training rows**, in 52 repeated sample-ID groups. All are
  retained to preserve baseline exposure; no deduplication was performed.
- **One identical question with conflicting answers:** training rows 44,776 and
  198,445 ask about wheels on 7 motorcycles and 12 buses. One rationale assumes
  four wheels per bus (answer 62); the other assumes six (answer 86). Both local
  arithmetic chains are valid. The question does not resolve the assumption.
- **194 training answers fail plain numeric parsing.** Examples include valid
  fractions, currency-prefixed answers, approximate numbers and time/measurement
  strings. These are a formatting/category finding, not 194 proven wrong answers.
  Official validation/test answers all satisfy numeric parsing.
- Empty rationales: 89 training, 6 validation and 18 test rows. They are retained.
- No literal `<SA>` or LOTUS latent-control-token collisions were found.

The independent membership checker marks primary-comparison readiness false due
to conflicting gold answers. Do not silently choose one value, filter records,
resplit, or normalize away the issue. Any declared treatment must preserve matched
conditions with the baseline; an unchanged-source comparison can disclose and
retain this noise rather than claiming it was repaired.

## Tokenizer audit

The tokenizer/config only was downloaded from `yingfanbot/gsm-cot-gpt2` revision
`e0ef1887404adbf6c531d2cac7804694321f87bd`, with file hashes. No model weights were
downloaded or smoke-tested. The original vocabulary has 50,257 entries and a
1,024-position context. `<SA>` is registered as a distinct additional token.

Full train/validation offsets, causal checkpoint positions, controller schedules,
source-text roundtrip and context lengths are checked without truncating or
writing a massive tokenized dataset. Test is restricted to source format and
control-token checks in this pass; frozen-rule test annotation coverage above is
strictly descriptive. No thresholds, model settings, parser rules or checkpoints
were selected from test results.

Train and validation each passed complete mapping for both policies: 386,120
rows and 614,654 events per policy, with zero annotation/mapping/source-text
roundtrip failures, no shared event token endpoints and no endpoints inside BPE
tokens. Longest explicit sequence: 255 training / 209 validation tokens, below the
1,024-position limit without truncation. With 4 workspace tokens, 1 anchor token,
the fixed 3-token answer prompt and a 64-token generation allowance, the longest
training question requires at most 286 positions.

The source GPT-2 tokenizer has no padding token; the pilot must set the same
explicit padding policy as its baseline when batching. Registration yields a
50,258-token vocabulary with `<SA>` ID 50,257. Real-model embedding resize/tying
validation belongs to the checkpoint pilot, since weights were not downloaded.

Training-only label counts at K=16 are 613,547 positives and 4,537,733 negatives
(11.91% positive). The un-clipped negative/positive ratio is 7.396; this is a
descriptive count, not a selected loss weight or threshold. Use a declared clipped
weight identically across matched arms if weighting is enabled.

## Artifacts and reproduction

Small, reviewable reports:

- `provenance/lotus_corpus_audit.json` (aggregate result)

- `provenance/lotus_download_manifest.json`
- `provenance/lotus_split_manifest.json` (summary pointing to full manifest)
- `provenance/lotus_membership_audit.json` (summary pointing to full details)
- `provenance/lotus_annotation_audit.json`
- `provenance/lotus_annotation_review.json`
- `provenance/lotus_data_quality_findings.json`
- `provenance/lotus_tokenizer_audit.json`

Large artifacts remain under ignored `data/` and `artifacts/` directories. Full
problem-group lists and ordered evaluation IDs are in
`artifacts/lotus-corpus-audit/split_manifest.json`. The streaming audit writes both
annotated policies plus lossless original JSONL under
`artifacts/lotus-annotation-audit/`, alongside its train-only review sample.

```bash
.venv/bin/python scripts/download_lotus_corpus.py
.venv/bin/python scripts/audit_lotus_splits.py
.venv/bin/python scripts/audit_lotus_membership.py \
  --output artifacts/lotus-corpus-audit/membership_audit.full.json
.venv/bin/python scripts/audit_lotus_annotations.py \
  --split-manifest artifacts/lotus-corpus-audit/split_manifest.json \
  --output-dir artifacts/lotus-annotation-audit-rerun \
  --report artifacts/lotus-corpus-audit/annotation-rerun.json
.venv/bin/python scripts/audit_lotus_tokenizer.py \
  --input-dir data \
  --tokenizer artifacts/lotus-tokenizer-audit/source-tokenizer \
  --source-metadata artifacts/lotus-tokenizer-audit/tokenizer_source.json \
  --output provenance/lotus_tokenizer_audit.json \
  --samples artifacts/lotus-tokenizer-audit/mapping_samples.jsonl
.venv/bin/python -m unittest discover -s tests -q
```

Membership audit exits with code 2 when it records a readiness issue; the saved
report remains available. Download reuse checks existing source bytes before
proceeding. Annotation reruns require a fresh output directory. The audit itself
is complete independently of subsequent data-quality decisions and model runs.

Validation: **129 unit tests pass**. Independently produced annotation and
tokenization counts agree; all raw/converted source hashes were rechecked after
the audits. No datasets were repaired, deduplicated, resplit or filtered.

## Subsequent personal semantic review

The user requested assistant review in place of the previously pending external
review. All 194 unusual-answer flags, all 52 duplicate groups and the conflicting
question pair have now been individually adjudicated. Results: 123 correct, 24
acceptable marked approximations, 16 incorrect, 17 ambiguous, 12 underdetermined
and 2 nonanswers. The duplicate exposure is preserved; the bus question has no
unique answer from its wording. See `LOTUS_PERSONAL_REVIEW.md` and
`provenance/lotus_personal_review.json` for the complete ledger and proposed
49-row clean-sensitivity exclusion list. Original data remains unchanged. The
source-issue review is complete; no additional human-review step is imposed for
the small implementation pilot. These targeted findings do not establish the
semantic correctness of the entire training corpus.
