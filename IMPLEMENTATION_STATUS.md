# Implementation status — updated 13 October 2026

First-stage data/tokenization and tiny-model implementation is working. This is
not a trained benchmark model or evidence for the anchoring hypothesis.

The subsequently supplied `DSA_IMPLEMENTATION_HANDOFF.md` was compared in full
with the PDF: the Revision 4 requirements are substantively identical. The
unchanged Markdown is preserved at `docs/DSA_IMPLEMENTATION_HANDOFF.md` and is the
handoff used going forward. Document hashes and verification results are recorded
in `provenance/handoff_verification.json`.

## Completed

- LOTUS cloned on `research/dynamic-state-anchoring` from
  `eb77e2f7909c5006f58ff0ad7cd6629b942caa9e`; four inspection sources pinned.
  Source SHAs, relevant file hashes, licenses and actual upstream interfaces are
  recorded in `provenance/sources.json` and `docs/UPSTREAM_INSPECTION.md`.
- New exact-arithmetic annotation implementation, JSONL CLI, two policies,
  immutable source records, grouped IDs, deterministic training-only loop labels,
  malformed-record rejection and coverage reports. The PDF's referenced original
  package was absent, so none of its reported checks were assumed to pass here.
- Source-preserving LOTUS adapter, overlap-stopping split-manifest builder and
  corpus preparation CLI. Primary membership is never regenerated.
- Tiny byte tokenizer and HF fast-tokenizer support, special-token registration,
  full marked explicit tokenization, Unicode character offsets, pre-marker hidden
  indices, checkpoint tying/new-row gradients, tokenized artifacts/provenance CLI.
- GPT-2/Llama hidden-only adapters; immutable question inputs, full recomputation,
  persistent A and S=4 workspace, padding-aware positions, four matched write
  policies and primary straight-through hard gate.
- Shared explicit/answer losses, masked gate BCE and detached normalized candidate
  alignment in all write-policy arms; no-event/overflow rows retain CE losses.
- Single-device training, logs, question-only greedy generation, pinned LOTUS
  answer scoring, save/reload and exact CPU resume. Resume rejects changed model,
  loss settings, tokenized record content/order and scheduler presence.

## Observed validation

Exact commands run from this repository root:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/annotate_milestones.py \
  --input examples/synthetic_gsm8k.jsonl \
  --output examples/annotated_gsm8k.jsonl \
  --report examples/annotation_report.json
.venv/bin/python scripts/tokenize_dsa_data.py \
  --input examples/annotated_gsm8k.jsonl \
  --output examples/tokenized_gsm8k.jsonl \
  --report examples/tokenization_report.json \
  --tokenizer-dir examples/tiny_tokenizer --loops 4
.venv/bin/python scripts/smoke_dsa.py --steps 120 --loops 4 --output artifacts/tiny-smoke
.venv/bin/python -m compileall -q dsa scripts/annotate_milestones.py scripts/prepare_dsa_data.py scripts/smoke_dsa.py scripts/tokenize_dsa_data.py
git diff --check
```

Initial implementation suite: **108 tests passed**. Log: `artifacts/validation-unittest.txt`.
Tests include causal future-token isolation, batch/unbatched padding parity,
per-example writes, exact closed-gate persistence, surrogate gradients,
matched-control supervision, gold-input isolation, marker tying/reload,
resume equality/config rejection, malformed-file atomicity and split overlaps.

Synthetic smoke: random two-layer width-32 GPT-2, 67,841 parameters, seed 7,
4 loops, 120 optimizer steps for each static/learned arm, identical initialization
and data order. Loss moved 11.2953→0.3372 (static), 11.2953→0.2855 (learned).
Both generate `10` and `14` correctly from the two training questions; checkpoint
reload preserves outputs. The learned arm makes **zero hard writes** at 0.5.
These are memorization/correctness observations, not comparative research results.
Tracked run manifest: `provenance/tiny_smoke_report.json`.

Raw-data development annotation (official TRAIN only, separate from LOTUS):
7,473/7,473 rows retained. Conclusive policy retains 16,718 events with 350
no-event rows; verified-only retains 16,714 events with 352 no-event rows.
Programmatic source/span checks passed on every row. Reports are under
`provenance/raw_annotation_dev_*_report.json`; a seed-7 sample awaits human review.
Official test was not used for this development audit.

## Issues found and resolved

- LOTUS hardcodes recurrence KV caches; DSA therefore uses its own hidden-only
  full-recomputation adapter rather than changing a misleading cache flag.
- Some BPE vocabularies merge the answer prompt and target into one token. The
  latent route now deliberately fixes the generation prompt tokens before
  tokenizing continuation targets. The explicit route remains tokenized once.
- Raw GSM8K contains Unicode line separators inside a JSON string. JSONL readers
  now iterate physical LF records instead of calling `str.splitlines()`.
- Resume previously lacked experiment-setting validation; checkpoints now include
  model/loss settings and a tokenized record-order checksum.
- Handoff reconciliation found that upstream's broad `data/` Git ignore rule also
  hid the new `dsa/data/` source modules. It is now scoped to `/data/`, retaining
  the root dataset exclusion while making all implementation sources visible.

## LOTUS corpus audit

The corpus audit is complete with data-quality findings; 129 unit tests pass.
All 385,620 training, 500 validation and 1,319 test records are retained. Raw
source hashes match the pinned dataset, no exact/format-normalized split overlap
was found, and ordered test questions/answers match the official test release.
Both annotation policies yield 613,547 retained training events, covering 83.49%
of training rows at K=16. No parser extension or corpus filtering was performed.

Findings include 60 exact repeated training rows, one ambiguous identical question
with conflicting answers, 194 answers outside plain numeric formatting, and no
original problem IDs to rule out paraphrased duplicates. A 32-record stratified
TRAIN-only sample has AI-agent review. The user subsequently authorized personal
semantic source-issue review, now complete: all 194 unusual answers, 52 duplicate
groups and the conflicting pair were adjudicated; see `docs/LOTUS_PERSONAL_REVIEW.md`.
The full GPT-2 tokenizer mapping pass has zero failures on all train/validation
rows; maximum explicit lengths are 255/209 tokens, with no truncation. See
`docs/LOTUS_CORPUS_AUDIT.md` for source evidence and tokenizer results.

## Colab pilot workflow

`notebooks/DSA_Colab.ipynb` is built for the public
`sttruji/dynamic-state-anchoring` repository. Its first cell clones GitHub;
later cells install isolated dependencies, mount Drive, cache pinned data/model
downloads, run an unmodified-checkpoint smoke, train/resume matched policies and
evaluate question-only predictions. Final full test evaluation is explicitly
disabled during pilot development. See `docs/COLAB_QUICKSTART.md` and
`docs/COLAB_ENVIRONMENT.md`.

Persistent preparation validates source and derivative hashes. Training saves
model, tokenizer, optimizer, scheduler, RNG and frozen experiment/runtime
identities, retaining two complete checkpoints by default. Evaluation recovers
interrupted prediction logs, binds checkpoint/source hashes and preserves the
complete final-test denominator. Exact training resume rejects changed runtime
or experimental identities.

Initial Colab workflow validation: **175 tests pass**, including actual execution of the embedded
CPU preflight and notebook summary. A CLI integration prepared 32 unchanged LOTUS
training records with the offline tokenizer, trained a random tiny GPT-2 to step
3, resumed to step 5 and evaluated eight validation questions at K=16. This
verifies plumbing; it is not pretrained inference or an accuracy result. Detailed
verification is recorded in `provenance/colab_validation.json`; the suite log is
`artifacts/colab-validation-unittest.txt`.

Hosted Colab/Drive authorization and pretrained CUDA execution remain pending.
No source dataset records were changed or filtered.

The user's Python 3.13 Colab runtime subsequently failed during venv creation.
Setup now creates a pip-less environment and uses the host pip with an explicit
target interpreter, repairs partial environments, and preserves the installed
CUDA stack. The preflight now accepts Python 3.13. **186 tests pass on both
Python 3.12 and Python 3.13 locally**, and a real Python 3.13 setup installation
completed. Original failure stderr was not supplied, so its exact cause is not
asserted. See `provenance/colab_setup_fix.json`; hosted GPU verification remains
pending.

## Remaining work

1. Use the reviewed unchanged-source policy for the implementation pilot and
   predeclare a separate clean sensitivity experiment if desired. Source-issue
   adjudication is complete; the proposed 49-row quality exclusions and optional
   60-row deduplication are not applied to the primary benchmark.
2. Execute the unmodified pretrained inference smoke at the pinned
   `yingfanbot/gsm-cot-gpt2` revision
   `e0ef1887404adbf6c531d2cac7804694321f87bd`. Tokenizer mapping/context audits are
   complete, but pretrained model weights have not been downloaded locally.
3. Run the new Colab notebook from the public
   `sttruji/dynamic-state-anchoring` repository:
   verify hosted dependencies/GPU, overfit a small training subset and profile
   memory/throughput. The single-device pilot launcher, Drive cache and resumable
   training/evaluation workflow are implemented; hosted execution is pending.
4. Implement controlled addition/function-composition data with solver states,
   fixed train/test difficulty splits, depth curves and causal interventions.
5. Run matched B0/B1/B2/DSA and unmodified LOTUS comparisons with validation-only
   selection, complete ordered test denominators, three seeds and measured costs.
   Distributed training and optimized caches remain deferred.
