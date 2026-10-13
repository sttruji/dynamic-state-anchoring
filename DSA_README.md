# Dynamic State Anchoring: first implementation

This checkout adds the initial data/tokenization and tiny-model implementation of
the [Revision 4 Markdown handoff](docs/DSA_IMPLEMENTATION_HANDOFF.md) to LOTUS,
preserving its existing code and README. The originally supplied PDF contains the
same substantive specification; switching to the Markdown requires no rebuild.
Branch: `research/dynamic-state-anchoring`. Upstream base:
`eb77e2f7909c5006f58ff0ad7cd6629b942caa9e`.

The PDF describes an accompanying reference package, but the supplied workspace
contained the PDF alone. The annotator, fixtures and tests here are **new
implementations of the written contracts**, not recovered reference files.

## Run locally

The [Colab notebook](notebooks/DSA_Colab.ipynb) provides GitHub checkout, dependency
installation, Drive-backed corpus/model caches, a pretrained inference smoke,
resumable training and question-only evaluation. It clones
`sttruji/dynamic-state-anchoring`. See the [Colab quickstart](docs/COLAB_QUICKSTART.md)
or [open the notebook in Colab](https://colab.research.google.com/github/sttruji/dynamic-state-anchoring/blob/main/notebooks/DSA_Colab.ipynb).

From this repository root, the prepared `.venv` can run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/annotate_milestones.py \
  --input examples/synthetic_gsm8k.jsonl \
  --output examples/annotated_gsm8k.jsonl \
  --report examples/annotation_report.json
.venv/bin/python scripts/smoke_dsa.py --steps 120 --loops 4
.venv/bin/python scripts/tokenize_dsa_data.py \
  --input examples/annotated_gsm8k.jsonl \
  --output examples/tokenized_gsm8k.jsonl \
  --report examples/tokenization_report.json \
  --tokenizer-dir examples/tiny_tokenizer --loops 4
```

To reproduce the environment elsewhere, use Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dsa.lock.txt
```

`requirements-dsa.txt` declares the direct dependencies; the lock records all
installed versions. This is a CPU correctness environment. The existing LOTUS
CUDA environment and requirements have not been merged with inspection repos.

## What works

- Exact, bounded rational arithmetic through an AST whitelist; calculator
  equalities, whole-line numeric equalities and structural variable assignments.
  Two annotation policies, terminal-answer exclusions, lossless spans and raw
  records, stable problem groups and rationale IDs, line-level diagnostics.
- Split manifests that preserve source partitions/order and stop on overlapping
  problem IDs or normalized questions. Source files and preprocessing hashes;
  separately reported comparison with official GSM8K test membership.
- One marker token, full-sequence tokenization, raw Unicode endpoint mapping,
  causal pre-marker targets, collision checks, embedding resize and tying.
  A deterministic byte tokenizer supports offline CPU tests; the HF fast-tokenizer
  adapter is tested with a locally trained BPE vocabulary.
  The latent answer route deliberately fixes the inference prompt tokens before
  tokenizing the answer continuation, preventing cross-boundary BPE merges from
  changing teacher-forcing inputs relative to generation.
- GPT-2 and Llama hidden-only backbone adapters. Each recurrent forward rebuilds
  question + anchor + workspace, with explicit padding-aware position IDs and
  `use_cache=False`. Only answer/explicit routes materialize vocabulary logits.
- Static, every-loop, periodic and learned write controls with identical modules.
  Hard forward writes and straight-through training gradients; no gold write
  schedule in the model input API.
- Same-backbone explicit/answer CE, masked controller BCE, and detached,
  normalized candidate alignment for all four write policies.
- Tiny single-device training, per-loss/gradient logs, question-only greedy
  generation, tokenizer/model checkpoint reload and CPU optimizer/RNG resume.
  Evaluation helpers mirror the pinned LOTUS extraction and scoring rules.

## Synthetic smoke artifacts

`artifacts/tiny-smoke/report.json` records the seed, all dependency versions,
implementation hashes, initialization/data identity, losses, parameters, gate
traces, question-only predictions and checkpoint checks. Each arm also saves its
backbone, tokenizer, anchor configuration and weights, optimizer/RNG resume state,
and step metrics.

The two synthetic training examples test memorization and plumbing only. At seed
7 after 120 updates, both arms memorize their answers; the learned gate makes
zero writes at threshold 0.5. This is **not evidence that anchoring improves
reasoning, extrapolation, or benchmark performance**. Output artifacts are ignored
by Git; small provenance and coverage reports are retained.

## Corpus audit and next pilot

The pinned LOTUS corpus has been downloaded and audited. See the
[corpus audit](docs/LOTUS_CORPUS_AUDIT.md) for coverage, ordered test membership,
tokenizer checks, reproducible commands and data-quality findings. All source
records are retained; 83.49% of training rows have controller/checkpoint labels at
K=16. One repeated training question has conflicting gold answers; the personal
review documents it as underdetermined and retains it in the unchanged primary
pilot. Quality exclusions remain a separate proposed sensitivity experiment.

Use `.venv/bin/python scripts/download_lotus_corpus.py` to verify/reuse the exact
pinned sources; this preserves raw files and rejects checksum mismatches. Full
annotation derivatives and group/evaluation manifests are stored under ignored
`artifacts/` directories; concise hashed reports are under `provenance/`.

The next model step is to smoke-test the unmodified pinned GPT-2 checkpoint and
then overfit a small training subset with profiling. A tokenizer/config pin is
recorded, but pretrained model weights have not been downloaded. The [personal semantic review](docs/LOTUS_PERSONAL_REVIEW.md) is complete;
source-data noise is documented and proposed clean sensitivity exclusions remain
separate from the unchanged primary benchmark. No GPU jobs have run.

The pilot launcher and persistence workflow are implemented. The first hosted
run still needs to verify unmodified pretrained inference, overfit/profile a
small training subset, and inspect validation outputs before scaling up.

See [implementation status](IMPLEMENTATION_STATUS.md),
[upstream inspection](docs/UPSTREAM_INSPECTION.md), and
[pinned sources](provenance/sources.json).
