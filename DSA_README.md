# Dynamic State Anchoring: the original design

**Status, 12 November 2026: this file describes the original DSA design and its implementation, built in October
2026.** That design did not work, and the project moved on to testing its premise on published latent CoT checkpoints.
The [README](README.md) gives where the project ended. The design is specified in the
[Revision 4 handoff](docs/DSA_IMPLEMENTATION_HANDOFF.md), and its results are in [FINDINGS.md](docs/FINDINGS.md).

The code is on `main`. The upstream LOTUS base is `eb77e2f7909c5006f58ff0ad7cd6629b942caa9e`. The handoff describes an
accompanying reference package, but the supplied workspace contained the PDF alone. The annotator, fixtures and tests
here are **new implementations of the written contracts**, not recovered reference files.

## The design

DSA adds to a shared GPT-2 backbone four latent workspace vectors, one persistent anchor vector, an anchor projection
head and a scalar write controller. At each loop, the workspace is updated from the question, the previous anchor and the
previous workspace. A mean-pooled summary yields a candidate anchor and a write probability, and a hard gate at 0.5
(straight-through in training) decides whether the candidate replaces the stored anchor.

- **Write supervision.** A deterministic annotator inserts an `<SA>` marker after each locally verified arithmetic
  conclusion in a GSM8K-Aug rationale, excluding the final answer. Marked positions are mapped to loop indices, and the
  explicit route's hidden states before each marker supply anchor targets. Gold markers are never given at inference.
- **Loss.** Answer cross-entropy + explicit-CoT cross-entropy + 0.1 controller BCE + 0.1 normalized anchor alignment.
- **Recurrence.** Adapted from the Fully Looped Transformer (FLT): previous-loop memory vectors supply the queries at
  every block ([K8_FLT_PILOT.md](docs/K8_FLT_PILOT.md)).
- **Start and controls.** Training starts from the LOTUS GPT-2 CoT checkpoint `yingfanbot/gsm-cot-gpt2`. The controls
  are no writes (`static`), `every_loop` and `periodic` (every fourth loop), each sharing the learned arm's
  initialization, data order and supervision.

## What happened

| Run | Result | Details |
|---|---|---|
| K=16 pilot, standard recurrence, 32 questions, 200 updates | Learned and static each 3/128 on validation; answer-gradient norms grew from about 3×10² at one loop to 2.4×10⁶; controller AUROC 0.510 | [FINDINGS §1](docs/FINDINGS.md#1-small-standard-recurrence-pilot-k16) |
| K=8 pilot, standard against FLT recurrence | FLT cut median gradient norms 88–279-fold and fit 29–31 of the 32 training questions, but validation stayed at 2–3/128; controller AUROC 0.507 | [FINDINGS §2](docs/FINDINGS.md#2-small-k8-recurrence-ablation) |
| FLT data-scale run, 512 questions, 1,024 updates | No writes 23/500, every loop 20, periodic 17, learned 14, with no pairwise difference at p < 0.05; the controller made 8 writes in 4,000 loop decisions | [FINDINGS §3](docs/FINDINGS.md#3-bounded-flt-data-scale-experiment) |
| Local inference follow-up | The original checkpoint's explicit route answered 25 of 64 validation questions, the fine-tuned routes 13–17 | [FINDINGS §4](docs/FINDINGS.md#4-local-inference-follow-up) |
| CODI-style recipe, 256 updates | Explicit route 205/500; latent answers 37/500 with no writes, 44 every loop and 45 periodic | [FINDINGS §5](docs/FINDINGS.md#5-codi-recipe-no-write-arm) |

The controller never learned when to write, and no write policy beat writing nothing, so the design was parked. The
premise was then tested directly on the published CODI and SIM-CoT checkpoints, which led to the reframing of DSA as a
gate over latent cache entries ([README](README.md#how-the-project-got-there)).

## Run locally

For native Apple GPU training and matched local checks, see the
[MPS workflow](docs/LOCAL_MPS.md). The trainer, checkpoint smoke and answer
evaluator automatically select MPS when CUDA is unavailable and MPS is available.

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

## Colab notebooks for this design

| Notebook | What it runs |
|---|---|
| [DSA_Colab.ipynb](notebooks/DSA_Colab.ipynb) | The 32-question pilots: K=16, then K=8 standard against FLT ([protocol](docs/K8_FLT_PILOT.md), [quickstart](docs/COLAB_QUICKSTART.md)) |
| [DSA_Pilot_Diagnostics.ipynb](notebooks/DSA_Pilot_Diagnostics.ipynb) | The older standalone diagnostics for the saved K=16 pilot |
| [DSA_FLT_DataScale.ipynb](notebooks/DSA_FLT_DataScale.ipynb) | The FLT data-scale run ([protocol](docs/FLT_DATA_SCALE.md)) |
| [DSA_Format_Test_Colab.ipynb](notebooks/DSA_Format_Test_Colab.ipynb) | Retrains the learned data-scale arm with one reasoning step per line ([audit](docs/AUDIT_20261025.md), finding 1); no results from it are in the repository |
| [DSA_CODI_Recipe_Colab.ipynb](notebooks/DSA_CODI_Recipe_Colab.ipynb) | The no-write arm on CODI's recipe, and every baseline on every evaluation set |
| [DSA_CODI_Writes_Colab.ipynb](notebooks/DSA_CODI_Writes_Colab.ipynb) | The every-loop and periodic arms on the same recipe |

The [CODI-style options](docs/CODI_RECIPE.md) add padded batching, LoRA, a separate
new-module learning rate, warmup/cosine schedules and CODI's answer-position
distillation; all default to off. Colab setup is described in
[COLAB_ENVIRONMENT.md](docs/COLAB_ENVIRONMENT.md).

## What the implementation does

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
- Single-device training, per-loss/gradient logs, question-only greedy
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

## Corpus audit

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

See [implementation status](IMPLEMENTATION_STATUS.md),
[upstream inspection](docs/UPSTREAM_INSPECTION.md), and
[pinned sources](provenance/sources.json).
