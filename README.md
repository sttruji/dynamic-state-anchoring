# Dynamic State Anchoring

An ECS 282 (Advanced NLP, UC Davis) course project by Steven Trujillo, Joshua Dias Barreto and Xingming Xu, on why
latent chain-of-thought (CoT) models lose accuracy when run past their trained number of latent steps. The repository
is [`sttruji/dynamic-state-anchoring`](https://github.com/sttruji/dynamic-state-anchoring), a fork of
[LOTUS](https://github.com/yingfan-bot/lotus), whose documentation follows at the end.

**Status, 12 November 2026:** the experiments are finished, and the course report is a first draft in
[`paper/main.tex`](paper/main.tex). The original Dynamic State Anchoring (DSA) design did not work. Testing its premise
showed why latent CoT models lose accuracy past their trained depth, and it led to a training-free repair that holds
accuracy at depth on four checkpoints. Nothing tested turns the extra depth into accuracy.

## Headline results

- **The original DSA failed.** On GPT-2, its learned anchor-write controller was at chance and stopped writing, and no
  write policy beat writing nothing.
- **Latent CoT drifts past its trained depth.** The published CODI and SIM-CoT checkpoints, at GPT-2 and LLaMA-3.2-1B
  scale, all lose accuracy at 24 latent steps against the 6 they were trained with, on held-out data.
- **Two mechanisms, split exactly by a position control.** Near-copy latent cache entries dilute the answer's attention,
  which dominates at GPT-2 scale; under rotary position embeddings (RoPE), the answer's growing distance from the
  latents dominates at 1B.
- **A matching training-free repair holds depth.** Soft deduplication on GPT-2, and soft deduplication plus re-anchoring
  the answer at 1B, keep accuracy at 24 steps within 2 points of the 6-step score on all four models.
- **Repair, not gain.** No policy, trained gate or further fine-tuning turns the extra depth into accuracy.

## How the project got there

### 1. The original design

The proposal argued that intermediate conclusions are corrupted as latent states pass through many recurrent
transformations, so a model that keeps completed conclusions in a persistent anchor state should stay accurate, or
improve, given more iterations. DSA added to a shared GPT-2 backbone four latent workspace vectors, one persistent anchor
vector, an anchor projection head and a scalar write controller with a hard gate. A deterministic annotator marked
613,547 arithmetic conclusions on 83.5% of GSM8K-Aug training rows to supervise write timing and anchor targets. The
recurrence was adapted from the Fully Looped Transformer (FLT), training started from the LOTUS GPT-2 CoT checkpoint,
and the controls were no writes (`static`), `every_loop` and `periodic`, each sharing the learned arm's initialization,
data order and supervision. See [DSA_README.md](DSA_README.md) and the
[original specification](docs/DSA_IMPLEMENTATION_HANDOFF.md).

### 2. Why it failed

- **The data-scale run** (FLT, 8 loops, 512 training questions, 1,024 updates, all 500 validation questions): no writes
  scored highest, 23/500, against 20 for every loop, 17 for periodic and 14 for learned writes. No pairwise difference
  reached p < 0.05.
- **The controller** ranked true conclusion loops at chance (event AUROC 0.513), and its write rate fell to zero: at
  inference it made 8 writes in 4,000 loop decisions.
- **The explicit route was damaged.** On 64 validation questions the original checkpoint answered 25 correctly, and the
  fine-tuned routes 13–17.
- **A stronger CODI-style recipe did not help either.** With LoRA rank 128, 128 records per update and answer-position
  distillation, one pass over 32,768 records (256 updates) kept the explicit route at 205/500 but reached only 37/500
  latent answers with no writes. Every-loop and periodic arms on identical batches scored 44 and 45 (p = 0.34 and 0.23
  against no writes).

The design was parked. See [FINDINGS.md](docs/FINDINGS.md) and [CODI_RECIPE.md](docs/CODI_RECIPE.md).

### 3. Testing the premise on published checkpoints

The published CODI GPT-2 and SIM-CoT GPT-2 checkpoints, both trained with 6 latent steps, were run past that depth with
no training. These analyses are exploratory and use the 500-question `gsm8k_valid` set, which was reused for discovery
throughout; the first preregistered test was replicated on SVAMP.

| Correct of 500 | 6 steps | 12 steps | 24 steps |
|---|---:|---:|---:|
| CODI GPT-2 | 206 | 192 | 173 |
| SIM-CoT GPT-2 | 204 | 185 | 170 |

- **Two preregistered ideas failed.** Agreement between depths was a weaker confidence signal than the 6-step answer's
  token log-probability (SVAMP AUROC 0.81 against 0.91), and re-feeding the step-6 latent did not beat the plain loop
  (165 against 173 at 24 steps).
- **Not mainly because the states deteriorate.** CODI's latents alternate between two states and lose question content (a
  step-24 latent identifies its own question among the 500 only 67.4% of the time). SIM-CoT's latents settle and keep
  their content (97.6%), yet it loses as many answers.
- **The copy test.** After 6 real steps, appending exact copies of the step-6 cache entry until the cache holds 24
  latent entries scores 172 on SIM-CoT, against 170 for the real 24-step loop: copies reproduce its entire loss. On
  CODI, copies of the last entry or the last pair score 196 and 188 against 173, about a third to a half of its loss.
- **The answer's attention moves.** Its first token's attention share on latents 1–6 falls from 0.158 after 6 steps to
  0.033 after 24 in CODI, and from 0.099 to 0.036 in SIM-CoT.

See [CODI_TRAIL.md](docs/CODI_TRAIL.md) for CODI's trail, and [LATENT_GATE_IDEA.md](docs/LATENT_GATE_IDEA.md) for
SIM-CoT and the copy test.

### 4. DSA reframed: a gate over latent cache entries

If the cache, not the state, is the problem, the thing to anchor is which latent entries the answer can see. The
reframed DSA scores each latent entry when it is produced, adds log g to every attention logit pointing at it, and keeps
at most 6 entries, the trained count, evicting the lowest-scoring one. A training-free alternative, **soft dedup**,
applies once there are more than 6 latent entries: each entry gets the bias −log c, where c counts the latent entries
whose fed latents have cosine similarity of at least 0.95 with it, so a cluster of k exact copies draws the attention of
a single entry.

- **Stage 0, eviction at fixed weights.** At 24 steps with at most 6 entries visible, keeping the first six scores 205
  (CODI) and 204 (SIM-CoT), restoring the 6-step score, and dropping near-duplicates scores 201 and 203. Recency and
  random eviction hurt. The planned gate label, a logit-lens decode matched to gold intermediate results, predicted an
  entry's causal effect worse than chance (AUROC 0.41–0.45) and was dropped.
- **Stage 1, fine-tuning with the gate.** The main hypothesis, H2, was that a model fine-tuned at 12 steps under the
  gated budget beats the same model fine-tuned at 6 (`ft6`) on the 173 problems with 4 or more rationale steps. It
  failed on both models (SIM-CoT gates 26 and 28 against 31; CODI gate 21 against 30), so the planned stop rule fired.
  On SIM-CoT the gate rediscovered "keep the first six" (exactly entries 1–6 on 196 of 500 questions); on CODI it
  failed. Soft dedup held the 6-step score on both base models with no training: at 24 steps, 207 against 204 on
  SIM-CoT and 201 against 206 on CODI.

See [LATENT_GATE_IDEA.md](docs/LATENT_GATE_IDEA.md),
[LATENT_GATE_STAGE1_RESULTS.md](docs/LATENT_GATE_STAGE1_RESULTS.md) and
[LATENT_GATE_LITERATURE.md](docs/LATENT_GATE_LITERATURE.md).

### 5. The preregistered held-out and scale study

A plan frozen with a SHA-256 hash before any run tested the diagnosis on untouched data and at a second scale. Four
published checkpoints, all trained with 6 latent steps (CODI and SIM-CoT, each at GPT-2 and LLaMA-3.2-1B), were scored
on a primary held-out pool of 2,818 questions (GSM8K test 1,319, GSM-Hard 1,319, MultiArith 180) and on SVAMP (1,000)
as a secondary set. The LLaMA port reproduces the published GSM8K accuracy: 55.5% for CODI-1B (published 55.6%) and
55.7% for SIM-CoT-1B (published 56.1%).

| Primary pool, correct of 2,818 | `plain`@6 | `plain`@24 | `dedup_soft`@24 | `dedup_anchor`@24 |
|---|---:|---:|---:|---:|
| SIM-CoT GPT-2 | 799 | 757 | 801 | – |
| CODI GPT-2 | 854 | 718 | 843 | – |
| CODI-1B | 1077 | 1027 | 1029 | 1055 |
| SIM-CoT-1B | 1075 | 976 | 997 | 1049 |

`plain` keeps every latent entry; `dedup_anchor` is re-anchoring after soft dedup (below) and applies to the RoPE models
only. The paper's main table adds more policies and depths, and the per-set
counts are in [report.txt](results/latent_gate/heldout_scale/report.txt) and
[amendment_report.txt](results/latent_gate/heldout_scale/amendment_report.txt).

- **The drift is real on every model** (rule D1): 24 steps cost 1.5 to 4.8 points against 6.
- **Soft dedup fixes GPT-2 but not 1B** (rule D2): it lands within half a point of the 6-step score on both GPT-2
  models, but stays 1.7 and 2.8 points below it on the 1B models.
- **The position control** (`shift`@6: 6 steps, with the answer placed where it would sit after 24) splits the loss
  exactly into a position part and a content part. On GPT-2, with learned absolute positions, the whole loss is content,
  the near-copy dilution that soft dedup removes. On the 1B LLaMA models, position accounts for all of CODI-1B's loss and
  two thirds of SIM-CoT-1B's.
- **Amendment 1, B3** (written after the main runs and before any run of what it added): `dedup_anchor` runs a soft-dedup
  loop, then rotates cached RoPE keys so that each later near-copy moves to its earlier copy's position, every other later
  entry takes the next free position, and the answer starts right after the last position used. At 24 steps it holds both
  1B models within 0.8 and 0.9 points of the 6-step score, inside the 2-point margin.
- **Amendment 1, Part C**, an exploratory fine-tune of SIM-CoT-1B with the Stage 1 recipe, had no passing arm, and the
  fine-tune itself cost accuracy (`ft6` scored 246 against the base model's 277 of 500 on validation).
- **No policy at any depth significantly beats stopping at 6 steps** on the primary pool. The one exception in the
  study is on the secondary set (SIM-CoT GPT-2 with soft dedup at 24 steps on SVAMP, 415 against 401, p = 0.016), which
  is not counted as a claim.

See [LATENT_GATE_HELDOUT_RESULTS.md](docs/LATENT_GATE_HELDOUT_RESULTS.md) and the
[frozen plan](results/latent_gate/heldout_scale/PLAN.md).

### 6. The write-up

[`paper/`](paper) holds the ACL-format course report draft, *Dynamic State Anchoring Revisited: Why Latent
Chain-of-Thought Loses Accuracy Past Its Trained Depth*. It is the current account of the project, and its numbers come
from the docs and records below.

A separate track tried methods meant to raise accuracy above the 6-step score; it is summarized in
[BRAINSTORM_SUMMARY.md](docs/BRAINSTORM_SUMMARY.md) and is not part of the report.

## Where to find things

**The main line, in order**

| Doc | What it covers |
|---|---|
| [docs/FINDINGS.md](docs/FINDINGS.md) | The original DSA: pilots, the data-scale run, the local follow-up and the CODI-recipe arms |
| [docs/CODI_RECIPE.md](docs/CODI_RECIPE.md) | The CODI-style training options and their Colab notebooks |
| [docs/CODI_TRAIL.md](docs/CODI_TRAIL.md) | CODI GPT-2 past its 6 trained steps, with two preregistered tests |
| [docs/LATENT_GATE_IDEA.md](docs/LATENT_GATE_IDEA.md) | The reframing: SIM-CoT, the copy test, Stage 0 and the gate |
| [docs/LATENT_GATE_LITERATURE.md](docs/LATENT_GATE_LITERATURE.md) | The literature review behind the reframing, written before Stage 0 finished |
| [docs/LATENT_GATE_STAGE1_RESULTS.md](docs/LATENT_GATE_STAGE1_RESULTS.md) | Stage 1 on SIM-CoT GPT-2 and the CODI replication |
| [docs/LATENT_GATE_HELDOUT_RESULTS.md](docs/LATENT_GATE_HELDOUT_RESULTS.md) | The held-out and scale study, Amendment 1's B3 and Part C |
| [paper/main.tex](paper/main.tex) | The course report draft |

**Records.** The plans in these folders were written before their runs; the latent-gate plans carry a `PLAN.sha256`.
- [results/leaderboard.csv](results/leaderboard.csv): benchmark rows for the baselines and the original DSA arms.
- [results/codi_trail/](results/codi_trail): the CODI trail's preregistration, dynamics plan and outputs.
- [results/latent_gate/](results/latent_gate): `simcot_probe/`, `copy_test/`, `stage0/`, `stage1/` and `heldout_scale/`,
  with their plans, records and reports.
- [provenance/](provenance): hashes and verification records for the original implementation and data.

**Infrastructure and history**

| Doc | What it covers |
|---|---|
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | What was built for each stage, where its code and records are, and how it ran |
| [DSA_README.md](DSA_README.md) | The original DSA design and how to run it |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | Pinned datasets, the evaluation harness and the baselines |
| [docs/LOCAL_MPS.md](docs/LOCAL_MPS.md) | Running on Apple MPS |
| [docs/DSA_IMPLEMENTATION_HANDOFF.md](docs/DSA_IMPLEMENTATION_HANDOFF.md) | The original Revision 4 specification (superseded plan) |
| [docs/K8_FLT_PILOT.md](docs/K8_FLT_PILOT.md), [docs/FLT_DATA_SCALE.md](docs/FLT_DATA_SCALE.md), [docs/COLAB_QUICKSTART.md](docs/COLAB_QUICKSTART.md), [docs/COLAB_ENVIRONMENT.md](docs/COLAB_ENVIRONMENT.md) | Protocols and Colab setup for the original pilots; their results are in FINDINGS |
| [docs/AUDIT_20261025.md](docs/AUDIT_20261025.md), [docs/CHECKPOINT_LABEL_AUDIT_20261019.md](docs/CHECKPOINT_LABEL_AUDIT_20261019.md), [docs/MPS_CONTROLLED_EXPERIMENTS_20261019.md](docs/MPS_CONTROLLED_EXPERIMENTS_20261019.md) | Dated audits from October, and a superseded experiment design |
| [docs/UPSTREAM_INSPECTION.md](docs/UPSTREAM_INSPECTION.md), [docs/LOTUS_CORPUS_AUDIT.md](docs/LOTUS_CORPUS_AUDIT.md), [docs/LOTUS_PERSONAL_REVIEW.md](docs/LOTUS_PERSONAL_REVIEW.md) | The October audits of the upstream code and the LOTUS training corpus |

## Reproduce

Set up the environment and the pinned datasets, then score a baseline:

```bash
git clone https://github.com/sttruji/dynamic-state-anchoring.git
cd dynamic-state-anchoring
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dsa.lock.txt
.venv/bin/python scripts/setup_data.py
.venv/bin/python scripts/evaluate.py --model codi --suite dev
```

Re-apply the held-out study's decision rules to the committed records, with no GPU. These print `report.txt` and
`amendment_report.txt` in [results/latent_gate/heldout_scale](results/latent_gate/heldout_scale):

```bash
.venv/bin/python scripts/latent_gate/heldout_report.py results/latent_gate/heldout_scale
.venv/bin/python scripts/latent_gate/amendment_report.py results/latent_gate/heldout_scale
```

The GPU runs used Colab notebooks; each notebook's first cell says which runtime it needs and how it resumes.

| Stage | Notebook |
|---|---|
| Original DSA pilots and data-scale run | [DSA_Colab.ipynb](notebooks/DSA_Colab.ipynb), [DSA_FLT_DataScale.ipynb](notebooks/DSA_FLT_DataScale.ipynb) |
| CODI-recipe arms and the baseline rows | [DSA_CODI_Recipe_Colab.ipynb](notebooks/DSA_CODI_Recipe_Colab.ipynb), [DSA_CODI_Writes_Colab.ipynb](notebooks/DSA_CODI_Writes_Colab.ipynb) |
| Stage 1 | [DSA_Latent_Gate_Stage1_Colab.ipynb](notebooks/DSA_Latent_Gate_Stage1_Colab.ipynb) |
| Held-out and scale, Parts A and B | [DSA_Latent_Gate_HeldOut_Scale_Colab.ipynb](notebooks/DSA_Latent_Gate_HeldOut_Scale_Colab.ipynb) |
| Amendment 1, B3 and Part C | [DSA_Latent_Gate_Anchor_PartC_Colab.ipynb](notebooks/DSA_Latent_Gate_Anchor_PartC_Colab.ipynb), [DSA_Latent_Gate_PartC_Colab.ipynb](notebooks/DSA_Latent_Gate_PartC_Colab.ipynb) (Part C only) |

The CODI trail, the copy test and Stage 0 ran locally on Apple MPS; their commands are in
[CODI_TRAIL.md](docs/CODI_TRAIL.md#reproduce) and [scripts/latent_gate/evidence/README.md](scripts/latent_gate/evidence/README.md),
and in the docstrings of `scripts/latent_gate/stage0_*.py`.

Build the report, which needs a TeX distribution with `pgfplots`:

```bash
cd paper && latexmk -pdf main.tex
```

`python3 paper/figures/depth_figure.py` regenerates its figure from the held-out records.

The original LOTUS documentation and attribution follow.

---

# Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers

This project is for the paper: [Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers](https://arxiv.org/abs/2606.31779).

<!-- Project Page URL assumes GitHub Pages is enabled (Settings -> Pages -> deploy from branch: main, folder: /docs) on the public yingfan-bot/lotus repo. -->
🔗 **[Project Page](https://yingfan-bot.github.io/lotus/)** &nbsp;·&nbsp; 📄 **[arXiv](https://arxiv.org/abs/2606.31779)** &nbsp;·&nbsp; 🤗 **[Models &amp; collection](https://huggingface.co/collections/yingfanbot/looped-padded-6a552f7ef667cb41db2431a3)** &nbsp;·&nbsp; 𝕏 **[Summary](https://x.com/yingfan_bot/status/2077441955202482419)**

## Repository layout

```text
.
├── args/                    # YAML training/evaluation configs
├── data/                    # JSON datasets used by the provided configs
├── preprocessing/           # GSM8K download + preprocessing scripts
├── launch_train.sh          # Public torch.distributed launcher
├── environment.yml          # Conda env (Python 3.12, PyTorch 2.7 + CUDA 12.8)
├── requirements.txt         # Python deps layered on top of the NGC image
└── scripts/                 # Python entry points and modules
    ├── run.py               # Main training entry point
    ├── eval.py              # Standalone evaluation entry point
    ├── dataset.py           # Data loading/collation utilities
    ├── lotus.py             # LOTUS latent reasoning model wrapper
    └── utils.py             # Config and seed helpers
```

## Environment

We recommend the NVIDIA NGC PyTorch image (it provides a CUDA-matched PyTorch stack, so
`requirements.txt` does not install `torch`):

```text
nvcr.io/nvidia/pytorch:25.03-py3
```

### Docker

From the repository root:

```bash
docker run --gpus all --rm -it \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD":/workspace/lotus \
  -w /workspace/lotus \
  nvcr.io/nvidia/pytorch:25.03-py3

pip install -r requirements.txt
```

### Conda environment

If you are not using a container, an `environment.yml` is provided that mirrors the NGC
image (Python 3.12, PyTorch 2.7 + CUDA 12.8). From the repository root:

```bash
conda env create -f environment.yml
conda activate lotus
```

## Authentication

If you need private Hugging Face models or W&B logging, log in before launching:

```bash
huggingface-cli login
wandb login
```

The launcher itself does not handle credentials.

## Data

The configs read JSON splits of the form `{"question", "steps", "answer"}` from `./data/`
(git-ignored, not shipped). To download and preprocess the augmented-CoT GSM8K splits into
that format, run from the repo root:

```bash
bash preprocessing/gsm_icot.bash
```

This produces `data/gsm_{train,valid,test}.json`. The scripts under
[`preprocessing/`](preprocessing) are adapted from
[Coconut](https://github.com/facebookresearch/coconut) and pull the augmented GSM8K data
from [Internalize_CoT_Step_by_Step](https://github.com/da03/Internalize_CoT_Step_by_Step).

## Configuration

Each run is configured by a YAML file passed to `scripts/run.py` (via `CONFIG`). Configs compose
through a `base:` chain under [`args/base/`](args/base) (shared defaults → model → method
→ dataset), so each leaf config in [`args/`](args) only overrides what it needs, and every
field is documented inline in the YAML. The essentials:

- `looped` — `True` for LOTUS (looped latent training); `cot: True` for plain CoT fine-tuning.
- `load_model_path` — checkpoint (HF repo id or local path) to initialize LOTUS from a CoT-tuned model, or to evaluate.
- `c_thought` / `max_latent_stage` / `epochs_per_stage` — latent-token count and curriculum schedule.

## Models

Trained checkpoints are on the Hugging Face Hub — collected in the
[**LOTUS collection**](https://huggingface.co/collections/yingfanbot/looped-padded-6a552f7ef667cb41db2431a3)
(paper + models):

| Model | Description | GSM8K (GSM8k-Aug) |
| --- | --- | --- |
| [`yingfanbot/gsm-lotus-llama3b`](https://huggingface.co/yingfanbot/gsm-lotus-llama3b) | LOTUS, Llama-3.2-3B | 70.05% |
| [`yingfanbot/gsm-lotus-llama3b-codi`](https://huggingface.co/yingfanbot/gsm-lotus-llama3b-codi) | LOTUS + CODI, Llama-3.2-3B | 70.58% |

Stage-1 CoT initialization checkpoints:
[`gsm-cot-gpt2`](https://huggingface.co/yingfanbot/gsm-cot-gpt2),
[`gsm-cot-llama1b`](https://huggingface.co/yingfanbot/gsm-cot-llama1b),
[`gsm-cot-llama3b`](https://huggingface.co/yingfanbot/gsm-cot-llama3b).

## Training

Use `launch_train.sh` for distributed training. Important environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `CONFIG` | `args/gsm8k_cot_llama1b.yaml` | YAML config passed to `scripts/run.py` |
| `RUN_NAME` | empty | Optional run-name override |
| `NPROC_PER_NODE` | `4` | GPUs per node; set to `128 / batch_size_training` so the overall batch is 128 (GPT-2: 2, Llama-1B: 4, Llama-3B: 8) |
| `MASTER_PORT` | `29500` | Torch distributed master port |

LOTUS is trained in **two stages**: (1) CoT supervised fine-tuning, then (2) looped
latent training initialized from the Stage-1 checkpoint. Setting `RUN_NAME` controls
the output directory (`./outputs/<RUN_NAME>/`), which keeps the wiring between the two
stages predictable.

> By default, the LOTUS configs initialize from the published CoT checkpoints on Hugging Face
> (`yingfanbot/gsm-cot-{gpt2,llama1b,llama3b}`), so Stage 2 can be run directly without first
> running Stage 1. To initialize from your own CoT checkpoint instead, set `load_model_path`
> (and `teacher_model_path`, if CODI distillation is enabled) to
> `./outputs/<stage-1 RUN_NAME>/checkpoint_final`.

**GPT-2**

```bash
# Stage 2 — LOTUS (loads the published GPT-2 CoT checkpoint by default)
CONFIG=args/gsm8k_lotus_gpt2.yaml RUN_NAME=gsm-lotus-gpt2 NPROC_PER_NODE=2 ./launch_train.sh

# (optional) retrain the CoT stage yourself, then point load_model_path at the output:
# CONFIG=args/gsm8k_cot_gpt2.yaml RUN_NAME=gsm-cot-gpt2 NPROC_PER_NODE=2 ./launch_train.sh
#   -> ./outputs/gsm-cot-gpt2/checkpoint_final
```

**Llama-3.2-1B**

```bash
# Stage 2 — LOTUS (loads the published Llama-1B CoT checkpoint by default)
CONFIG=args/gsm8k_lotus_llama1b.yaml RUN_NAME=gsm-lotus-llama1b NPROC_PER_NODE=4 ./launch_train.sh

# (optional) retrain the CoT stage yourself, then point load_model_path at the output:
# CONFIG=args/gsm8k_cot_llama1b.yaml RUN_NAME=gsm-cot-llama1b NPROC_PER_NODE=4 ./launch_train.sh
#   -> ./outputs/gsm-cot-llama1b/checkpoint_final
```

**Llama-3.2-3B**

```bash
# Stage 2 — LOTUS (loads the published Llama-3B CoT checkpoint by default)
CONFIG=args/gsm8k_lotus_llama3b.yaml RUN_NAME=gsm-lotus-llama3b NPROC_PER_NODE=8 ./launch_train.sh

# (optional) retrain the CoT stage yourself, then point load_model_path (+ teacher_model_path) at the output:
# CONFIG=args/gsm8k_cot_llama3b.yaml RUN_NAME=gsm-cot-llama3b NPROC_PER_NODE=4 ./launch_train.sh
#   -> ./outputs/gsm-cot-llama3b/checkpoint_final
```

## Evaluation

`scripts/eval.py` is the standalone evaluation script (single GPU). It can load weights two ways:

- **Local checkpoint** — pass `--checkpoint ./outputs/<run>/checkpoint_final` (a torch state dict
  produced by training) together with the matching `--model_id` base model.
- **Published HF model** — pass `--model_id <hf-repo>` and omit `--checkpoint`; the trained weights load
  straight from the repo's `safetensors`. `from_pretrained` loads the weights only — the looped padded
  architecture is provided by the `Lotus` wrapper (`eval.py` runs it), so no separate checkpoint file is needed.

**LOTUS model on GSM8K (local checkpoint):**

```bash
python scripts/eval.py \
  --checkpoint ./outputs/gsm-lotus-llama3b/checkpoint_final \
  --model_id meta-llama/Llama-3.2-3B-Instruct \
  --datasets gsm8k \
  --n_looped_iters 6 --c_thought 25 --bf16
```

**LOTUS model on GSM8K (published HF model — no checkpoint file):**

```bash
python scripts/eval.py \
  --model_id yingfanbot/gsm-lotus-llama3b \
  --datasets gsm8k \
  --n_looped_iters 6 --c_thought 25 --bf16
```

**LOTUS model on out-of-distribution sets** (GSM-Hard / MultiArith / SVAMP):

```bash
python scripts/eval.py \
  --checkpoint ./outputs/gsm-lotus-llama3b/checkpoint_final \
  --datasets gsm-hard multi-arith svamp \
  --n_looped_iters 6 --c_thought 25 --bf16 \
  --save_preds preds.json
```

**Plain CoT model** (add `--cot`):

```bash
python scripts/eval.py \
  --checkpoint ./outputs/gsm-cot-llama3b/checkpoint_final \
  --datasets gsm8k --cot --bf16
```

Match `--n_looped_iters` / `--c_thought` to how the checkpoint was trained — use `--c_thought 13`
for GPT-2 checkpoints and `--c_thought 25` for Llama-1B/3B (and set `--model_id` to the matching
base model).

## Citation

If you find it useful, please cite:

```bibtex
@article{fan2026bridging,
  title={Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers},
  author={Fan, Ying and Svete, Anej and Lee, Kangwook},
  journal={arXiv preprint arXiv:2606.31779},
  year={2026}
}
```

## License

Released under the MIT License (see [`LICENSE`](LICENSE)).

## Acknowledgements

The training, evaluation, and data-preprocessing code is adapted from
[Coconut](https://github.com/facebookresearch/coconut)
(*Training Large Language Models to Reason in a Continuous Latent Space*).
