# Benchmark: reproducible data and a fixed evaluation harness

Two scripts are all most work needs:

- `scripts/setup_data.py` downloads every dataset at a pinned revision, normalizes and tags it, writes
  fixed splits, and checks each file's SHA-256 against `provenance/bench_data_lock.json`. Identical
  bytes on every machine, or it stops.
- `scripts/evaluate.py` runs any model over fixed datasets, saves every output, and prints the model
  against the persistent leaderboard in `results/`. Finished results are reused, never silently mixed.

Everything runs on CUDA, Apple MPS and CPU (`--device auto` picks CUDA, then MPS, then CPU).

## Setup

Local (macOS or Linux, Python 3.12):

```bash
git clone https://github.com/sttruji/dynamic-state-anchoring.git
cd dynamic-state-anchoring
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dsa.lock.txt
.venv/bin/python scripts/setup_data.py          # ~2 min, ~1 GB under data/
.venv/bin/python scripts/evaluate.py --model gpt2-sft --suite dev
```

Colab: open [`notebooks/Benchmark_Colab.ipynb`](../notebooks/Benchmark_Colab.ipynb) and run its cells.
It clones, builds the pinned environment, keeps data, model downloads and results on Drive, and
evaluates. An interrupted evaluation resumes from its saved predictions.

`setup_data.py --eval-only` skips the large training corpora. `--verify` re-checks existing files
without downloading. `$DSA_DATA_DIR` moves the data directory (default `<repo>/data`, git-ignored).

## Data

| File (under `data/bench/`) | Rows | Source (pinned) | Use |
| --- | ---: | --- | --- |
| `train/gsm8k_aug.jsonl` | 385,620 | LOTUS mirror of GSM8K-Aug, `da03/Internalize_CoT_Step_by_Step@e06a32ee` | DSA training corpus |
| `train/gsm8k_aug_nl.jsonl` | 384,618 | `zen-E/GSM8k-Aug-NL@e20551f8` | NL training corpus |
| `train/csqa_cot.jsonl` | 8,196 | `zen-E/CommonsenseQA-GPT4omini@742d45af` | sanity-check corpus |
| `train/overfit32.jsonl` | 32 | reviewed fixture (below) | overfitting / mechanism checks |
| `eval/overfit32.jsonl` | 32 | same questions, eval view | does the model fit what it trained on? |
| `eval/dev64.jsonl` | 64 | first 64 GSM8K-Aug validation questions | fast development set |
| `eval/gsm8k_valid.jsonl` | 500 | GSM8K-Aug validation | model selection |
| `eval/gsm8k_test.jsonl` | 1,319 | official GSM8K test | in-distribution test |
| `eval/svamp.jsonl` | 1,000 | `ChilleD/SVAMP@5e0bf1e5`, train+test, `Body + " " + Question` | OOD test |
| `eval/gsm_hard.jsonl` | 1,319 | `reasoning-machines/gsm-hard@960448f7` | OOD test |
| `eval/multiarith.jsonl` | 180 | `ChilleD/MultiArith@144d44c3`, test | OOD test |
| `eval/csqa.jsonl` | 1,221 | CommonsenseQA-CoT validation | multiple-choice test |

The OOD conventions match the CODI and LOTUS evaluation scripts. `data/gsm_{train,valid,test}.json`
are also written, so the existing `scripts/prepare_dsa_pilot.py --source-dir data` keeps working.

**Tags.** Every training row carries deterministic `<SA>` milestone tags from the existing
annotator (`dsa/data/milestones.py`, `conclusive` policy, terminal answer excluded):
`rationale_marked`, `event_count` and `events` (character offsets into `rationale`). They do not
depend on the tokenizer. Loop schedules depend on K and the tokenizer, so they are computed at
preparation time. GSM8K-Aug gets 613,547 events on 83.5% of rows, the same as the October corpus
audit. GSM8K-Aug-NL and CommonsenseQA get **zero** events: the annotator recognizes `<<a=b>>`
calculator spans and whole-line equations, not arithmetic written inline in prose.

**overfit32** is the reviewed 32-question TRAIN fixture from
[`provenance/mps_experiment_plan_20261019.json`](../provenance/mps_experiment_plan_20261019.json):
29 retained pilot questions plus replacements for 207,002 / 288,908 / 305,659, with the
`<<120=120>>` given-restatement event removed from row 112,564. Construction asserts its reviewed
identities and its counts (32 records, 58 events, 3 no-event controls). To train DSA on it:
`scripts/prepare_dsa_pilot.py --source-dir data --fixture overfit32 --loops 8 --output-dir ...`.
`--steps-join newline` writes the explicit training text one step per line, the format the CoT
checkpoint was trained on; the default `space` keeps the original DSA contract.
[`notebooks/DSA_Format_Test_Colab.ipynb`](../notebooks/DSA_Format_Test_Colab.ipynb) runs that
comparison on the data-scale learned arm.

**Contamination.** Setup checks every evaluation question against every training corpus after
normalization; the lock records zero overlaps for all sets.

## Evaluate

```bash
python scripts/evaluate.py --model gpt2-sft --suite overfit,dev,valid
python scripts/evaluate.py --model codi --suite dev
python scripts/evaluate.py --model dsa --arg checkpoint=runs/learned --arg loops=16 --name dsa-learned-k16 --suite dev
python scripts/evaluate.py --model dsa --arg checkpoint=runs/learned --arg route=explicit --name dsa-learned-explicit --suite dev
python scripts/evaluate.py --model gpt2-sft --suite test --test    # final numbers only
python scripts/evaluate.py --model gpt2-sft --suite dev --limit 4  # smoke check, never saved
python scripts/evaluate.py --table
```

Suites: `overfit` (overfit32), `dev` (dev64), `valid` (gsm8k_valid), `ood` (svamp, gsm_hard,
multiarith), `test` (gsm8k_test + ood), `csqa`. Held-out sets need `--test`.

Two accuracies are reported. `acc` keeps the pinned LOTUS extraction (text after the last `###`,
otherwise the last number; float equality), so new rows compare directly with every earlier DSA
number. `strict` also requires one clean scalar after the answer marker, which removes the
incidental credit noted in FINDINGS (`iced.5` or `10*5` scored as 5). Non-integer gold answers
(GSM-Hard) are compared with a 1e-4 tolerance.

**Why results stay comparable.** Each result is keyed by its row name and dataset, and stores the
model's identity (checkpoint hash, loop budget, decoding settings), the dataset hash and the scorer
version. Re-running an identical configuration reuses the saved result. A different configuration
under the same name is refused: pick a new `--name`, or pass `--overwrite`. Results from an older
scorer or dataset version are left out of the table and listed as stale.

## Baselines

| `--model` | Weights | Inference | Verified |
| --- | --- | --- | --- |
| `gpt2-vanilla` | `gpt2@607a30d7` | question + `\n`, greedy, 192 tokens | — (floor) |
| `gpt2-sft` | `yingfanbot/gsm-cot-gpt2@e0ef1887` (DSA's initialization), or `checkpoint=<dir>` | same | 25/64 on dev64, the recorded FINDINGS value |
| `codi` | `zen-E/CODI-gpt2@fd641b3d`, LoRA merged in FP32 | question + `<bot>`, 6 latent steps, `<eot>`, greedy, 256 tokens | outputs identical to official CODI code (peft) on 32/32 questions |
| `lotus` | **needs `checkpoint=`** (no public GPT-2 LOTUS weights) | question + `\n` + 78 latents, 6 loops, greedy, 64 tokens | runs on MPS through the upstream `Lotus` class |
| `dsa` | `checkpoint=<model, checkpoint or run dir>`; optional `loops=`, `write_mode=` | question-only latent loop, 64 tokens | all four data-scale K=8 checkpoints reproduce their recorded dev64 scores |
| `dsa`, `route=explicit` | same checkpoints | the trained backbone's ordinary CoT route: question + `\n`, greedy, 192 tokens | reproduces all 256 explicit-control outputs from FINDINGS (four arms × 64) |

`gpt2-vanilla` and `gpt2-sft` share prompt and decoding, so the only difference is the weights.
CODI ran in bf16 upstream; FP32 here can differ slightly from the paper.

**LOTUS weights.** The `yingfanbot` hub account publishes only Llama-3B LOTUS models.
`yingfanbot/gsm-lotus-gpt2`, named in the CoT model card, returns 401 (private or missing).
Either ask the authors for it or train one with `scripts/run.py args/gsm8k_lotus_gpt2.yaml`, then
pass `--arg checkpoint=<checkpoint_final>`. The paper reports 44.1 ± 0.7 on GSM8K for GPT-2 LOTUS
(three seeds), against 42.7 for explicit CoT and 42.0 for CODI.

Retraining is not cheap. The recipe is 30 epochs over all 385,620 rows at global batch 128, with
one curriculum stage per epoch up to stage 6 (78 latents, 6 loops), so a stage-6 sample pushes
about 616 tokens through GPT-2 against 113 for plain CoT: about 4.7e18 training FLOP in fp32.

- **This M1 (measured 25 Oct, MPS):** 2.4 samples/s at stage 0 (batch 16), 0.3–0.4 samples/s at
  stage 6 (batch 4, the memory limit). About 44 h for the first epoch and 250–400 h per stage-6
  epoch: the full recipe would take most of a year. Not an option.
- **One CUDA GPU (estimated, not run):** `torchrun --nproc_per_node=1` should run unchanged, but
  needs `datasets` and `wandb` installed (or `WANDB_MODE=disabled`), gradient accumulation to
  reach batch 128 (the per-GPU batch of 64 needs about 80 GB at stage 6), and
  `allow_resume_from_checkpoint: True` so a disconnect resumes from the last epoch. About 26–43 h
  on an A100 with TF32 matmuls enabled, 110–165 h without; much longer on an L4 or T4.
- The trainer picks its best epoch on `gsm_valid`, which is the leaderboard's `gsm8k_valid`;
  LOTUS's validation column is then optimistic. Test sets are unaffected.

A reduced budget (fewer rows or epochs) is a weaker baseline than the paper's and must not be
reported as LOTUS.

**Explicit-only control.** `scripts/train_sft.py --train overfit32 --output runs/sft-overfit32`
fine-tunes the CoT checkpoint with the DSA pilot's sample order, optimizer and clipping, using
LOTUS's CoT text format (one step per line). Evaluate it with `--model gpt2-sft --arg checkpoint=...`.

## Add a model

Write a subclass anywhere; the harness never changes:

```python
from dsa.bench.models import BenchModel

class MyModel(BenchModel):
    name = "my-model"
    batch_size = 8

    def __init__(self, checkpoint, device="auto"):
        ...  # load weights; resolve the device with dsa.devices.resolve_device(device)

    def identity(self):  # anything that could change outputs
        return {"checkpoint_sha256": ..., "max_new_tokens": 64}

    def generate(self, questions):  # question strings in, raw text out
        return [...]
```

```bash
python scripts/evaluate.py --model my_package.my_module:MyModel --arg checkpoint=... --suite dev
```

`generate` receives question text only, never gold answers, rationales or tags.

## Sharing results

Commit `results/` with your change: `runs/<model>/<dataset>.json` is the source of truth,
`predictions/` keeps every output for paired comparisons, and `leaderboard.csv` is rebuilt from
`runs/` after every evaluation. Each result is its own file, so teammates' additions merge cleanly.
If `leaderboard.csv` conflicts, run `python scripts/evaluate.py --table` to rebuild it. On Colab,
copy the Drive `results/` folder into your local clone before committing.
