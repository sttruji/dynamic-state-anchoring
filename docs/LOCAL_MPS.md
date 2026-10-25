# Local GPT-2 experiments on Apple MPS

The trainer, unmodified-checkpoint smoke and answer evaluator support `--device
mps`. Their `auto` setting selects CUDA, then MPS, then CPU. Explicitly requesting
an unavailable accelerator fails instead of silently falling back to CPU.
The existing `.venv` supplies the project dependencies.

Verified on 18 October 2026 with an Apple M1 and 16 GiB unified memory:
the 125,037,313-parameter GPT-2 DSA model completed four K=8 updates for each
of the four policies, from identical initialization. The full check took
95.8 seconds; per-policy median update times excluding the first update ranged
from 3.86 to 5.78 seconds. The largest observed Metal-driver allocation was
5.06 GiB. All losses, gradients and final parameters were finite, but every
update required gradient clipping. No validation or held-out test performance
was measured. See [the validation record](../provenance/local_mps_validation.json).

Native execution is required for Metal access. The Codex filesystem sandbox can
report MPS unavailable even when it works in a native terminal; GPU checks in
Codex may therefore need an approved execution outside that sandbox.

## Prepare a separate local experiment

Run from this repository's root. The cache stays inside ignored `artifacts/`.
Disabling Xet uses the standard HTTP model downloader; no model revision changes.

```bash
export HF_HOME="$PWD/artifacts/hf-cache"
export HF_HUB_DISABLE_XET=1

.venv/bin/python scripts/smoke_pretrained_dsa.py \
  --device mps \
  --output artifacts/local-mps-20261018/unmodified-smoke.json \
  --max-new-tokens 4

.venv/bin/python scripts/prepare_dsa_pilot.py \
  --source-dir data \
  --output-dir artifacts/local-mps-20261018/prepared \
  --train-limit 32 --loops 8 --seed 7
```

These commands retain the pinned pretrained checkpoint, original split
membership, annotation policy and fixed question-only inference interface.
Prepared data and smoke proofs are checked by the same training contract as
the Colab runs. Use a new output directory when changing the experiment identity.

## Check actual optimizer cost before committing to a run

```bash
.venv/bin/python scripts/check_local_dsa.py \
  --prepared-dir artifacts/local-mps-20261018/prepared \
  --smoke-report artifacts/local-mps-20261018/unmodified-smoke.json \
  --output-dir artifacts/local-mps-20261018/optimizer-check \
  --device mps --recurrence-mode flt_memory --steps 4
```

The check performs four matched write policies with identical initialization,
sample order, FP32 arithmetic, AdamW settings and loss weights. It journals
individual losses, gate probabilities, write counts, unclipped gradient norms,
synchronized update times and MPS memory snapshots. It saves no full model or
Adam checkpoints, avoiding several gigabytes of duplicate checkpoint files for
a feasibility check. It never evaluates validation or test answers.

This is a bounded device/optimization check, not evidence of improved reasoning.
The first update includes optimizer allocation and kernel warm-up; both overall
and subsequent-update medians are reported. MPS allocation snapshots and their
observed maxima are not peak-memory measurements or available/free system RAM.

## Run resumable training and validation

For a scientifically specified experiment, retain the existing full trainer:

```bash
.venv/bin/python scripts/train_dsa.py \
  --prepared-dir artifacts/local-mps-20261018/prepared \
  --smoke-report artifacts/local-mps-20261018/unmodified-smoke.json \
  --output-dir artifacts/local-mps-20261018/runs/learned \
  --device mps --recurrence-mode flt_memory --write-mode learned \
  --loops 8 --steps 200 --save-every 25

.venv/bin/python scripts/evaluate_dsa.py \
  --prepared-dir artifacts/local-mps-20261018/prepared \
  --run-dir artifacts/local-mps-20261018/runs/learned \
  --output-dir artifacts/local-mps-20261018/validation/learned \
  --device mps --split valid --max-new-tokens 64
```

Use separate run directories for `static`, `every_loop`, `periodic` and `learned`;
hold all other conditions fixed. Full training retains two complete checkpoints,
including Adam and RNG states, and needs several gigabytes of disk per arm.
The explicit request `--final-test` remains required for complete test evaluation.

Checkpoint RNG includes MPS state. A hardware-gated tiny integration test checks
FLT training, interrupted/resumed parameter agreement within tolerance, and
question-only validation. It does not establish bitwise reproducibility for
pretrained MPS runs or across CPU, MPS and CUDA backends.

```bash
.venv/bin/python -m unittest tests.test_devices -v
```

## Refinement priorities

Keep the prior Colab results documented in `FINDINGS.md`. FLT improved observed
conditioning, but the learned write controller collapsed and explicit-route
accuracy declined in the previous experiments. Before increasing scale, name one
mechanistic change, compare it against the unchanged controls, and check gate
learning and retained explicit capability together with answer accuracy.

The local feasibility check and standard answer evaluator require no
intermediate-state probing. The older comprehensive diagnostic script uses
float64 reductions and has not been ported or verified on MPS.

The [19 October checkpoint-label audit](CHECKPOINT_LABEL_AUDIT_20261019.md)
reconstructs all 512 historical training rows and records personal semantic
judgments on 76 questions, including the complete local pilot. The
[next controlled MPS experiment design](MPS_CONTROLLED_EXPERIMENTS_20261019.md)
freezes a proposed reviewed fixture and specifies matched write-policy and
backbone-warm-up comparisons. It is a design, not an executed experiment; its
fixture builder and warm-up support still require implementation.
