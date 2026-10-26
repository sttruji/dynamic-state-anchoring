# CODI-style training options

`scripts/train_dsa.py` can now train DSA with the parts of CODI's GPT-2 recipe
(Shen et al., EMNLP 2025; `scripts/train_gpt2_gsm8k-aug.sh` and `src/model.py` in
[zhenyi4/codi](https://github.com/zhenyi4/codi)) that DSA shares. Every option
defaults to off. With the defaults the trainer reproduces earlier runs bit for bit
(see *Verification*).

| Option | Default | CODI GPT-2 | What it does |
| --- | --- | --- | --- |
| `--batch-size N` | 1 | 64 per GPU × 2 accumulation | Pads N records into each forward pass. Each loss is a per-record mean averaged over the batch, so N records in one batch train like N records accumulated one at a time, only faster. Updates see `batch-size × gradient-accumulation` records. |
| `--lora-rank R` | 0 (whole backbone) | 128 | LoRA on every `c_attn`, `c_proj` and `c_fc`, including the FLT memory queries. The rest of GPT-2 is frozen except the new `<SA>` embedding row. `--lora-alpha` (32) and `--lora-dropout` (0.1) match CODI. |
| `--new-module-learning-rate` | `--learning-rate` | (one rate) | Learning rate for the anchor, workspace, controller and anchor projection, and for the `<SA>` row under LoRA. |
| `--lr-schedule cosine`, `--warmup-steps W` | constant, 0 | cosine, 3% warmup | Hugging Face's warmup and cosine schedules. Cosine binds `--steps`, so such a run cannot be extended. |
| `--answer-distill-weight λ` | 0 | 1 | CODI's answer-position distillation (below). |

## Answer-position distillation

CODI matches the student's hidden states to the explicit teacher's at the token
just before the answer, at every layer: smooth L1 per layer divided by the teacher
state's standard deviation, averaged over layers, with the teacher detached. DSA
adds this to its existing `<SA>` alignment rather than replacing it:

```
L = L_answer + 1.0·L_CoT + 0.1·L_gate + 0.1·L_anchor + λ·L_distill
```

The student's states come from the answer readout pass (question, anchor,
workspace, then `\n### `). The explicit text is tokenized once, so GPT-2 merges
the prefix's trailing space into the answer (`###`, ` 18`), while the student's
fixed prefix keeps it separate (`###`, ` `, `18`). The loss therefore pairs the
teacher's token just before the answer with the same token in the student's
prefix. On all 512 records of both prepared data-512 sets, that is `###` with `###`.
`dsa.data.tokenize.answer_alignment` finds the pair and refuses records whose
explicit text does not end in the prefix and answer.

Differences from CODI: the standard deviation is taken per record, not over the
batch, so batching does not change the loss. Training stays FP32, where CODI
used bf16.

## Example: CODI's settings on a GPU

```bash
python scripts/train_dsa.py --prepared-dir PREPARED --output-dir RUN \
  --smoke-report SMOKE.json --device cuda \
  --recurrence-mode flt_memory --write-mode static --loops 8 --workspace 4 \
  --steps STEPS --warmup-steps WARMUP --lr-schedule cosine \
  --batch-size 32 --gradient-accumulation 4 \
  --lora-rank 128 --lora-alpha 32 --lora-dropout 0.1 \
  --learning-rate 3e-3 --weight-decay 0.1 --max-grad-norm 2 \
  --answer-distill-weight 1
```

`32 × 4` matches CODI's 128 records per update. The batch size that fits depends
on the GPU and has not been measured here. Raise `--batch-size` and lower
`--gradient-accumulation` while memory allows. The 512-record data-scale set is
far too small for this recipe: CODI trains 40 epochs over 385,620 records.
Prepare a larger set with `prepare_dsa_pilot.py --train-limit 0`, which selects
the complete split.

LoRA checkpoints save merged GPT-2 weights in `model/`, so `scripts/evaluate.py
--model dsa` and every other loader read them unchanged. `training_state.pt` also
keeps the adapters. Resuming rebuilds the seeded initial model, checks it against
the run's recorded initial-state hash and restores the adapters, so resumed runs
match uninterrupted ones exactly on CPU.

## Verification

- Defaults are unchanged. A tiny-model run produced identical metrics and final
  weights from the previous code and this code, for standard and FLT recurrence
  with learned and periodic writes (6 updates, accumulation 2). On real GPT-2,
  FLT K=8 with dropout on, one two-record update produced bit-identical losses
  and gradients.
- One padded batch of 4 real GPT-2 records matched accumulating the same 4, with
  distillation on and dropout off: losses agreed to 6 digits and gradients to
  within 1.2e-5 of their largest entry. The batch took 5.5 s against 12.5 s on
  CPU. GPU speedups have not been measured.
- A 3-update CPU run on real GPT-2 completed with LoRA rank 128, batch 4,
  distillation, cosine schedule and new-module rate, at 2.3 GB peak RSS. The
  harness loaded its checkpoint for both routes. The merged weights matched the
  adapter model to within 3e-4 in the answer logits.
- `tests/test_codi_recipe.py` covers batch/accumulation equivalence, alignment,
  distillation, adapter initialization and freezing, FLT memory queries under
  LoRA, merged saving, parameter groups, exact LoRA resume and schedules.
