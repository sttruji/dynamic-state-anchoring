# K=8 pilot with FLT-inspired memory injection

Open the current [Colab notebook](https://colab.research.google.com/github/sttruji/dynamic-state-anchoring/blob/main/notebooks/DSA_Colab.ipynb), choose **T4 GPU**, and **Run all**. A saved notebook copy does not update automatically. The fresh run is named `gpt2-k8-flt-v1-<code commit>`, separate from the completed K=16 pilot.

## Four matched conditions

| Notebook condition | Recurrence | Anchor write policy |
|---|---|---|
| `static` | Existing memory feedback | Never write |
| `learned` | Existing memory feedback | Learned hard gate with straight-through training |
| `flt_static` | FLT-inspired memory queries at every block | Never write |
| `flt_learned` | FLT-inspired memory queries at every block | Learned hard gate with straight-through training |

All four start from identical parameters of the pinned `yingfanbot/gsm-cot-gpt2` snapshot, including the registered marker and initialized memory/controller parameters. Each uses 32 questions, seed 7, 200 one-example updates, **K=8**, four workspace vectors and one anchor. FP32, activation checkpointing, AdamW, learning rate 3e-5, clipping at 1, and loss weights remain fixed. The same numerical initialization hash is enforced before optimizer updates. Runtime, prepared data, actual example sequences and parameter counts are recorded.

The verified source corpus and Hugging Face cache are reused from Drive. Preparation regenerates milestone schedules for K=8 using the same seeded raw-data selection. When the original K=16 prepared cache exists, the notebook checks the ordered 32 sample IDs against it. No source-quality exclusions are introduced. All selected questions remain in answer/CoT losses; the original auxiliary masking policy remains in effect. The selected maximum event count is five, so K=8 does not create a new overflow case.

## What the adaptation changes

This design is inspired by [Simply Stabilizing the Loop via Fully Looped Transformer](https://arxiv.org/html/2605.18797v1) and its [author implementation](https://github.com/FuRuF-11/FullyLoopedTransformer/blob/main/nanochat/gpt.py). It is a **fixed-memory adaptation**, not a reproduction of their full-sequence model or a guarantee of gradient stability.

The first latent pass is the same ordinary GPT-2 pass in both recurrences. On later FLT-inspired passes:

1. Rebuild the current residual stream from question embeddings and the original learned memory seeds, with the usual GPT-2 positional embeddings and dropout.
2. At every GPT-2 block, memory positions obtain queries from the previous loop's anchor/workspace through the existing `ln_1` and Q projection. Question positions keep ordinary current-layer queries.
3. Keys and values come from the current layer. Existing causal/padding masks, output projection, residual connections and feed-forward block are retained. Previous memory is never directly added into the current residual stream.
4. Apply the existing final normalization and workspace normalization; compute the unchanged candidate projection and anchor-write gate.

Only the five memory vectors recur. The question hidden-state sequence is not stored across loops. There are no added trainable parameters, extra cross-attention blocks, or detached loop states. The current residual stream resets, while gradients remain connected through the previous-memory queries and anchor updates. The explicit teacher and final answer decoder retain their ordinary GPT-2 computation and share the same weights.

The adapter supports the pinned eager **GPT-2 / Transformers 4.46.2** implementation. It deliberately does not replace pretrained GPT-2 normalization, activation functions, positional encoding or optimizer with the author's pretraining recipe. Projection reuse preserves parameter count; it does not imply identical execution speed. Actual time and GPU memory are reported.

## Bounded gradient screen and evaluation

Before any optimizer updates, the notebook screens one fixed training question at K=1/8/16 in all four conditions using identical weights and seeds. It saves answer-gradient norms, per-loop memory gradients, and residual norms **before normalization**, and checks that weights remain unchanged. Nonfinite results stop training and remain in the review package. A larger finite gradient is a recorded scientific result, not silently treated as an improvement or an automatic reason to discard a condition. The screen is descriptive and too small to prove stability.

Each trained condition then receives the existing comprehensive review:

- Question-only answer generation on all 32 training questions and 128 source-order validation questions.
- Eight fixed training questions under depth, no-write/always-write and final-memory-zeroing interventions.
- Deterministic losses, gate labels/decisions and candidate-teacher alignment on every training row.
- Four training rows with answer-gradient depth probes and weighted joint-loss contributions from the same graph.
- Per-block memory residual RMS and maximum magnitude before attention normalization, before FFN normalization and before final normalization. These are separate from the normalized workspace RMS.
- Explicit controls on eight training/eight validation questions for the original model and each trained backbone.

K=16 probes are extrapolation beyond the K=8 training budget. The held-out test remains disabled. Comparisons in `scale_readiness.json` separate recurrence changes at fixed write policy from policy changes within one recurrence. Stability, training-set learning, controller event separation and validation are assessed together; no larger run is approved automatically.

## Checkpoints and handoff

Training writes to VM disk. After **each completed arm**, the notebook creates one full ZIP, copies it to Drive, checks mounted readback and requests a browser download before the next arm. Keep the **four full checkpoint ZIPs** on your computer and allow multiple browser downloads if prompted. Each includes model/tokenizer, optimizer, scheduler, RNG, journals and a SHA256 inventory. Mounted readback cannot verify server-side durability, and a download request does not confirm transfer completion.

The last cell downloads **one small review ZIP** containing all four conditions, the initialization screen, diagnostics, comparisons, logs and completeness inventory. **Hand back that small review ZIP.** Full weights and test answers are excluded from it.

On reconnect, run setup again. `auto_pilot` reuses verified finished arms and restarts partial or incompatible arms from scratch. `restart_all` forces all four to retrain. The previous K=16 run is preserved. Its archived implementation can be reproduced by setting `REPO_REF` to its recorded commit; the new architecture code does not claim compatibility with its old implementation hashes.

## Verification boundary

Local CPU tests check block equivalence when queries agree, identical parameter initialization and first-loop outputs, causal padding, previous-memory connectivity at every block, ordinary teacher/readout paths, finite gradients, checkpoint recomputation with dropout, save/load, and the four-condition notebook workflow through screening, training, diagnostics and all exports. These tests do not execute pretrained training on a hosted T4 or verify real Drive/browser transfers. The new GPU run remains an experiment.
