# FLT data-scale experiment v1

Open [DSA_FLT_DataScale.ipynb in Colab](https://colab.research.google.com/github/sttruji/dynamic-state-anchoring/blob/main/notebooks/DSA_FLT_DataScale.ipynb), select **T4 GPU**, and **Run all**. The first cell clones GitHub; subsequent cells install requirements, mount Drive, reuse verified corpus/model caches, prepare data, train, evaluate and download the review ZIP. The original [32-record pilot notebook](../notebooks/DSA_Colab.ipynb) remains available separately.

## Why this experiment

The completed K=8 pilot used 32 training records and 200 updates per arm. Standard/static and standard/learned fit 6/32 and 4/32 training answers; their FLT-inspired counterparts fit 29/32 and 31/32. FLT reduced median training gradient norms approximately 88-fold and 279-fold. Its validation results remained poor (3/128 and 2/128), and learned write-event discrimination was near chance (AUC 0.507). Forcing all writes preserved the eight FLT/learned intervention answers. These observations motivate testing more data and simple write policies, without increasing model size or changing the objective.

## Frozen defaults

| Setting | Value |
|---|---|
| Backbone | `yingfanbot/gsm-cot-gpt2`, revision `e0ef1887404adbf6c531d2cac7804694321f87bd` |
| Dataset | Original LOTUS augmented GSM8K splits, source revision `e06a32ee5e4cd117171daeb4755d2a97ece62761` |
| Training | 512 records selected without replacement using seed 7, restored to source order; 1,024 single-record updates per arm (two passes) |
| Latent computation | K=8, four workspace vectors, one anchor, FLT-inspired memory-query injection in every arm |
| Write policies | No writes (`static`); every loop; periodic at loops 4/8; learned hard threshold 0.5 with straight-through training |
| Optimization | AdamW, learning rate 3e-5, weight decay 0.01, gradient norm clipping 1, FP32, activation checkpointing |
| Losses | Answer CE + explicit-CoT CE + 0.1 gate BCE + 0.1 candidate alignment; positive gate weight 1 |
| Validation | All 500 original questions, K=8, greedy decoding, 64-token cap, final checkpoint at update 1,024 |
| Test | Disabled; reserved for a later frozen protocol |
| Training diagnostics | 64 records sampled with seed 7 and restored source order; same IDs across policies |
| Interventions | First eight diagnostic records; K=1/4/8, no/every writes, zero final anchor/workspace/both |
| Gradient diagnostics | First two diagnostic records; K=1/4/8 answer probes and joint loss contributions at K=8 |
| Explicit controls | Original cached checkpoint plus four trained backbones; eight training and eight validation records each, 192-token cap |

All policies contain the same modules and receive the same gate/candidate supervision, including policies whose write decision is fixed. Only the declared write policy changes. The ordinary explicit teacher route shares the backbone and supplies detached pre-marker milestone targets; no separate teacher training or CODI answer-boundary objective is added. Gold answers and write labels do not enter question-only latent inference.

The 512 selected records may include augmented variants of the same question; the notebook separately prints the unique question count. No-event and overflow records retain main losses, with the existing auxiliary masking policy. Source quality exclusions remain off. Training and validation membership are checked by the established preparation contract.

Training settings and implementation commit identify a new experiment directory. Changing diagnostic settings creates a separate review identity; completed checkpoint bytes further bind the output directory. Editing training settings before execution creates a different experiment. Record `REPO_COMMIT`; set the clone cell's `REPO_REF` to that SHA to reproduce a version.

## Runtime and persistence

The previous T4 profile suggests approximately two hours for the 4,096 training updates alone. Loading, validation, diagnostics, checkpoint archival and browser transfers add time, and different sequence lengths can change throughput. This is a budget estimate rather than a promised runtime. The training and diagnostic limits are exposed together in the configuration cell.

Training writes to VM disk, retaining two local periodic checkpoints. Every finished arm receives a complete archive containing model, tokenizer, optimizer/RNG state, manifests and metrics. The archive is copied to Drive, checked by mounted readback and requested as an independent browser download before the next arm. Keep the four full ZIPs and let transfers finish. Mounted readback does not establish current cloud durability.

On reconnect, run setup and the workflow again: finished compatible arms are restored/reused, while missing or partial arms restart. There is no partial-checkpoint repair or automatic continuation into a larger budget. `RUN_MODE="restart_all"` explicitly retrains every arm. A restored/reused arm must pass settings, runtime, data, initialization and actual weight checks.

## Evidence and interpretation

The final small review ZIP contains all 2,000 validation predictions, full per-update journals, sampled training outputs and teacher-forced profiles, initialization probes, controller discrimination, paired policy comparisons, intervention outcomes, explicit controls, runtime reports, source/configuration hashes and stage failures. Full weights stay in separate checkpoint ZIPs. The archive excludes held-out test data.

Training diagnostics explicitly identify the full training denominator, selected IDs and sampled denominator. They are not accuracy on all 512 records. The prior 80% memorization heuristic is disabled for this data-scale experiment; low training accuracy is a result to interpret, not missing engineering evidence. Exact paired validation comparisons are exploratory, unadjusted and based on one seed.

The main questions are whether validation improves with more training data and whether learned writes outperform no writes, every-loop writes and periodic writes. Compare actual validation differences, validity, timing, controller event discrimination and gradient behavior. Good training accuracy, low gradient norms, an AUC proxy, or memory sensitivity alone does not demonstrate improved reasoning. The small explicit control uses a longer decoding budget and does not establish equal-compute superiority. Repeated seeds and stronger controls are required before a research claim.

The architecture is inspired by Fu et al.'s [Simply Stabilizing the Loop via Fully Looped Transformer](https://arxiv.org/html/2605.18797v1), also listed as reference [13] in the [research handoff](DSA_IMPLEMENTATION_HANDOFF.md#11-references-and-provenance). Our adaptation persists only the five memory vectors and changes their query path in GPT-2; the paper studies a different architecture and training setup. See [K8_FLT_PILOT.md](K8_FLT_PILOT.md) for implementation details.
