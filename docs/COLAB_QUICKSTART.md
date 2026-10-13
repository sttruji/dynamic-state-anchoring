# One Colab notebook for the pilot and its review

Open [DSA_Colab.ipynb in Colab](https://colab.research.google.com/github/sttruji/dynamic-state-anchoring/blob/main/notebooks/DSA_Colab.ipynb),
select **T4 GPU**, and run all cells in order. This is the original notebook,
expanded to collect the evidence needed before choosing a larger experiment.
A saved Drive copy does not update automatically; open the current GitHub version.

## Run the automatic small pilot

The default `RUN_MODE="auto_pilot"` selects `gpt2-pilot-v1-ba4e69315086`.
It checks actual model, tokenizer and training-state files at the **200-step target**.
A complete compatible arm is reused. A missing, corrupt, partial or incompatible
arm restarts from the original pretrained model for 200 updates. It does not
repair or resume partial checkpoints. The historical Drive folders are preserved.
The expected cost of restarting one arm is roughly the previous 8-minute training
measurement, plus model loading, checkpoint writing and backup transfer.

Both arms must match initialization, prepared data, sample order, settings and
training runtime. If the current GPU/packages differ from a saved arm, it can
retrain that arm too so the comparison remains matched. Initialization is checked
before optimizer updates on the second arm. Set `RUN_MODE="restart_all"` if you
prefer both arms to start fresh without inspecting old checkpoints.

Active checkpoints live at `/content/dsa-pilot-working/<run>/<arm>` on the VM.
After each completed arm, a full ZIP is written locally, copied as **one file**
to Drive and checked by mounted readback. After training, both arm ZIPs are
combined into an independent browser download, **before diagnostics**. Keep that
file on your computer and wait for its download to finish before disconnecting.
Mounted readback cannot verify server-side durability.

On a fresh VM, run setup and Drive mounting again. Cached corpus/model downloads
and checksum-selected prepared data are reused if available; missing inputs are
rebuilt. The runner tries completed local/Drive archives and historical files,
then restarts any unusable arm. Report folders include settings, code and actual
checkpoint identities so predictions from a different restart cannot be mixed in.
Completed diagnostic cases can be reused within the same identity. No larger
training run starts automatically.

## Evidence collected in one run

| Question | Evidence |
|---|---|
| Did both arms receive matched training? | Initialization/data/order/runtime/settings identities and actual per-update sample sequences |
| Did the models learn their training examples? | Question-only final-answer generation on all 32 training questions; deterministic teacher-forced losses on every training row |
| Does more latent depth change answers? | Same saved weights, same first eight training questions, K=1/2/4/8/16 |
| Do anchor writes and final memory matter? | Paired no-write, every-write, zero-anchor, zero-workspace and zero-memory inference interventions |
| Does the controller follow its labels? | Training gate targets, probabilities, decisions, precision/recall and positive/write rates |
| Do latent states settle or destabilize? | Per-loop anchor/workspace scales, changes and cosine similarities; comparison to question-embedding scale |
| Are teacher milestones being aligned? | Scheduled candidate/teacher normalized MSE, cosine and state norms |
| Where do large gradients arise? | Answer-only depth probes on four fixed training rows; weighted loss contributions at the training budget; actual per-loop input/output gradients |
| Do losses compete at the controller? | Gate-head and candidate-projection gradient direction cosines between loss components |
| Is the pretrained explicit reasoning route capable? | Original cached model versus each arm's trained shared backbone on the same eight train/eight validation questions |
| Does performance extend beyond the training set? | First 128 original validation questions, paired scores and raw outputs; use `VALID_LIMIT=0` for all 500 |
| Did generation end normally? | Raw generated token IDs, token counts and EOS/token-cap stop reasons in training/control diagnostics |
| What does the run cost and fit? | Original per-update timing, allocated GPU memory and loss/gradient journals; inference/control timings |
| Is enough evidence present to choose a larger run? | Hashed inventory, stage status, missing artifacts, paired validation counts and `scale_readiness.json` |

Intermediate reasoning stays in continuous vectors. Decoded text is an output
check rather than a transcript of those vectors. Alignment/state similarities
are proxies; interventions can be out of distribution. Explicit controls have a
192-token cap versus 64 for latent answer decoding, so they are not a matched
compute comparison. Depth interventions use saved weights, not separately
trained depth controls. Per-loss gradient norms cannot be added as scalars.

Probe rows are fixed in prepared source order before seeing their results.
No-event and overflow rows remain in the main losses, with the original auxiliary
masking policy. The source-data quality exclusions remain unapplied. Validation
is for development; the full test evaluation remains disabled by default.

## The final review ZIP

The last cell saves and downloads a **small review ZIP** under
`MyDrive/dynamic-state-anchoring/bundles/`. Its name includes the run, code,
settings and checkpoint identities. **Hand back this ZIP for analysis.**
It contains training journals/manifests, checkpoint metadata, prepared train/valid
records, old/new predictions, diagnostics, explicit controls, provenance, logs,
per-file hashes and completeness status. It excludes weights and test answers.
Recorded GPU-stage failures do not prevent partial export; missing evidence is
listed explicitly. Setup and Drive authorization still require a working runtime.

The separate `<run>-full-checkpoints.zip` downloads after training. **Keep it as
your independent backup.** It can be several GB and includes model/tokenizer,
optimizer, scheduler, RNG and journals, with a SHA256 inventory inside each arm ZIP.
Individual full archives also live at
`MyDrive/dynamic-state-anchoring/bundles/<run>/checkpoints/`.
If the Drive copy fails, the local archive remains downloadable. The export cell
retries initiating a full download if the earlier attempt raised an error.
A browser download request is not confirmation that the transfer completed.

`scale_readiness.json` uses an explicitly labeled 80% training-accuracy heuristic
for memorization and reports nonfinite gradients, mismatched controls or missing
evidence as blockers. It does not automatically approve scaling or prove a
reasoning advantage. We review the full evidence first, then freeze a larger
exploratory protocol: data size, update budget, stabilization choices, all four
write policies, repeated seeds and a validation plan. Held-out test evaluation
comes after that protocol is frozen. No larger run is started automatically.

## Starting another small pilot

Set `RUN_MODE="train_pilot"` and choose `NEW_RUN_NAME`. Defaults remain
32 unchanged training rows, 200 one-example updates per static/learned arm,
K=16, workspace four, FP32 and activation checkpointing. A fully completed
compatible arm can be reused; any partial arm restarts. This notebook deliberately
favors cheap small-pilot restarts over exact-resume recovery. The underlying CLI
still supports exact resume for larger experiments when its full contract matches.
`review_existing` remains an explicit review-only option for intact historical runs.

## Setup and verification boundary

The dependency helper creates a system-site environment without `ensurepip` and
installs through host pip with an explicit target interpreter. It preserves
Colab's CUDA/PyTorch packages. Preflight supports Python 3.10–3.13, reports GPU
visibility, tests a tiny GPT-2 forward and exercises the review pipeline locally.
Subprocess output is streamed and saved under the selected review's `logs` folder.
CUDA unavailable means select a T4 runtime and reconnect before changing packages.

The expanded workflow has end-to-end CPU tests using tiny saved checkpoints,
including automatic retraining from metadata-only runs, full-backup restoration,
Drive copy failures, full/partial exports, latent intervention isolation and
preservation of the joint objective. Pretrained T4 execution of
these new checks still needs to happen in Colab. No claim is made that every
future scientific question can be answered from a small single-seed pilot.
