# One Colab notebook for the pilot and its review

Open [DSA_Colab.ipynb in Colab](https://colab.research.google.com/github/sttruji/dynamic-state-anchoring/blob/main/notebooks/DSA_Colab.ipynb),
select **T4 GPU**, and run all cells in order. This is the original notebook,
expanded to collect the evidence needed before choosing a larger experiment.
A saved Drive copy does not update automatically; open the current GitHub version.

## Reconnect to the completed pilot

The default `RUN_MODE="review_existing"` selects
`gpt2-pilot-v1-ba4e69315086`. Fetching newer review code does not select a new run.
The notebook locates the original prepared Drive cache by manifest checksum and
reads each arm's latest completed checkpoint. This mode skips training, corpus
preparation/downloads and original-model smoke generation. It does not create an
optimizer. Model controls load original pretrained weights from the HF cache with
`local_files_only=True`; a missing cache is reported rather than downloaded.

On a fresh VM, dependency setup and Drive mounting must run again. Dataset files,
model downloads, model/optimizer checkpoints and completed review cases persist
on Drive. Inputs are staged back to the VM from the existing prepared cache.
The default project folder is `MyDrive/dynamic-state-anchoring`.

`REVIEW_NAME="pilot-review-v2"` keeps the new review separate from the original
200-step reports. A review freezes its configuration and code commit in
`protocol.json`; changed settings/code require a new review name. Individual
runners also validate source, checkpoint, tokenizer, implementation and runtime
identities before reusing results. Identical interrupted cases resume without
regenerating completed examples. Reviews can record different hardware from the
training run, but do not claim exact training continuation across hardware.

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

The last cell creates and downloads:

```text
MyDrive/dynamic-state-anchoring/bundles/
  gpt2-pilot-v1-ba4e69315086-pilot-review-v2.zip
```

Hand back this one ZIP. It includes original training journals/manifests and
checkpoint metadata, prepared training/validation records and selection metadata,
old/new validation predictions, diagnostics, explicit controls, runtime/protocol
identities, logs, source scripts, per-file hashes and a completeness manifest.
Weights, optimizer tensors, HF downloads, the raw full training corpus and test
questions/answers are excluded. Checkpoints remain on Drive for continuation.

GPU stage failures are logged in `session_status.json` and do not stop later
stages or the final export. Partial ZIPs explicitly report missing evidence.
Setup/Drive/input-location failures still need resolution before a review can
run. If disconnected before export, reconnect with the same review configuration;
finished cases are reused and remaining cases continue. The bundle also stays on
Drive if the browser download is interrupted.

`scale_readiness.json` uses an explicitly labeled 80% training-accuracy heuristic
for memorization and reports nonfinite gradients, mismatched controls or missing
evidence as blockers. It does not automatically approve scaling or prove a
reasoning advantage. We review the full evidence first, then freeze a larger
exploratory protocol: data size, update budget, stabilization choices, all four
write policies, repeated seeds and a validation plan. Held-out test evaluation
comes after that protocol is frozen. No larger run is started automatically.

## Starting another small pilot

Set `RUN_MODE="train_pilot"` and choose `NEW_RUN_NAME`. Defaults remain the
original 32 unchanged training rows, 200 one-example updates per static/learned
arm, K=16, workspace size four, FP32 and activation checkpointing. Training starts
from matching seeds and data order and retains two complete checkpoints per arm.
`--resume auto` restores model, Adam, scheduler and RNG state only when the exact
training contract matches; changed dependencies/hardware/settings/code need a
separate experiment. Increase the total step target only for an authorized
extension of the same matched experiment.

## Setup and verification boundary

The dependency helper creates a system-site environment without `ensurepip` and
installs through host pip with an explicit target interpreter. It preserves
Colab's CUDA/PyTorch packages. Preflight supports Python 3.10–3.13, reports GPU
visibility, tests a tiny GPT-2 forward and exercises the review pipeline locally.
Subprocess output is streamed and saved under the selected review's `logs` folder.
CUDA unavailable means select a T4 runtime and reconnect before changing packages.

The expanded workflow has end-to-end CPU tests using tiny saved checkpoints,
including reconnect without training, full/partial exports, latent intervention
isolation and preservation of the joint objective. Pretrained T4 execution of
these new checks still needs to happen in Colab. No claim is made that every
future scientific question can be answered from a small single-seed pilot.
