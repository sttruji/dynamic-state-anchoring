# Run the DSA implementation pilot in Colab

Open [the pilot notebook in Colab](https://colab.research.google.com/github/sttruji/dynamic-state-anchoring/blob/main/notebooks/DSA_Colab.ipynb).
The first cell clones `sttruji/dynamic-state-anchoring`, using
`main` by default; set `REPO_REF` to a commit SHA for a reproducible run. The
repository is public, so its default clone needs no GitHub token.
Configure authenticated Git separately if using a private fork;
never save tokens in the notebook.

Select a GPU runtime and run cells in order:

1. Clone the implementation and install `requirements-colab.txt` in an isolated
   environment that reuses Colab's existing CUDA/PyTorch stack.
2. Mount Drive and configure persistent dataset, Hugging Face, checkpoint and
   report directories under `MyDrive/dynamic-state-anchoring`.
3. Review the pilot settings, verify the actual runtime and run correctness tests.
4. Download/checksum the pinned LOTUS splits once, prepare a deterministic
   training subset and stage the prepared files on the VM.
5. Verify inference from the unmodified pinned GPT-2 checkpoint before adding the
   marker or anchor wrapper. Model downloads reuse the Drive cache.
6. Train static and learned policies with matching initialization/data order;
   save resumable model, optimizer and RNG state on Drive.
7. Evaluate question-only predictions on validation. Enable final full test
   evaluation only after experimental choices are frozen.

Defaults are 32 training rows, 200 updates per arm, K=16, workspace size 4,
FP32, gradient checkpointing, and 32 validation questions. These are plumbing and
profiling settings. They do not establish comparative benchmark performance.
Every original validation/test row is prepared, even when validation generation
uses a diagnostic subset. No source-quality exclusions are applied.

## What you get from the small GPU pilot

The pilot produces two trained checkpoints: a static-anchor control and a model
with learned anchor writes. Each sees the same 32 training examples and receives
200 one-example optimizer updates (about 6.25 passes through that subset).
Both perform all 16 recurrent loops; fewer writes do not reduce the loop count.

Its first purpose is feasibility: verify the original pretrained checkpoint
loads, the GPU can run forward/backward passes with finite losses, and the
memory/runtime fit the intended experiment. Metrics show answer, explicit
reasoning, gate and anchor losses, controller gradients/probabilities and write
rates. These let us investigate unstable training or a controller that always
opens or closes before scaling up.

Its second purpose is inspection: you get question-only predictions on the first
32 validation questions, correctness and invalid-answer rates, per-question
latency and per-loop write traces. The saved predictions let us inspect actual
outputs instead of relying only on training losses. All results and resumable
model/optimizer/RNG state stay on Drive.

These outputs support a decision to debug or scale the experiment. They do not
establish that anchoring improves reasoning: one tiny training subset, one seed,
two write policies and 32 validation questions are insufficient for that claim.
The original-checkpoint smoke is an inference check, not a full accuracy
baseline. The current notebook does not generate answers on the training subset,
so decreasing teacher-forced loss alone does not confirm successful overfitting.
Training-set generation is a useful follow-up, then larger matched comparisons,
all four write policies, multiple seeds and causal intervention tests.

Update timings exclude setup/checkpoint overhead; GPU memory records allocated
tensors rather than total device occupancy. Use the measured profile to plan the
next run, without treating it as a complete billing/runtime estimate.

## Reconnect and resume

Rerun setup with the same implementation ref and settings. Verified corpus/model
downloads are reused; rerun training with `--resume auto` to restore the last
complete checkpoint. The default retains two complete checkpoints per arm and
saves every 25 updates. Updates after the most recent completed checkpoint may
need to be repeated. Reserve several GB per arm for FP32 model and Adam state.

Increase `STEPS` to extend an existing run. Changed model/data/settings, code,
package versions, CUDA or GPU identity are rejected by exact resume; start a new
experiment name when those change. A new Colab session can have different
hardware, so exact continuation is conditional on a matching runtime. Download
reuse remains available even when training needs a fresh run.

Training writes `run_manifest.json`, `metrics.jsonl`, `train_report.json`,
`latest.json` and complete checkpoint directories. Evaluation writes an identity,
append-only resumable predictions and `report.json`. The last notebook cell
writes `notebook-summary.json`. Reports include ordered evaluation IDs, invalid
answer rates, latency and recurrent-forward counts; training metrics record
update duration and peak memory. The notebook summary derives throughput and
loss/write profiles from those metrics. Keep these artifacts with any result
you report.

## Repair an older notebook copy after a setup failure

The corrected dependency cell uses `scripts/setup_colab.py`. It creates the
environment without invoking `ensurepip`, installs through Colab's existing pip
with an explicit target interpreter, and repairs a partial environment left by
an earlier failed attempt. The preflight accepts Python 3.13.

A saved Drive notebook does not update when the repository changes. Open the
current GitHub notebook linked above, or rerun your first clone cell and replace
the dependency and preflight cells with their current versions from
`notebooks/DSA_Colab.ipynb`. Merely rerunning the old dependency cell leaves its
old setup command in place. This repair requires no GPU upgrade or global
PyTorch reinstall.

## Current verification boundary

Notebook syntax, dependency isolation, source caching, tiny-model training,
checkpoint resume and evaluation are tested locally. Hosted Colab, Drive
authorization and the pretrained CUDA run still need execution.
The next experiment is the unmodified-checkpoint smoke followed by this small
matched training pilot. Inspect losses, generation, memory and throughput before
increasing training size or running the four-policy, multi-seed study.

See `docs/COLAB_ENVIRONMENT.md` for dependency and persistence details,
`docs/LOTUS_CORPUS_AUDIT.md` and `docs/LOTUS_PERSONAL_REVIEW.md` for the source-data
findings, and `IMPLEMENTATION_STATUS.md` for the remaining research work.
