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

## Diagnose the saved 200-step pilot

Open [the checkpoint diagnostics notebook in Colab](https://colab.research.google.com/github/sttruji/dynamic-state-anchoring/blob/main/notebooks/DSA_Pilot_Diagnostics.ipynb)
on a T4 GPU. Run its cells in order. It targets
`gpt2-pilot-v1-ba4e69315086` at step 200, independently of the newer diagnostic
code commit. It finds the original prepared Drive cache by manifest checksum;
it does not call training, download data/model weights, or use validation/test
answers. It loads the existing checkpoint files and creates no optimizer.

The notebook generates answers on all 32 training questions, then probes the
first four prepared training rows using K=1,2,4,8,16 and disabled/always-on anchor
writes. Each intervention is paired with normal inference on the same questions.
These are interventions on saved weights, not separately trained depth controls.
Controller precision/recall against training labels is reported at the original
budget. Generated token IDs, EOS/cap stop reasons and source identities are saved.

On the first two prepared training rows it measures answer-loss gradients at
each depth, then separates weighted answer/explicit-CoT/gate/alignment and total
gradients at K=16. Each component shares a forward graph and dropout realization.
The full objective keeps the original labels and training straight-through gate;
the depth sweep uses answer loss only. Hooks measure the actual recurrent input
tensors. State norms, changes, similarities and gradients are numerical
diagnostics; decoded text is not a transcript of continuous latent reasoning.

No clipping or optimizer updates occur. Model-state checksums must remain equal
before/after. Outputs persist under
`reports/gpt2-pilot-v1-ba4e69315086/checkpoint-diagnostics-v1/`. Each completed
prediction and gradient case is reusable after interruption with matching
settings/code/runtime. Changed diagnostic settings need a new `DIAGNOSTIC_NAME`.
The last cell downloads `dsa-checkpoint-diagnostics.zip` for review. The diagnostic
pipeline is tested locally with tiny checkpoints; the pretrained T4 diagnostic
still requires execution in Colab.

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

If preflight reports CUDA unavailable, inspect its printed runtime probe. It
shows the PyTorch version, CUDA build, availability and visible device count.
Select a T4 GPU in Colab's runtime settings, save/connect, and rerun setup after
the runtime changes. Keep `DEVICE="cuda"` for the GPU pilot. If a GPU is already
allocated, compare `nvidia-smi` with that probe before changing packages; CUDA
unavailability alone does not identify a package defect.

Project subprocess output is streamed into the notebook and saved under the
experiment's `reports/.../logs` directory on Drive. A failed command displays
the underlying error and its log path, instead of only a generic
`CalledProcessError` containing the entire inline Python program.

Notebook syntax, dependency isolation, source caching, tiny-model training,
checkpoint resume and evaluation are tested locally. Each new hosted runtime
still needs its own preflight. For an already completed pilot, the saved-checkpoint
diagnostics above are the next step before increasing training size or running
the four-policy, multi-seed study. The pretrained CUDA diagnostic has not been
executed locally.

See `docs/COLAB_ENVIRONMENT.md` for dependency and persistence details,
`docs/LOTUS_CORPUS_AUDIT.md` and `docs/LOTUS_PERSONAL_REVIEW.md` for the source-data
findings, and `IMPLEMENTATION_STATUS.md` for the remaining research work.
