# Implementation status — updated 12 November 2026

**Status:** the experiments are finished, and the course report is a first draft in [`paper/main.tex`](paper/main.tex).
The [README](README.md) tells the story and gives the headline results. This file lists, for each stage of the main
line, what was built, where its code and records live, how it ran, and which doc reports it. The October implementation
log this file used to hold is in git history (last changed in `41c81d4`).

All work is merged into `main`. Unit tests are in `tests/` and run with
`.venv/bin/python -m unittest discover -s tests`.

## 1. The original DSA on GPT-2

The gated anchor design of the [Revision 4 specification](docs/DSA_IMPLEMENTATION_HANDOFF.md): a learned, hard-gated
anchor write controller with FLT recurrence, starting from the LOTUS GPT-2 CoT checkpoint, against `static`,
`every_loop` and `periodic` controls. Outcome: the controller was at chance and stopped writing, and no write policy
beat writing nothing.

- **Code.** `dsa/`: the model (`dsa/models/anchored_loop.py`), FLT recurrence (`dsa/models/flt_memory.py`), LoRA
  (`dsa/models/lora.py`), losses (`dsa/training/objectives.py`), `<SA>` annotation (`dsa/data/milestones.py`) and
  tokenization (`dsa/data/tokenize.py`). Entry points: `scripts/prepare_dsa_pilot.py`, `scripts/train_dsa.py`,
  `scripts/evaluate_dsa.py` and, for the CODI-style recipe, `scripts/run_codi_recipe.py`.
- **Runs.** The 32-question pilots (K=16, then K=8 standard against FLT) and the 512-question FLT data-scale run were
  Colab runs from `notebooks/DSA_Colab.ipynb` and `notebooks/DSA_FLT_DataScale.ipynb`, followed by a local inference
  check of the data-scale checkpoints. Their implementation references are in the
  [findings table](docs/FINDINGS.md#cost-and-reproducibility). The CODI-recipe no-write arm ran on one A100 from
  `notebooks/DSA_CODI_Recipe_Colab.ipynb` at code `143f24a`, and the every-loop and periodic arms from
  `notebooks/DSA_CODI_Writes_Colab.ipynb` at code `1e80405`.
- **Records.** `provenance/findings_summary.json`; `results/leaderboard.csv` rows `dsa-data512-*` and `dsa-codi-*`;
  per-update metrics of the no-write CODI-recipe arm in `results/training/dsa-codi-static-n32k`.
- **Docs.** [FINDINGS.md](docs/FINDINGS.md), [CODI_RECIPE.md](docs/CODI_RECIPE.md), [DSA_README.md](DSA_README.md),
  and the protocols [K8_FLT_PILOT.md](docs/K8_FLT_PILOT.md) and [FLT_DATA_SCALE.md](docs/FLT_DATA_SCALE.md).
- **Earlier groundwork.** The upstream inspection, the LOTUS corpus audit and personal review, the checkpoint-label
  audit and the local MPS checks are in [UPSTREAM_INSPECTION.md](docs/UPSTREAM_INSPECTION.md),
  [LOTUS_CORPUS_AUDIT.md](docs/LOTUS_CORPUS_AUDIT.md), [LOTUS_PERSONAL_REVIEW.md](docs/LOTUS_PERSONAL_REVIEW.md),
  [CHECKPOINT_LABEL_AUDIT_20261019.md](docs/CHECKPOINT_LABEL_AUDIT_20261019.md) and [LOCAL_MPS.md](docs/LOCAL_MPS.md),
  with their hashed records under `provenance/`.

## 2. Benchmark data and evaluation harness

Pinned datasets and one scorer for every model, added on 25 October.

- **Code.** `scripts/setup_data.py` and `dsa/bench/data.py` build every dataset from pinned sources and check each file
  against `provenance/bench_data_lock.json`. `scripts/evaluate.py`, `dsa/bench/harness.py` and the wrappers in
  `dsa/bench/models/` (`gpt2-vanilla`, `gpt2-sft`, `codi`, `lotus`, `dsa`) score models into the leaderboard.
- **Records.** `results/leaderboard.csv`, rebuilt from `results/runs/`, with every output in `results/predictions/`.
  The leaderboard has no LOTUS row; no GPT-2 LOTUS checkpoint is public (see BENCHMARK.md).
- **Docs.** [BENCHMARK.md](docs/BENCHMARK.md); the repository audit of that day is
  [AUDIT_20261025.md](docs/AUDIT_20261025.md).

## 3. CODI past its trained depth

No-training measurements of the published CODI GPT-2 at more than 6 latent steps, with two preregistered tests (both
failed), added on 2 November.

- **Code.** `scripts/codi_trail/`.
- **Runs.** Local, on Apple MPS.
- **Records.** `results/codi_trail/`: the preregistration, the dynamics plan, the harness sweep, the probe records and
  the report outputs. The probe's latent files are not committed.
- **Doc.** [CODI_TRAIL.md](docs/CODI_TRAIL.md).

## 4. SIM-CoT, the copy test and Stage 0

SIM-CoT GPT-2 drifts as much as CODI with stable states, and appended copies of one cache entry reproduce its loss. Stage
0 then tested eviction rules at fixed weights. All on `gsm8k_valid`, added on 2 November.

- **Code.** `scripts/latent_gate/fetch_simcot.py`, the evidence scripts in `scripts/latent_gate/evidence/` (SIM-CoT
  probe, attention shares and the copy test), and `scripts/latent_gate/stage0_evict.py`, `stage0_labels.py`,
  `stage0_report.py` and `stage0_labels_report.py`.
- **Runs.** Local, on Apple MPS, with no training.
- **Records.** `results/latent_gate/simcot_probe/`, `results/latent_gate/copy_test/` and `results/latent_gate/stage0/`.
  The copy test and Stage 0 plans were written before their runs and carry a `PLAN.sha256`.
- **Docs.** [LATENT_GATE_IDEA.md](docs/LATENT_GATE_IDEA.md),
  [scripts/latent_gate/evidence/README.md](scripts/latent_gate/evidence/README.md) and the literature review
  [LATENT_GATE_LITERATURE.md](docs/LATENT_GATE_LITERATURE.md).

## 5. Stage 1: fine-tuning with the gate

Eight arms on SIM-CoT GPT-2 and three on CODI GPT-2, one seed (7). H2 failed on both, so the stop rule fired.

- **Code.** `scripts/latent_gate/stage1_train.py`, `stage1_eval.py`, `gate.py` and `stage1_report.py`.
- **Runs.** `notebooks/DSA_Latent_Gate_Stage1_Colab.ipynb` on a Colab A100 at commit `161751d`.
- **Records.** `results/latent_gate/stage1/PLAN.md` and `PLAN.sha256`; `results/latent_gate/stage1/{simcot,codi}/`, each
  with `evals/`, `training/` and `report.txt`.
- **Doc.** [LATENT_GATE_STAGE1_RESULTS.md](docs/LATENT_GATE_STAGE1_RESULTS.md).
- **Known issue.** Both `gate12_hard` arms trained with a tie bug at the top-6 cut. It was fixed for the held-out study
  (`gate12_fixed`), and the gate still failed on CODI.

## 6. The held-out and scale study

Four published checkpoints (CODI and SIM-CoT at GPT-2 and LLaMA-3.2-1B) on GSM8K test, GSM-Hard, MultiArith and SVAMP,
under a plan frozen before any run, plus one amendment.

- **Code.** The batched evaluator `scripts/latent_gate/batched.py` and `heldout_eval.py`; the engineering checks
  `check_batched.py` (E1) and `check_llama.py` (E2); the LLaMA port `dsa/bench/models/codi_llama.py`; the reports
  `heldout_report.py`, `heldout_explore.py` and `amendment_report.py`; and `partc_train.py` for Amendment 1's Part C.
- **Runs.**
  - Parts A and B: `notebooks/DSA_Latent_Gate_HeldOut_Scale_Colab.ipynb` on a Colab A100 at commit `837f5df`.
  - Amendment 1 at commit `48a461d`: `notebooks/DSA_Latent_Gate_Anchor_PartC_Colab.ipynb` (B3, then Part C) and the
    Part C-only `notebooks/DSA_Latent_Gate_PartC_Colab.ipynb`. Part C trained on a Colab G4 (RTX PRO 6000).
  - Not run: the optional 3B SIM-CoT model.
- **Records.** `results/latent_gate/heldout_scale/`: `PLAN.md` and `PLAN.sha256`, one folder of records per model and
  arm, `report.txt`, `explore.txt`, `amendment_report.txt`, `partc_verdict.json` and the training logs under
  `training/`. `heldout_report.py` and `amendment_report.py` reproduce `report.txt` and `amendment_report.txt` from
  these records without a GPU.
- **Doc.** [LATENT_GATE_HELDOUT_RESULTS.md](docs/LATENT_GATE_HELDOUT_RESULTS.md).

## 7. The course report

- **Source.** `paper/main.tex`, with `paper/references.bib` and the ACL style files. The figure
  `paper/figures/depth.tex` is written by `paper/figures/depth_figure.py` from the held-out records.
- **Build.** `cd paper && latexmk -pdf main.tex`. Build outputs, including `main.pdf`, are ignored by git.
- **State.** A first draft.

## Separate exploration

A parallel brainstorm track, after Stage 1, is summarized in [BRAINSTORM_SUMMARY.md](docs/BRAINSTORM_SUMMARY.md). Its
code is under `scripts/inner_loop/`, `scripts/step_select/`, `scripts/plan_search/`, `scripts/step_probe/` and
`scripts/ffn_sparsity/`, and none of it is part of the course report.
