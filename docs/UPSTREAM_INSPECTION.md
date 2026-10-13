# Upstream inspection, 12 October 2026

The initial implementation uses LOTUS as its checkout and reads the other
repositories as implementation references. The PDF is a research specification,
not evidence of trained models or benchmark results. The current source pins and
inspected-file SHA256s are recorded in `provenance/sources.json`.

| Source | Actual checkout SHA | License observed |
| --- | --- | --- |
| LOTUS | `eb77e2f7909c5006f58ff0ad7cd6629b942caa9e` | Root `LICENSE`: MIT, 2026 The Authors; reused scripts retain Meta/Coconut notices. |
| looped-tf | `33650c3c0cec3dd1d466b32489aaf30df4bda796` | No standalone license file located in tracked checkout. Inspection only; no upstream implementation copied. |
| CODI | `2c2314662c63e9f482ebc46614ffe9af17a241e5` | No standalone license file located in tracked checkout. Inspection only; no upstream implementation copied. |
| grade-school-math | `3101c7d5072418e28b9008a6636bde82a006892c` | Root `LICENSE`: MIT, 2021 OpenAI. |
| Internalize_CoT_Step_by_Step | `e06a32ee5e4cd117171daeb4755d2a97ece62761` (detached) | Root `LICENSE`: MIT, 2024 Yuntian Deng. |

Inspection sources live beside the main checkout in `../dsa-sources/`. The
Internalize checkout used `GIT_LFS_SKIP_SMUDGE=1`; this machine lacks `git-lfs`,
so the partial checkout was recovered with LFS filters disabled. Its large
`train.txt` remains a pointer and must never be parsed as examples.

## LOTUS model and harness

[`scripts/lotus.py` at the inspected commit](https://github.com/yingfan-bot/lotus/blob/eb77e2f7909c5006f58ff0ad7cd6629b942caa9e/scripts/lotus.py)
provides GPT-2/Llama model detection, embedding access, latent feedback,
answer decoding, and auxiliary losses. It refines the contiguous latent region,
while retaining the question prefix. Its default feedback is original embedding
plus current hidden state; the `replace` variation removes that residual. The
comment mentioning division by two is stale: the executed default addition has
no division. It does not implement separate persistent anchor `A` and workspace
`Z`, a write gate, or milestone candidate supervision.

The constructor asserts `use_kv_cache is True` (line 292), and
`efficient_forward` hardcodes `use_cache=True`. Therefore a correctness-first DSA
implementation requires its own full-recompute backbone adapter; setting the
upstream flag to false is insufficient. Prefix caching, mutable DynamicCache
cloning, auxiliary decoders, frozen teachers, and external auxiliary checkpoints
are upstream options, not the requested initial DSA experiment. The latter uses
one shared backbone and detached, freshly computed explicit targets.

[`scripts/dataset.py`](https://github.com/yingfan-bot/lotus/blob/eb77e2f7909c5006f58ff0ad7cd6629b942caa9e/scripts/dataset.py)
tokenizes question-plus-newline, each step-plus-newline, and `### ` plus answer
separately. A first-row assertion checks agreement with combined tokenization.
The collator shifts latent positions together using leading pads and pads labels
with `-100`. Dataset builders replace scheduled reasoning fragments with latent
tokens and can choose stages using gold step length. Those schedules and
fragment tokenizations must not become DSA's production inference input or
marker alignment logic. DSA tokenizes its complete marked explicit sequence once
and separately maps raw-rationale offsets.

[`scripts/run.py`](https://github.com/yingfan-bot/lotus/blob/eb77e2f7909c5006f58ff0ad7cd6629b942caa9e/scripts/run.py)
registers `<|start-latent|>`, `<|end-latent|>`, and `<|latent|>` using
`add_tokens`, loads Stage-1 checkpoints, resizes embeddings, and uses distributed
FSDP/DDP conventions. It does not register `<SA>`. DSA needs collision checks,
one indivisible special marker token, trainable new embedding rows, and preserved
weight tying. The supplied GPT-2 LOTUS config initializes from
`yingfanbot/gsm-cot-gpt2`; its model revision is not resolved by the YAML. This
checkpoint has not been downloaded or smoke tested in this initial work.

[`scripts/eval.py`](https://github.com/yingfan-bot/lotus/blob/eb77e2f7909c5006f58ff0ad7cd6629b942caa9e/scripts/eval.py)
reads local `data/gsm_test.json` for `--datasets gsm8k`, preserving its row order.
It strips questions, removes commas and strips supplied answers, encodes the
question followed by a newline, and defaults to `max_new_tokens=64`. Extraction
uses the final `###` segment, then the last numeric substring, then a `#` fallback.
Scoring uses exact float equality with stripped string equality on conversion
failure. DSA benchmark evaluation must explicitly pin these rules and save
paired predictions; a more general numeric scorer is a separately reported
condition. The split manifest stores the ordered sample IDs, question hashes,
input-text hashes, normalized supplied answers, and an ordered-record checksum.

## Pinned benchmark conversion

The actual
[`preprocessing/gsm_icot.bash`](https://github.com/yingfan-bot/lotus/blob/eb77e2f7909c5006f58ff0ad7cd6629b942caa9e/preprocessing/gsm_icot.bash)
matches the PDF's data-source commit exactly. It downloads training data from
the media endpoint and validation/test from raw endpoints. It creates
`data/gsm_{train,valid,test}.txt`, converts each file, then removes these text
downloads. The actual output names differ from the PDF's early reference to
`data/gsm8k/train.txt`: the runnable script and dataset YAML agree on
`data/gsm_train.json`, `data/gsm_valid.json`, and `data/gsm_test.json`.

[`gsm_icot.py`](https://github.com/yingfan-bot/lotus/blob/eb77e2f7909c5006f58ff0ad7cd6629b942caa9e/preprocessing/gsm_icot.py)
extracts question text before `||`, strips the rationale before `##`, splits it
using `.split(" ")`, and strips the final answer. These are space-delimited
fragments, not guaranteed semantic lines. DSA therefore joins `steps` with a
single space, preserving empty fragments and all original JSON fields. Original
arrays remain unchanged; JSONL serialization preserves each row's JSON value,
although serialization bytes necessarily differ from the source array.

HTTP **HEAD** checks on all three exact pinned URLs returned **200**. Reported
lengths were 87,805,358 bytes for train, 143,419 for valid, and 388,019 for test.
The training media ETag and length matched the checked-out LFS pointer's SHA256
OID and size. These are endpoint/header checks, not downloaded-payload checksums.
The full augmented training payload was not downloaded. No complete LOTUS split
overlap, annotation coverage, or official-test membership audit has been run.

`scripts/prepare_dsa_data.py` accepts supplied arrays, verifies the pinned
download URLs and hashes both actual preprocessing scripts, audits explicit
problem groups and normalized question duplicates across every partition, and
stops on overlap before writing derivatives. It then writes unchanged row values
to JSONL, annotates both `conclusive` and `verified_only`, retains every row,
and writes coverage reports and source/derivative checksums. An optional
`--official-test` compares question membership, order, and normalized answers
against a separately supplied original GSM8K JSONL. Normalized hashes detect
case, Unicode-format, and whitespace duplicates; they do not prove the absence
of paraphrased duplicates. Tokenizer mapping and loop-overflow checks remain a
separate stage.

```bash
python3 scripts/prepare_dsa_data.py \
  --train data/gsm_train.json --valid data/gsm_valid.json --test data/gsm_test.json \
  --official-test ../dsa-sources/grade-school-math/grade_school_math/data/test.jsonl \
  --output-dir data/dsa_prepared
```

The command requires the original arrays to exist and an empty output directory.
It does not download the benchmark or change the supplied partitions.

## Controlled task references

[`src/train.py`](https://github.com/UW-Madison-Lee-Lab/looped-tf/blob/33650c3c0cec3dd1d466b32489aaf30df4bda796/src/train.py)
uses online generators from `src/generate_training_data.py`, a length curriculum,
and task-specific recurrent output selection. Nonmultiplication examples select
the state at sampled digit length minus one; multiplication uses the product of
the operand lengths minus one. The generator's `max_num_digits` bound is
exclusive. These are task/depth conventions, not train/validation/test files.

| Config | Curriculum upper bound | Held-out test length | Architecture |
| --- | --- | --- | --- |
| `src/conf/addition.yaml` | 2 to 20, increment every 1600 steps; effective max train digits 19 | 30 | width 256, 3 layers, 8 heads |
| `src/conf/parity.yaml` | 2 to 21, every 500; effective max train digits 20 | 40 | width 256, 1 layer, 64 heads |
| `src/conf/multi.yaml` | 2 to 12, every 500; effective max second operand digits 11 | 15 | width 256, 4 layers, 8 heads |

`multi` is binary multiplication, not multi-hop function composition. Its first
operand has one or two bits during training and two at test. Addition supplies
two equal-length binary operands, preserves a carry digit, and returns exact
masked targets. Parity supplies a scalar modular sum. These generators do not
provide DSA intermediate solver-state annotations; those must be defined
independently with explicit operation difficulty and milestones.

`src/models.py` injects the original embedded source on every recurrence,
starting with zeros, and calls `GPT2Model.forward_no_position`. That method
belongs to the repository's vendored Transformers fork, not ordinary current
Transformers. Porting this architecture therefore requires an explicit position
policy and cannot assume the upstream Python module imports unchanged. Its
environment targets Python 3.9.19 and Torch 2.1.2/CUDA 12.1; LOTUS requirements
target an NGC PyTorch image and Transformers 4.46.2. These environments were
inspected, not installed together.

## CODI reference

[`src/model.py`](https://github.com/zhenyi4/codi/blob/2c2314662c63e9f482ebc46614ffe9af17a241e5/src/model.py)
uses the same `self.codi` backbone for explicit and latent routes. It computes
explicit targets under `torch.no_grad`, separately computes explicit CE with
gradients, and explicitly detaches gathered reference states for alignment.
Targets are gathered at answer-boundary positions across hidden layers;
alignment is applied after the last latent step. Defaults use SmoothL1; `l2`
selects MSE. Its optional standard-deviation scaling divides twice for `l2`.
These details differ from DSA's intermediate, final-layer, pre-marker targets
and non-affine-layer-normalized candidate MSE.

CODI sequentially feeds a last hidden vector into the next latent token using
KV cache. It includes optional 4-bit loading, LoRA, and a projection module.
These are not part of the DSA initial scope. The reusable design observation
is shared-width routes with fresh detached reference targets and explicit-route
supervision; the requested anchor and learned gate require a separate model.

## Validation and remaining work

`python3 -m unittest discover -s tests -p test_provenance.py -v` passes 13 fixture
tests covering split overlap, group IDs, ordered evaluation records, pointer and
shape rejection, lossless conversion, both annotation policies, missing events,
source hashes, output protection, and official-test discrepancies. These tests
do not establish benchmark corpus quality or model accuracy.

Before a meaningful GSM8K run: obtain the exact LOTUS arrays, run the split and
official-test comparisons, review actual augmented-rationale coverage and a
reproducible human sample, resolve and smoke test the GPT-2 checkpoint revision,
and finish tokenizer alignment and loop-controller coverage audits.
