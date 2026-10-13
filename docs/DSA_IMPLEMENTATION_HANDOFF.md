# Dynamic State Anchoring for Looped Transformers
## Anchoring-only research specification and coding-agent handoff
Revision 4 | 12 October 2026 | Status: proposed model; tested reference preprocessing

## 1. Start here: the research decision

Implement one focused hypothesis: selectively writing intermediate conclusions into a persistent latent anchor improves the reliability of a looped transformer's subsequent computation, particularly beyond its training depth.

Use one backbone size and initialization across the anchoring ablations. Do not introduce cross-size teachers, pruning, layer removal, quantization, or a parameter-reduction objective. A modest checkpoint may be selected for affordability, but its size is fixed across the experiment.

The experimental variable is how the anchor is updated. Original question context stays accessible. The anchor is an additional evolving memory, not a replacement for the problem statement. An anchor is a predicted intermediate representation; a high gate probability does not verify its correctness.

The proposal has no training results or established novelty claim. The target contribution is selective, milestone-supervised anchor updates with controlled evidence of useful state preservation. Do not claim that an update resets all accumulated error or permits infinite extrapolation.

### Brief abstract

Looped transformers reuse their weights to perform additional computation in latent space, but useful intermediate states can deteriorate at greater recurrent depths. We propose Dynamic State Anchoring: a persistent latent memory that is selectively refreshed from the current working state. A deterministic preprocessing pipeline identifies conclusive arithmetic statements and isolated-variable assignments in existing reasoning traces, inserting a `<SA>` annotation at each accepted intermediate boundary. These annotations supervise a learned write controller and an auxiliary checkpoint representation objective; they are never supplied as oracle triggers at inference. Comparisons hold backbone size, initialization, reasoning budget, and milestone supervision fixed to isolate the contribution of selective writes. Controlled algorithmic tasks test depth extrapolation, while GSM8K provides natural-language arithmetic reasoning. The central hypothesis is improved continuation from intermediate states.

### What is included in this handoff

- This self-contained model and experiment specification.
- `scripts/annotate_milestones.py`: executable standard-library reference preprocessing and a training-only loop-label mapper.
- `tests/test_milestones.py`: 33 behavioral checks.
- `examples/`: eight synthetic input records, annotated records, and an annotation report.
- `anchor_config.example.yaml`: proposed configuration contract, not a working training launcher.

The model wrapper, tokenizer adapter, trainer, GPU evaluation, and full-corpus audits still need implementation. No checkpoint was trained for this revision. The reference tagger was tested on the included synthetic records, not on the complete GSM8K corpus.

## 2. Sources and implementation starting point

Use LOTUS as the main repository and create an anchoring branch.

```bash
git clone https://github.com/yingfan-bot/lotus.git dynamic-state-anchoring
cd dynamic-state-anchoring
git switch -c research/dynamic-state-anchoring
git rev-parse HEAD
```

Record the actual checkout SHA, package versions, model revision, and dataset revision in each run manifest.

Clone the inspection sources [2], [3] and [4] beside the main checkout. Run these commands from the `dynamic-state-anchoring` repository root:

```bash
mkdir -p ../dsa-sources
git clone https://github.com/UW-Madison-Lee-Lab/looped-tf.git ../dsa-sources/looped-tf
git clone https://github.com/zhenyi4/codi.git ../dsa-sources/codi
git clone https://github.com/openai/grade-school-math.git ../dsa-sources/grade-school-math
git -C ../dsa-sources/looped-tf rev-parse HEAD
git -C ../dsa-sources/codi rev-parse HEAD
git -C ../dsa-sources/grade-school-math rev-parse HEAD
```

The raw GSM8K files in [4] support annotation development. Benchmark training uses the LOTUS splits specified below. Its data dependency can also be inspected at the pinned source revision [12]:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/da03/Internalize_CoT_Step_by_Step.git \
  ../dsa-sources/internalize-cot
GIT_LFS_SKIP_SMUDGE=1 git -C ../dsa-sources/internalize-cot checkout --detach \
  e06a32ee5e4cd117171daeb4755d2a97ece62761
```

This inspection clone skips large Git LFS downloads. Obtain the benchmark data through LOTUS's download script below; do not parse LFS pointer files as training data.

### Component map

| Source | Reuse or inspect |
| --- | --- |
| LOTUS [1] | `scripts/lotus.py` for latent feedback and forward/generation structure; `scripts/dataset.py` for data and collation; `scripts/run.py`, `scripts/eval.py` and `args/base/` for harness conventions. |
| Looped Transformers for Length Generalization [2] | `src/train.py`, `src/conf/addition.yaml`, `src/conf/parity.yaml` and `src/conf/multi.yaml`; inspect the model and task implementations after checkout. Reuse task generators and length-split conventions. |
| CODI [3] | `src/model.py` for explicit/latent auxiliary learning, stop-gradient hidden targets and alignment-loss conventions. Keep the two routes at the same width and size. |
| GSM8K [4] | Original `question`/`answer` records, calculator equations and the `####` final-answer separator. |

These are extraction points, not a request to merge three repositories or environments. LOTUS and looped-tf publish different environment recipes; port only the required components into one pinned environment and retain upstream licenses. Existing LOTUS is a looped padded architecture, not the exact anchor-memory wrapper specified below. An adapter and model unit tests are required.

### Initialization

Start the natural-language pilot from `yingfanbot/gsm-cot-gpt2` [5], using the same checkpoint in all looped arms. The published `yingfanbot/gsm-cot-llama1b` is a later scale-up option; compare methods at the same backbone size within each experiment. Run an unmodified checkpoint inference smoke test before modifying it.

Adding `<SA>` requires resizing the embeddings/output vocabulary, preserving any weight tying, and training the new embedding row. Do not freeze the new row. Any freezing or adapter policy must be identical across compared arms.

## 3. Deterministic milestone annotation

Identify isolated-variable assignments and closed arithmetic conclusions, since GSM8K frequently uses calculator annotations without named variables [4]. Treat these as weak supervision for a semantic milestone, with explicit evidence levels.

| Rule | Example | Evidence and behavior |
| --- | --- | --- |
| R1: calculator equality | `<<8+4=12>>` | Evaluate the numeric expression exactly and compare with its scalar result. Mark after `>>` when valid. |
| R2: bare numeric equality | A whole line `8 + 4 = 12.` | Same arithmetic check. Mark after the numeric conclusion, before terminal punctuation. |
| R3: isolated variable | A whole line `x = 12.` | Mark as `structural_only`. Its syntax concludes a value, but the tagger has not proved that value from the problem. |

V1 deliberately requires whole-line R2/R3 statements. It does not search arbitrary prose substrings for `x = value`, which could accept a hypothetical condition. Calculator wrappers may occur anywhere in the rationale. Expand coverage only after auditing precision.

### Acceptance policy

Default `conclusive` includes valid R1/R2 and structural R3. The `verified_only` policy includes R1/R2 only. Report both policies as an annotation sensitivity analysis. “Arithmetic verified” means only that the local equation is numerically consistent; it does not establish a correct solution or a sufficient checkpoint.

Support bounded numeric `+ - * /`, parentheses, unary signs, decimals, fractions and well-formed thousands separators. The reference normalizes mathematical multiplication/division/minus symbols only for evaluation. It uses an AST whitelist and exact fractions; it never uses `eval` or executes dataset text.

Reject wrong arithmetic, division by zero, functions, code, symbolic expressions, powers, percentages, inequalities, ambiguous chained equalities, and prose conditions in V1. Unsupported examples stay in the dataset with diagnostics; they are not automatically discarded.

### Final conclusions and unchanged source text

- Split raw GSM8K solutions at the single `####` separator. Scan only the rationale, never the question or final-answer field.
- By default, exclude the last accepted conclusion if its numeric value equals the final answer. This conservative `terminal_answer_match` policy can miss a legitimate intermediate with the same value; record that limitation.
- The final-value check is preprocessing for training labels only. It does not place the answer into the latent inference input.
- Insert exactly one `<SA>` at each accepted character end. Preserve all other characters, including original calculator wrappers and repeated result values.
- Removing inserted markers must reproduce the original rationale exactly.
- Keep separate raw and marked text, source hashes, event spans, rule IDs, values, evidence levels and rejection reasons.
- Do not use newline, equals sign, “Therefore,” or every sentence boundary alone as a positive label.

### Synthetic example

```text
Original rationale:
First, <<8+4=12>>12 marbles.
x = 12.
Then, <<12*2=24>>24 marbles.
#### 24

Marked intermediate rationale:
First, <<8+4=12>><SA>12 marbles.
x = 12<SA>.
Then, <<12*2=24>>24 marbles.
```

The final calculation is withheld by the terminal-answer policy. The two retained events have different evidence levels. The marker is a training annotation, not a value that is injected into the inference path by the preprocessing code.

### Default benchmark: mirror LOTUS's supplied splits

Use the same problem membership and augmented rationales as the unmodified LOTUS baseline. From the main repository root, run:

```bash
bash preprocessing/gsm_icot.bash
```

LOTUS downloads `data/gsm8k/train.txt`, `valid.txt` and `test.txt` from Internalize_CoT_Step_by_Step at commit `e06a32ee5e4cd117171daeb4755d2a97ece62761` [12]. Its conversion script produces `data/gsm_train.json`, `data/gsm_valid.json` and `data/gsm_test.json`; `args/base/datasets/gsm8k.yaml` points to these files. Confirm that the pinned LOTUS checkout still uses this revision; resolve any discrepancy before preparing runs.

Keep these source JSON files unchanged and create annotated derivatives. Do not generate a new random validation partition for the primary LOTUS comparison. Tune checkpoint selection, gate threshold and loop budgets on `gsm_valid.json`, then evaluate once on `gsm_test.json`. All rationale variants retain their source partition, including records with no accepted anchors.

The inspected standalone `scripts/eval.py` also reads local `data/gsm_test.json` for `--datasets gsm8k`. Verify that path in the actual checkout and give DSA the identical ordered question/answer list. Do not silently substitute a different Hugging Face split or assume test membership matches the official release; compare question hashes and answer normalization explicitly.

Implement a split manifest containing the data-source commit, LOTUS SHA, preprocessing-script SHA256, original-file SHA256s, row and unique-question counts, stable original-problem groups, and ordered evaluation IDs. Audit group overlap across train/valid/test. If supplied splits overlap, stop the comparison and report the discrepancy rather than quietly resplitting. Evaluate every test problem; annotation coverage must not determine the evaluation denominator.

LOTUS JSON uses `question`, `steps` and `answer`. The reference annotation CLI consumes JSONL, so its dataset adapter must convert each JSON-array row without changing its content or partition. The inspected `gsm_icot.py` splits rationale text on spaces: `steps` can be token fragments. Reconstruct with `--steps-join space`; use `newline` only for a separately verified line-step dataset.

Audit anchor coverage on the actual augmented traces before a training run. They may lack calculator wrappers and whole-line equations accepted by V1. Keep unsupported traces and report their masked auxiliary labels. A parser extension must have a declared deterministic rule and a precision audit. Never copy raw-solution event offsets onto a different augmented rationale.

### Raw-data annotation development

Use `openai/gsm8k`, configuration `main`, or the original JSONL files [4] for parser development and a separately named raw-data experiment. Group any development partition within official training data; keep official test data untouched for selection. This track does not replace the LOTUS-mirrored benchmark. Changing the rationale source or split changes the experimental condition and must be reported.

MATH is a later adapter, not a V1 dependency. Its LaTeX and boxed answers need different parsing; do not quietly apply GSM8K's simple rule set to it.

## 4. Dataset and tokenizer contracts

### Record shape

The reference emits JSONL with the following contract; actual span numbers and hashes are derived, not hand-written:

```json
{
  "schema_version": "dsa_rules_v1",
  "sample_id": "stable id for this rationale",
  "source_problem_id": "shared id for one original problem",
  "question_hash": "normalized-question hash",
  "original_record": {},
  "question": "unchanged question",
  "rationale_raw": "unchanged rationale before final delimiter",
  "rationale_marked": "same text plus inserted markers",
  "final_answer": "24",
  "policy": {
    "marker_token": "<SA>",
    "mode": "conclusive",
    "exclude_terminal": true,
    "steps_join": "space"
  },
  "events": [
    {
      "ordinal": 0,
      "start": 7,
      "end": 17,
      "text": "<<8+4=12>>",
      "rule": "calculator_equation",
      "validation": "arithmetic_verified",
      "variable": null,
      "value": "12"
    }
  ],
  "rejected_candidates": [],
  "event_count": 1,
  "original_sha256": "hash of untouched input record"
}
```

The example is a schema illustration, not a fixture with authoritative offsets. Use the executable example output for real offsets. Offsets are half-open Python Unicode character positions in `rationale_raw`, not UTF-8 byte positions or token indices. The marker string is part of the annotation policy and cache identity; rebuild from `original_record` if that policy differs.

Alternate rationales must have distinct `sample_id` values while sharing their original-problem group. The reference uses a normalized question hash when no original ID is available; that is a fallback, not a guarantee against paraphrased duplicates.

### Token handling

Use `<SA>` as the anchor annotation token. The inspected LOTUS latent-token setup uses `<|start-latent|>`, `<|end-latent|>` and `<|latent|>`; it does not define `<SA>`. Token availability must still be checked in the actual initialized tokenizer and source records.

Inspect `get_vocab()` and `all_special_tokens` before registration. If `<SA>` is already assigned an unrelated meaning, stop and resolve the collision. Otherwise register it once as an additional special token with normalization and whitespace stripping disabled. Assert it encodes as one distinct token and its ID differs from existing control tokens. Reject literal `<SA>` occurrences in original questions, rationales or answers. Save its string/ID, tokenizer and resized model together [6].

Tokenize the full marked explicit sequence once. Do not assume separately tokenized question/step fragments concatenate to the same BPE sequence. Discover marker positions by token ID. The checkpoint target is the final-layer hidden state immediately BEFORE each marker, which has already processed the conclusion but cannot causally see future rationale.

Build a separate tokenization of the raw, unmarked rationale with offsets. Map each accepted character end to its 1-based ending token position. Assert sorted ends and no mapping to a padding position. These raw positions are used only for the training schedule below.

### Split and audit requirements

For the default `lotus_mirror` regime, preserve the supplied train/valid/test membership and create the split manifest before annotation. Group by original problem ID; audit cross-partition overlap and separately compare the test questions with the official release. Keep all augmentations of a problem in their original partition. A normalized question hash detects exact/format duplicates, not every semantic paraphrase. Only the separately named `raw_annotation_dev` regime constructs its own training-only development split.

Report record counts, no-event rate, events per record, R1/R2/R3 counts, terminal exclusions, rejection reasons, tokenizer mapping failures and controller-label overflow. Human-audit a reproducible sample before training. Parser precision and coverage have not been measured for the full corpus in this handoff.

## 5. Same-backbone latent model

### State and layout

Use one causal LM backbone B_theta for both the explicit supervised route and latent route. The two routes share weights; stop-gradient is applied only to explicit hidden targets, while the explicit cross-entropy loss still trains the backbone.

For a correctness-first implementation, use a fixed-size workspace Z and one persistent anchor token A. Defaults: workspace `S=4` tokens, anchor `m=1`, training loop budget `K=16`. These are starting settings, not validated hyperparameters.

At loop k, rebuild a fresh input embedding sequence:

```text
[question embeddings X | anchor A_k | workspace Z_k]
```

The question stays unchanged and accessible. Initialize A_0 to a learned null anchor and Z_0 to learned workspace vectors. Keep their positions fixed across loops. Use an attention mask and explicit position IDs that correctly handle different question lengths. Question padding must not become an anchor or workspace position.

### Update equations

```text
Z_next = LN(B_theta([X, A_k, Z_k]) at workspace positions)
u      = mean_pool(Z_next)
p_k    = sigmoid(gate_head(u))
C_k    = anchor_projection(u)
A_next = (1 - write_k) * A_k + write_k * C_k
Z_k+1  = Z_next
```

C_k is the candidate checkpoint. Keep its hidden width identical to the backbone. One scalar gate controls the one anchor in V1; multi-slot routing is a later study.

This is gated memory writeback, not a guaranteed state reset. V1 does not reinitialize Z at a write. Any later workspace re-encoding must be a separately named ablation so its effect is not conflated with anchoring.

### Discrete gate and gradients

Primary control uses threshold `tau=0.5`:

```python
p = torch.sigmoid(gate_logit)
hard_write = (p >= tau).to(p.dtype)
write = hard_write + p - p.detach()  # training only
anchor = (1 - write) * anchor + write * candidate
```

At inference, use only `hard_write`. This is a biased straight-through gradient estimator [7], not exact differentiation through the threshold. Use BCEWithLogitsLoss on `gate_logit` for direct supervision. A soft-interpolation control is a declared sensitivity ablation; do not silently change the primary gate between training and evaluation.

Do not force a write at an annotated milestone during the latent training forward. Gold labels supervise the loss; the predicted gate determines writes. Oracle writes belong in a separately labeled diagnostic.

### Backbone adapter and decoding

Implement `BackboneAdapter.forward_embeds` that returns final hidden states without materializing vocabulary logits at every loop. Inspect GPT-2's transformer body and Llama's model body rather than assuming their APIs coincide.

V1 uses `use_cache=False` during recurrence and full recomputation. This is slower but avoids stale KV states when anchor/workspace embeddings change. Prefix-only cache optimization may be added after numerical parity tests; changing memory inputs invalidates caches dependent on them.

After K loops, rebuild the prefix using the latest A_K/Z_K, append the existing answer prefix `\n### `, and decode an answer. Do not reuse an obsolete recurrent cache. The plain explicit route uses the same answer formatting as the initialized CoT checkpoint. Mask prefix tokens out of answer loss.

The latent inference API accepts question token IDs, attention mask and a configured loop budget. It does not accept gold rationale, markers, event count, target states or answers. The write head predicts control internally; no external oracle inserts runtime markers.

## 6. Training labels, alignment and losses

### Explicit route

Use teacher forcing on:

```text
question + newline + marked rationale + newline + "### " + answer + EOS
```

Mask the question in the explicit language-model loss. Collect final-layer hidden states just before the inserted markers as checkpoint targets T_j. Ensure causal attention. Detach T_j for checkpoint alignment, but retain gradients for explicit CoT cross-entropy.

These same-model auxiliary targets provide a training signal. Hidden similarity does not prove that an anchor contains a sufficient computational state.

### Text boundary to recurrent depth: an explicit V1 heuristic

A text boundary is not a loop index. The initial implementation uses a deterministic monotonic schedule so the coding agent has a concrete, testable contract.

For M accepted events, raw rationale token length L, fixed loop budget K, and ending token positions b_j (1-based, unmarked):

```text
ideal_j = ceil(K * b_j / L)
k_0     = 0
k_j     = min(K - (M - j), max(k_(j-1) + 1, ideal_j))
```

Assign positive controller targets at k_j and zero targets at other supervised loops. This maps relative text progress to depth; it does not establish one-loop/one-semantic-step equivalence. Compare it later with an evenly spaced milestone schedule using the same M/K.

If M=0 or M>K, mask controller and checkpoint losses for that record, retain its answer/explicit losses, and report coverage. Do not silently merge events, drop the record, or expand its inference budget based on M. The provided `map_events_to_loops` implements this contract.

For controlled algorithmic tasks, exact solver states can provide an independently specified semantic schedule. Keep that experiment distinct from the text-progress heuristic.

### Joint objective

```text
L = L_answer
  + lambda_cot * L_explicit_marked_cot
  + lambda_gate * masked_BCE(gate_logits, gate_targets)
  + lambda_anchor * mean_j D(C_(k_j), stop_gradient(T_j))
```

Default D is MSE after non-affine layer normalization of both vectors. Align candidate C, not only the stored A, so all write-policy arms receive the same auxiliary checkpoint supervision even when their actual writes differ.

Start coefficients at `lambda_cot=1.0`, `lambda_gate=0.1`, `lambda_anchor=0.1`; tune on validation with equal trial budgets across arms. Report individual losses and gradient norms. If BCE positives are rare, compute a clipped positive weight from training labels only and use it identically in all arms.

The matched B0/B1/B2/DSA controls instantiate the same gate, writer and memory modules and receive the same marked explicit-route, gate and candidate losses. Their write policy is the controlled variable. Retain an unmarked explicit CoT-SFT baseline as a separate reference; its token generation cost differs and must be reported.

Cache annotation offsets and tokenized labels freely. Do not cache explicit hidden targets across optimizer updates when the shared backbone is changing. A frozen same-size reference route would be a later, declared variation.

## 7. Implementation order and interfaces

### Stage A: corpus preparation

1. Pin source revisions, reproduce the LOTUS data files and audit the original-problem split manifest.
2. Adapt the mirrored LOTUS JSON arrays to JSONL and reconstruct their augmented rationale text without changing source partitions.
3. Run both annotation policies and save coverage reports.
4. Add the marker to the tokenizer and test exact token/offset alignment.
5. Write tokenized JSONL/Arrow artifacts plus hashes and a provenance manifest.

The bundled command is already executable:

```bash
python3 scripts/annotate_milestones.py \
  --input examples/synthetic_gsm8k.jsonl \
  --output examples/annotated_gsm8k.jsonl \
  --report examples/annotation_report.json
python3 -m unittest discover -s tests -v
```

The CLI accepts JSONL. Convert upstream JSON arrays explicitly before running it. It rejects malformed records with their input line number instead of silently writing a partial corpus. Use `--mode verified_only` for the stricter annotation policy.

### Stage B: model correctness before scale

Implement the adapter and controller on a tiny deterministic CPU model first. Verify immutable question inputs, anchor persistence, candidate writes, nonzero controller gradients under straight-through training, and independent per-example memory.

Then overfit a small synthetic set and a small GSM8K training subset. This is an implementation smoke test, not evidence of generalization. Profile parameter counts, forward/backward memory and throughput before selecting paid compute.

### Stage C: controlled training

Start every looped arm from the exact same initialization, including new embeddings, workspace/anchor vectors and control heads. Use identical data order, optimizer, precision, loop budgets and losses. Preserve optimizer/scheduler/RNG state when resuming.

Run at least three seeds for the primary comparison once the pilot works. Add distributed training and KV-cache optimization only after single-device correctness and evaluation parity are established.

### Stage D: evaluation and packaging

Implement question-only generation, numeric answer scoring, traces of p_k/write decisions, and checkpoint save/reload. Save config, tokenizer, model weights, controller/writer weights, workspace initialization, provenance, split manifest and run metrics together.

### Proposed modules to implement

| Module | Required contract |
| --- | --- |
| `data/adapters.py` | Canonical source records and group IDs; no model code. |
| `data/tokenize.py` | Marker IDs, token offsets, masks and raw-rationale event ends. |
| `models/backbone.py` | Embedding/hidden-state/LM-head adapter for the selected checkpoint. |
| `models/anchored_loop.py` | Shared recurrent body, A/Z state, write modes and question-only inference. |
| `training/objectives.py` | Same-backbone explicit route, detached targets and masked losses. |
| `training/trainer.py` | Fixed-budget training, seeds, resume and per-loss logging. |
| `evaluation/evaluate.py` | Question-only predictions, answer scoring and resource profiles. |
| `evaluation/intervene.py` | Anchor-content and write-timing causal diagnostics. |
| `tests/` | Annotation, tokenizer, memory, gradient, leakage and reload checks. |

Names above are the proposed project API. They are not existing upstream commands or implemented model modules in this handoff.

## 8. Experiments that isolate the anchor

| Arm | Actual write policy and interpretation |
| --- | --- |
| B0: static anchor | A remains at its initial learned value. Receives identical milestone/gate/candidate supervision. Primary control for extra supervision. |
| B1: every-loop write | Always replace A with C. Tests whether selective timing beats ordinary feedback. |
| B2: periodic write | Write every four loops, fixed independently of each sample's milestones. Tests learned timing against scheduled updates. |
| DSA: learned write | Threshold the learned gate. Main proposed method. |
| Unmodified LOTUS | Upstream looped architecture trained at the same backbone size and CoT initialization on the mirrored data. External method comparison. |
| Oracle timing | Write at gold scheduled k_j during diagnostic evaluation. Uses labels and is not deployable accuracy. |
| Explicit CoT-SFT | Standard visible-rationale answer model at the same base size. Reference accuracy/resource comparison. |

An optional oracle-content experiment may construct checkpoint representations from gold prefixes. It is a privileged upper-bound diagnostic, not a valid question-only result.

First establish a signal on addition and multi-hop function composition with exact solver states; parity can be a lightweight pipeline check but should not be the only generalization result. Specify original problem difficulty and expected computational state independently of serialized text length.

Natural-language evaluation uses GSM8K, then SVAMP/GSM-Hard/MultiArith if their provenance and interfaces are verified. Such OOD sets do not by themselves establish reasoning-depth extrapolation.

### Direct comparison with LOTUS

Run an unmodified LOTUS baseline from the same backbone and Stage-1 CoT initialization as DSA, using the pinned train/valid/test files above. Published 3B results are context for a GPT-2 pilot, not its matched baseline. Match training-example exposure and document any optimizer or schedule differences required by each method.

Use the same ordered test IDs, question formatting, answer prefix, answer extraction/normalization, decoding rule and answer-token cap. The inspected LOTUS evaluator defaults to `max_new_tokens=64`; pin and record the actual evaluation settings rather than relying on defaults. Save predictions for paired comparison and report invalid-answer rates with the accuracy denominator.

A LOTUS loop processes a different number of latent positions from the DSA workspace, so equal loop counts do not imply equal compute. Report workspace/latent-token counts, iterations, measured latency, peak memory and total training cost. Include an accuracy-versus-measured-resource comparison. Use B0/B1/B2 to isolate anchor writes; LOTUS measures competitiveness of the full method.

### Measurements

- Exact final-answer accuracy and failure categories.
- Accuracy versus predeclared loop budgets, initially `4, 8, 16, 32`.
- For controlled tasks: train-depth versus test-depth curves, with input-length and distractor controls.
- Inference latency, throughput, peak memory and executed recurrent forwards.
- Number of writes, gate calibration to the annotation heuristic, and candidate-state error.
- Annotation coverage, controller coverage, and all exclusions/truncations.
- Training cost of the explicit auxiliary route, reported separately from deployment cost.

Choose thresholds and checkpoint selection using validation only. Inference budget must not depend on gold step count. Use all official test examples for accuracy; event-conditioned diagnostics are a separately labeled analysis.

### Causal diagnostics

Alter anchor content at a matched computational stage while preserving its scale and progress context. Test whether continuation changes in the expected direction. Also compare disabled writes and shifted write times. Avoid treating “a probe can decode the value” as sufficient evidence that the model uses it.

If DSA only beats a baseline without the same milestone supervision, the result does not isolate anchoring. If periodic writes match learned writes, the supported claim is scheduled refreshing rather than semantic controller timing. If larger loop counts worsen all arms, report the limitation instead of selecting each test example's best depth retrospectively.

## 9. Acceptance checklist for the local agent

### Already checked in the reference package

- Exact arithmetic, decimals/fractions/signs and grouped numbers.
- Rejection of unsupported expressions, code, conditions and unsolved equations.
- Final-answer exclusion, unchanged question/answer and source span preservation.
- Idempotency, alternate-policy rebuilds and augmented-rationale group IDs.
- Strictly ordered loop labels, collisions, empty traces and overflow masks.
- CLI processing of eight synthetic examples: six retain intermediate events.

### Required before the first meaningful model run

1. Annotator and tokenizer agree on every marker and conclusion endpoint.
2. `<SA>` is one distinct token with no source-text collisions; checkpoint reload preserves tokenizer and vocabulary tying.
3. Batch and unbatched forwards agree when padding is varied.
4. Closed gates leave A unchanged exactly; writes affect only the intended example.
5. Straight-through gate receives gradients; target hidden states are detached for alignment.
6. Changing a training target answer/rationale cannot change question-only inference inputs.
7. Gold marker count, schedule or hidden states never enter the production inference API.
8. Disabling writes reproduces the matched static control, apart from expected unused parameters.
9. Candidate supervision remains present in every looped arm.
10. Recurrent KV caches are disabled; any optimized version passes full-recompute parity.
11. Save/reload and resume reproduce the expected outputs and state.
12. No-event and over-budget examples retain answer supervision and appear in coverage reports.
13. DSA and unmodified LOTUS use identical test IDs and scoring; their source-file hashes match the benchmark manifest.

### First local-agent instruction

Read this document and run the bundled reference checks. Inspect the pinned upstream model/data code. Implement only data/tokenization and the tiny-model anchored loop first. Demonstrate a question-only inference smoke test and matched B0/DSA training paths before adding distributed training, larger checkpoints, MATH, or extra objectives.

Maintain a short implementation-status file listing completed modules, observed failures, exact validation commands and remaining work. Do not report the preprocessing tests as a completed model pipeline or claim research gains without experiments.

## 10. Open scientific choices and claim boundaries

The text-progress loop schedule is an operational first choice, not a discovery about how latent computation maps to teacher text. Treat schedule sensitivity and annotation policy as part of the experiment.

Milestone checkpoint targets may omit unresolved goals or progress information. Controlled solver-state experiments should test content and progress separately. A learned write can preserve an error; no runtime verifier is present.

Anchors, gated memory and recurrent reasoning already have related precedents [8-10]. The novelty case depends on the exact controller, supervision and causal results. Accurate hidden-state alignment or additional loop depth alone is insufficient.

The minimum useful outcome is a reproducible difference between learned and matched static/periodic controls, with usable checkpoint states and honest cost/depth limits.

## 11. References and provenance

References are primary sources. Repository paths were inspected through the upstream README and available code snapshots; the coding agent must resolve actual checkout SHAs and installed dependency versions.

[1] Fan, Svete and Lee. LOTUS / Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers. https://arxiv.org/abs/2606.31779
Code: https://github.com/yingfan-bot/lotus

[2] Fan, Du, Ramchandran and Lee. Looped Transformers for Length Generalization, ICLR 2025. https://arxiv.org/abs/2409.15647
Code: https://github.com/UW-Madison-Lee-Lab/looped-tf

[3] Shen et al. CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation. https://arxiv.org/abs/2502.21074
Code: https://github.com/zhenyi4/codi

[4] Cobbe et al. GSM8K / Training Verifiers to Solve Math Word Problems.
Data: https://huggingface.co/datasets/openai/gsm8k
Original files and calculator-format description: https://github.com/openai/grade-school-math

[5] LOTUS CoT initialization checkpoints.
https://huggingface.co/yingfanbot/gsm-cot-gpt2
https://huggingface.co/yingfanbot/gsm-cot-llama1b

[6] Hugging Face tokenizer special-token and offset APIs.
https://huggingface.co/docs/transformers/main_classes/tokenizer

[7] Bengio, Leonard and Courville. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.
https://arxiv.org/abs/1308.3432

[8] Liang, Jin and Pan. Beyond the Training Horizon: Mechanisms and Limits of Length Generalization in Looped Transformers, 2026 preprint.
https://arxiv.org/abs/2609.33144

[9] Kim et al. Looped Transformers with Source-Centered State Evolution, 2026 preprint.
https://arxiv.org/abs/2607.27656

[10] Bulatov, Kuratov and Burtsev. Recurrent Memory Transformer, NeurIPS 2022.
https://arxiv.org/abs/2207.06881

[11] Kuo et al. Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping, 2026 preprint.
https://arxiv.org/abs/2606.29983

[12] Internalize_CoT_Step_by_Step: pinned augmented GSM8K data used by LOTUS preprocessing.
https://github.com/da03/Internalize_CoT_Step_by_Step/tree/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k
LOTUS download/conversion script: https://github.com/yingfan-bot/lotus/blob/main/preprocessing/gsm_icot.bash
