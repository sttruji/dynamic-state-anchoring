# Dynamic State Anchoring: exploratory findings

Compiled 14 October 2026. Status: **current implementation parked; scaling unchanged is not justified**.

The experiments found no demonstrated validation advantage from learned anchor writes. FLT-inspired recurrence substantially improved gradient conditioning and fitting of a tiny training set, but the learned controller did not reliably distinguish milestone events. The larger-data experiment had poor validation accuracy, nearly stopped writing, and showed reduced explicit-route performance after fine-tuning in a local follow-up. These results support stopping compute expenditure on this version.

This is a negative result for the tested implementation and budgets. It does not establish that every possible latent anchoring design is ineffective. The intended selective-write mechanism never learned its target behavior reliably, so the broader hypothesis remains unresolved. That uncertainty provides no evidence-backed reason to scale this version.

The accompanying [machine-readable summary](../provenance/findings_summary.json) records exact metrics, experimental identities, paired counts, evidence hashes and runtime details. This document reports completed experiments; it does not propose an automatic next run.

## Experimental scope

All pretrained experiments used `yingfanbot/gsm-cot-gpt2` at revision `e0ef1887404adbf6c531d2cac7804694321f87bd`, seed 7, four workspace vectors and one anchor vector. They retained the pinned LOTUS augmented GSM8K partitions from data revision `e06a32ee5e4cd117171daeb4755d2a97ece62761`. Source quality exclusions were not applied. No-event records remained in the answer losses with their auxiliary supervision masked.

Within each training experiment, arms shared the initialization, data, actual sample order, parameter modules and optimization settings, except the declared write policy and, in the K=8 architecture comparison, recurrence mode. Common settings were AdamW with learning rate 3e-5, weight decay 0.01, FP32, gradient checkpointing and global gradient norm clipping at 1. The joint objective was answer CE + explicit-CoT CE + 0.1 controller BCE + 0.1 normalized candidate alignment. Positive controller labels had weight 1.

The explicit teacher used the shared backbone through the ordinary language route. Milestone targets were detached hidden states before the `<SA>` marker. There was no separately frozen teacher and no CODI answer-boundary alignment objective. Inference received the question and configured computation budget; gold milestones, answers and write schedules were not supplied as triggers.

`static` means no anchor writes; it still has a changing workspace and a learned initial anchor. `every_loop` always writes, `periodic` writes at loops divisible by four, and `learned` uses threshold 0.5 with a straight-through training estimator. All policies retain the same candidate and controller supervision.

The FLT-inspired adaptation resets the current residual stream each loop and uses previous memory vectors as attention queries at every GPT-2 block, with current-layer keys and values. It persists only the five memory vectors. This adapts [Fu et al., *Simply Stabilizing the Loop via Fully Looped Transformer*](https://arxiv.org/html/2605.18797v1), rather than reproducing the paper's full architecture or pretraining setup. Implementation details are in [K8_FLT_PILOT.md](K8_FLT_PILOT.md).

## 1. Small standard-recurrence pilot, K=16

The completed review covers 32 training questions, 200 updates per arm and the first 128 original validation questions. Static was reused and learned was retrained after checkpoint loss; the learned loss/write/gradient journal reproduced its original values. The table uses that completed review, rather than the earlier 32-question validation smoke output. Earlier synthetic CPU tests established plumbing only.

| Policy | Training correct, all 32 | Validation correct | Invalid validation outputs under scorer | Median training gradient norm before clipping |
|---|---:|---:|---:|---:|
| Static | 6/32 (18.75%) | 3/128 (2.34%) | 7/128 | 243,570 |
| Learned | 4/32 (12.50%) | 3/128 (2.34%) | 17/128 | 654,291 |

There was no observed accuracy advantage. Four-question answer-gradient probes increased from median norms of approximately 251/318 at K=1 to 2.41/2.50 million at K=16 for static/learned. Every training update exceeded the clipping threshold. Gradients remained finite, but their amplification exposed a serious conditioning problem.

The learned controller's event AUC was 0.510. Across supervised questions it wrote on 58.6% of loops against a target rate of 13.4%; across all 32 diagnostic questions it wrote on 61.3%. Candidate/teacher alignment was weak. Memory and depth interventions changed predictions without establishing a reliable accuracy benefit.

**Supported conclusion:** the unchanged K=16 implementation was unsuitable for a larger run; learned anchoring had no demonstrated benefit.

## 2. Small K=8 recurrence ablation

Four conditions trained on the same 32 questions for 200 updates, with 128-question validation. Initialization and settings were matched within this experiment.

| Recurrence / write policy | Training correct, all 32 | Validation correct | Median training gradient norm before clipping |
|---|---:|---:|---:|
| Standard / static | 6/32 (18.75%) | 1/128 (0.78%) | 7,111 |
| Standard / learned | 4/32 (12.50%) | 6/128 (4.69%) | 19,193 |
| FLT / static | 29/32 (90.63%) | 3/128 (2.34%) | 80.7 |
| FLT / learned | 31/32 (96.88%) | 2/128 (1.56%) | 68.7 |

FLT reduced median training gradient norms approximately 88-fold for static and 279-fold for learned, while fitting nearly all training answers. A separate one-question initialization screen also showed much lower K=8/K=16 gradients with identical initial weights, avoiding the confounding effect of differently fitted checkpoints. This supports an optimization improvement on the observed probes. Almost every FLT training update still required clipping.

Validation did not establish an FLT or learned-write advantage. The FLT learned controller had AUC 0.507 and wrote on 41.8% of loops across all training questions. Forcing writes every loop preserved all eight extracted answers in its intervention subset. Some memory removals harmed those seen-example answers, but that demonstrated sensitivity rather than useful selective timing.

**Supported conclusion:** FLT improved conditioning and tiny-set fitting. Successful memorization was an engineering milestone, not validation of DSA's central hypothesis.

## 3. Bounded FLT data-scale experiment

The next experiment fixed FLT at K=8 and compared four write policies on 512 unique training questions, with 1,024 updates per arm: two exposures per selected record. It evaluated all 500 original validation questions. Expensive training diagnostics used the same fixed seeded sample of 64 questions across policies, rather than the full training set. The frozen protocol is in [FLT_DATA_SCALE.md](FLT_DATA_SCALE.md).

| Write policy | Sampled training correct, 64 of 512 | Full validation correct | Median training gradient norm before clipping |
|---|---:|---:|---:|
| Static | 20/64 (31.25%) | 23/500 (4.60%) | 30.42 |
| Every loop | 24/64 (37.50%) | 20/500 (4.00%) | 27.80 |
| Periodic | 20/64 (31.25%) | 17/500 (3.40%) | 29.24 |
| Learned | 19/64 (29.69%) | 14/500 (2.80%) | 29.66 |

No writes had the highest observed validation score, but the policy differences were uncertain. For static versus learned, 15 questions were correct only for static, six only for learned, and eight for both; the exploratory exact paired two-sided p-value was 0.078. None of the six policy comparisons reached 0.05 before adjustment. These results establish no learned-write advantage; they do not prove that its true effect is exactly zero or universally harmful.

Gradient and residual diagnostics remained finite, and every update still exceeded norm 1. Answer loss continued falling: first-epoch means around 3.02–3.10 became 2.06–2.18 in the second epoch. Two exposures may have been insufficient. The lower sampled training accuracy cannot be attributed to data size alone because the smaller pilot used different membership and approximately 6.25 exposures per record. More training could improve fitting, but no evidence shows that it would repair the controller or produce useful generalization.

### Controller failure

Learned writes fell from 91.2% of loops in updates 1–128 to zero in the final 384 updates. At deterministic inference it made no writes on all 64 diagnostic training questions. On validation, 492 questions had no writes and eight had one: **eight writes out of 4,000 loop decisions (0.2%)**.

The 51 supervised diagnostic questions had 108 positive and 300 negative loop labels. Learned event AUC was **0.513**, with mean positive/negative probabilities 0.338/0.334. At threshold 0.5, event recall was zero. Its 73.5% label accuracy merely equaled always predicting no write. Lowering the threshold would increase write frequency without improving this near-chance ranking.

Loss-gradient probes did not identify a consistent universal cause across experiments. Gate supervision dominated the head gradients in the small FLT pilot; answer gradients dominated the two larger-data probes, opposing supervision on one and aligning on the other. These snapshots do not justify attributing every failure to one loss weight.

**Supported conclusion:** more data did not demonstrate a usable learned anchor mechanism under the tested budget. The conditioned recurrence remained trainable, but answer generalization and event timing were poor.

## 4. Local inference follow-up

The four completed data-scale checkpoints were restored from their downloaded full backups, matched to the reviewed hashes, and evaluated locally on CPU. The first 64 original validation questions were fixed before execution. No optimizer state was loaded, no training occurred, and model-state hashes remained unchanged. Local K=8 outputs exactly reproduced all **256/256** Colab strings on these questions.

| Policy | K=1 | K=8 | K=8, zero final anchor | K=8, zero final workspace | K=8, zero both memories |
|---|---:|---:|---:|---:|---:|
| Static | 3/64 | 2/64 | 2/64 | 3/64 | 3/64 |
| Every loop | 2/64 | 4/64 | 3/64 | 3/64 | 3/64 |
| Periodic | 2/64 | 3/64 | 2/64 | 2/64 | 3/64 |
| Learned | 2/64 | 2/64 | 2/64 | 4/64 | 4/64 |

There was no consistent K=8 advantage. For learned writes, changing K=8 to K=1 changed 27 extracted answers but preserved the total of two correct. Zeroing the anchor changed one answer and preserved accuracy; zeroing both memories changed 21 answers and increased the count from two to four, with exploratory paired p=0.5. These small counts do not establish that removal improves the method. Memory affects outputs, but a reliable accuracy benefit was not demonstrated.

Removals affect only the final answer readout; they preserve the computed latent trajectory, positions and original question access. Depth comparisons use the same saved weights, not independently trained depth-matched models. Continuous vectors were not decoded as reasoning transcripts.

### Expanded explicit-CoT control

The same 64 validation questions were generated through the original checkpoint and each trained backbone's ordinary explicit route. Greedy generation used a 192-token cap, versus 64 for latent answers; KV caching was disabled. The first eight local controls exactly matched the earlier Colab token IDs for all five models.

| Explicit-route model | Correct | Accuracy | Original-correct answers lost / original-wrong answers gained |
|---|---:|---:|---:|
| Original pretrained checkpoint | 25/64 | 39.06% | — |
| After static training | 17/64 | 26.56% | 10 / 2 |
| After every-loop training | 15/64 | 23.44% | 11 / 1 |
| After periodic training | 13/64 | 20.31% | 13 / 1 |
| After learned training | 16/64 | 25.00% | 10 / 1 |

The original model answered more questions correctly than every fine-tuned explicit route on this subset. The learned arm's paired exploratory p-value was 0.0117; the other three were 0.0386, 0.00635 and 0.00183. These are unadjusted comparisons from one seed and one source-order subset. They strengthen the concern that fine-tuning reduced explicit capability, alongside weak transfer into latent answers. Different explicit/latent token budgets prevent an equal-compute superiority claim.

## Evidence quality, scoring and limits

| Completed review | Inventoried files verified | Prediction scores independently checked | Raw-token records independently decoded |
|---|---:|---:|---:|
| K=16 pilot review | 242 | 512 | 256 |
| K=8 recurrence comparison | 464 | 1,008 | 496 |
| FLT data-scale comparison | 518 | 2,560 | 560 |
| Local inference follow-up | Verified checkpoint model files | 1,600 | 1,600 |

Matched identities and actual sample sequences were checked within training runs. All stages in the three completed review bundles passed. These checks support engineering integrity; passing them does not establish research gains. The 230-test CPU suite checked correctness of the workflow and model implementation, not scientific efficacy.

The pinned LOTUS numeric extractor was retained throughout. It can assign incidental credit: the K=8 standard-learned validation score includes `iced.5` for gold 5, and the data-scale learned score includes `10*5` for gold 5. Reported primary scores preserve those decisions. A zero invalid-output rate means successful finite-number extraction, not a valid calculation or well-formed scalar answer. Local latent follow-up hits were scalar outputs.

All experiments used one seed and small training budgets. Validation informed development and the local subset was source-order, not a new independent benchmark. The held-out test remained reserved. Interventions can be outside the training distribution, alignment is a proxy, and the explicit teacher shares weights that change during training. Cross-experiment data membership, exposures and diagnostic denominators differ. Final alignment has no initialization trajectory supporting a learning-improvement estimate.

Review bundles exclude model weights. The local follow-up verified the downloaded checkpoint models against the recorded hashes and reproduced Colab outputs, but this does not establish current Drive durability. Public provenance contains aggregate metrics and hashes; raw local archives, weights, dataset text and private filesystem paths are not committed.

## Cost and reproducibility

Measured update time was approximately 15.0 minutes across the two completed K=16 arms, 18.4 minutes across the four K=8 recurrence arms, and 93.1 minutes across the four data-scale arms. K=8 and data-scale recorded stage spans were 84.1 and 196.5 minutes, including loading, checkpoint saves/archives, diagnostics, evaluation and gaps between stage completions. These totals do not isolate the cost of any single overhead component. The local follow-up took 15.5 minutes, excluding extraction, benchmarking and the original model download.

Recorded implementation references:

| Experiment | Code reference | Run identity |
|---|---|---|
| K=16 training / completed review | `ba4e69315086` / `847e61bcbb79` | `gpt2-pilot-v1-ba4e69315086` |
| K=8 recurrence comparison | `eb4d7b20a33f` | `gpt2-k8-flt-v1-eb4d7b20a33f` |
| FLT data-scale and local checkpoints | `a980b68bad7f` | `gpt2-flt-data-v1-a980b68bad7f-df4e456c` |

Set the notebook clone cell's `REPO_REF` to the relevant code reference to reproduce an implementation. The [pilot notebook](../notebooks/DSA_Colab.ipynb) and [data-scale notebook](../notebooks/DSA_FLT_DataScale.ipynb) remain available as experiment artifacts. Reproducing numerical results requires matching the recorded data, model, runtime and decoding configuration. Review ZIP hashes, per-arm metrics, source audit hashes and local verification metadata are retained in [findings_summary.json](../provenance/findings_summary.json).

## Final conclusion

**Park the current DSA implementation and do not scale it unchanged.** The experiments support an FLT-inspired optimization improvement, but provide no demonstrated benefit from learned anchoring. The controller fails to learn event timing, validation remains poor, extra depth has no consistent measured benefit on the local subset, and explicit performance declines after fine-tuning on that subset.

Any reopening should have a specific mechanistic change and a cheap, decisive comparison against unchanged no-write/every-loop/periodic controls, with preserved explicit capability. More parameters, more loops or longer training alone is not an evidence-backed next investment. The implementation and negative result remain documented for reproducibility; no broad impossibility or novelty claim is made.
