# Stage 1 results: extra depth adds nothing, and soft dedup holds accuracy at depth

Status, 3 November 2026: Stage 1 has run on SIM-CoT GPT-2, and its planned replication on CODI GPT-2 has run too.

- **The runs.** Both ran on a Colab A100 at commit `161751d`, with one seed (7), and were scored on all 500 `gsm8k_valid`
  questions.
  - SIM-CoT ran all eight arms.
  - CODI ran three: `ft6`, the reference, and `ft12` and `gate12_hard`, the two best 12-step arms on SIM-CoT.
- **The plan.** [PLAN.md](../results/latent_gate/stage1/PLAN.md) was fixed before either run.
- **The full reports.** They are at [SIM-CoT](../results/latent_gate/stage1/simcot/report.txt) and
  [CODI](../results/latent_gate/stage1/codi/report.txt). Each holds the planned decision rules and, in a separate section
  marked exploratory, what the gates kept and some extra paired comparisons.
- **Background.** The idea is in [LATENT_GATE_IDEA.md](LATENT_GATE_IDEA.md) and the literature in
  [LATENT_GATE_LITERATURE.md](LATENT_GATE_LITERATURE.md).

## The models in the tables

Every model starts from a published checkpoint trained with 6 latent steps: SIM-CoT GPT-2 or CODI GPT-2. Each fine-tuned
arm then trains for one more pass over the same 32,768 GSM8K-Aug problems, in the same order.

- **Base model.** The published checkpoint with no further training.
- **6-step model (`ft6`).** Fine-tuned with 6 latent steps and the full cache. It is the control for the extra fine-tuning
  itself.
- **12-step model (`ft12`).** Fine-tuned with 12 latent steps and the full cache.
- **Soft-trained gate (`gate12`).** Fine-tuned with 12 latent steps. A small network scores each latent entry when it is
  produced, and log g is added to every attention logit pointing at that entry. A penalty pushes the sum of g towards 6.
- **Top-6-trained gate (`gate12_hard`).** The same as `gate12`, except that during training each query sees only the 6
  highest-scoring earlier entries. A tie bug weakened this, as described under the caveats.

"No eviction" means every latent entry stays in the cache. "Hard eviction" means that once more than 6 latent entries exist,
the lowest-scoring one is deleted. Scores are fixed when an entry is made, so a deleted entry never returns, and later
steps and the answer see at most 6 latent entries however long the loop runs.

"Soft dedup" needs no training. It groups latent entries whose input vectors have cosine similarity of at least 0.95, and
it subtracts log(group size) from each member's attention logits. A group of k copies then gets the attention weight of a
single entry.

## Results

Correct answers out of 500. The number in brackets is out of the 173 problems with 4 or more reasoning steps.

**SIM-CoT GPT-2**

| Model | 6 steps | 12 steps | 24 steps |
|---|---|---|---|
| Base model, no eviction | 204 (29) | 185 (20) | 170 (19) |
| Base model, soft dedup (no training) | – | 207 (28) | 207 (27) |
| 6-step model, no eviction | **215 (31)** | 202 (26) | 186 (22) |
| 12-step model, no eviction | 211 (29) | 207 (27) | 196 (22) |
| 12-step model, soft dedup | – | 206 (28) | 207 (28) |
| Soft-trained gate, hard eviction | – | 201 (26) | 201 (26) |
| Top-6-trained gate, hard eviction | – | 207 (28) | 206 (28) |

**CODI GPT-2 (the replication)**

| Model | 6 steps | 12 steps | 24 steps |
|---|---|---|---|
| Base model, no eviction | 206 (30) | 192 (21) | 173 (13) |
| Base model, soft dedup (no training) | – | 202 (25) | 201 (25) |
| 6-step model, no eviction | **209 (30)** | 201 (26) | 180 (17) |
| 12-step model, no eviction | 200 (24) | **210 (26)** | 203 (25) |
| 12-step model, soft dedup | – | 182 (23) | 184 (23) |
| Top-6-trained gate, hard eviction | – | 174 (21) | 168 (21) |
| Top-6-trained gate, no eviction | – | 198 (25) | 191 (24) |

**The planned test fails on both models.**

- **The test.** A gated model must beat the 6-step model at 6 steps on the 4+ step problems at p < 0.025.
- **SIM-CoT:** the two gates score 26 and 28 against 31 (p = 0.23 and 0.38).
- **CODI:** the gate scores 21 against 30 (p = 0.035).
- **Every 12-step arm.** None beats the 6-step model on those problems, on either model, which fires the stop rule. The
  four SIM-CoT arms not shown tell the same story; their rows are in the report.

**The CODI replication differs from SIM-CoT in three ways.** These comparisons are exploratory.

- **The trained gate fails on CODI.**
  - With hard eviction it scores 174 at 12 steps and 168 at 24. That is far below the 6-step model's 209 (p = 1e-5), and
    24 answers below the same gate without eviction (198, p = 0.0015).
  - It learned a different rule than on SIM-CoT. CODI's latents alternate between two phases, and its gate mostly keeps
    entry 1 plus one phase: entries 1, 2, 4, 6, 8 and 10 in 66 questions, or 1, 3, 5, 7, 9 and 11 in 65. At 24 steps it
    often keeps late entries, such as 1, 15, 17, 19, 21 and 23.
  - The tie bug below is a likely confound, so this does not yet show that the gate idea fails on CODI.
- **Training at 12 steps helps CODI far more.** The 12-step model scores 210 at 12 steps and 203 at 24, against the 6-step
  model's 180 at 24 (p = 4e-4). It still loses 7 between 12 and 24 steps (p = 0.04). On SIM-CoT the same comparison at 24
  was 196 against 186 (p = 0.12).
- **Soft dedup hurts a CODI model trained at 12 steps.**
  - It scores 182 against 210 without dedup (p = 2e-4), while SIM-CoT's 12-step model is unaffected (206 against 207).
  - A model trained with repeated entries can come to rely on their multiplicity. Soft dedup is therefore a fix for running a
    checkpoint past the depth it was trained at, not a rule to apply to any model.

## How this reduces the accuracy lost at deeper depths

**The problem.** Past the trained 6 steps, both base models lose answers. By 24 steps SIM-CoT loses 34 (204 → 170) and
CODI loses 33 (206 → 173). The 6-step fine-tuned models lose 29 each: SIM-CoT goes from 215 to 186 (p = 2e-6) and CODI from
209 to 180 (p = 8e-5).

**The cause.** Earlier work traced this mostly to the cache rather than to the latent states:

- Each latent step appends one entry to the key/value cache, and the answer attends over all of them.
- SIM-CoT's latents stay stable and keep their question content, yet it loses as many answers as CODI.
- After 6 steps, appending exact copies of the last entry reproduces SIM-CoT's whole drop: 172 correct against 170.
- For CODI, copies explain part of the drop. Copying the last entry reproduces about a third of it (196 against 173), and
  copying the last pair about half (188). CODI's latents also lose question content as the loop runs on.
- The answer's attention share on latents 1–6, which carry the reasoning, falls from 0.099 to 0.036 in SIM-CoT and from
  0.158 to 0.033 in CODI.
- Duplicating a key k times adds log(k+1) to its attention logit, so the repeats take attention away from the entries that
  matter.

**What removes the drop.**

- **Soft dedup, with no training, on both models.** It cancels the log-count boost of repeated entries directly.
  - SIM-CoT scores 207 at 24 steps instead of 170 (p ≈ 6e-8), no different from its 204 at 6 steps (p = 0.58).
  - CODI scores 201 instead of 173 (p = 2e-4), no different from its 206 at 6 steps (p = 0.36).
- **A 6-entry budget at fixed weights (Stage 0).** Keeping the first 6 entries scores 204 on SIM-CoT and 205 on CODI at 24
  steps. Dropping near-duplicates scores 203 and 201.
- **The trained gate, on SIM-CoT only.**
  - The answer never sees more than 6 latent entries, so accuracy stops depending on loop length. The top-6-trained gate
    scores 207 at 12 steps and 206 at 24, against 186 for the 6-step model at 24 (p = 0.004).
  - It mostly keeps the earliest entries. Entries 1–3 are kept in almost every question, and exactly the first six in 196
    of 500.
  - It therefore rediscovered Stage 0's best rule, "keep the early entries", and it is no better than soft dedup (206
    against 207, p = 1).
  - On CODI the gate did not hold accuracy, as described above.

**What only reduces it.** Training at 12 steps moves the drop later. CODI's 12-step model loses 7 answers between 12 and 24
steps, and SIM-CoT's loses 11 (207 → 196, p = 0.013).

**What makes it worse.** Keeping the most recent 6 entries, or evicting at random, scores below no eviction at 24 steps:

- SIM-CoT's window-trained model with the recency window scores 165, and its random-eviction model with random eviction 172.
- On the base CODI model, the recency window scores 141 and random eviction 157.

The budget has to keep the early entries.

**What "reduces" means here.** Soft dedup and a fixed budget take the loss at depth from about 33 answers to none on both
models, within noise. Accuracy at 24 steps is held at the 6-step level, but no method raises it above that level.

## What this means for the paper

For these checkpoints and this training recipe, the route to an accuracy gain is closed. No arm uses the extra steps to
answer more questions on either model, so the stop rule fires and Stage 2 does not run. The write-up becomes a diagnosis
with a cheap fix, replicated on two checkpoints:

1. **The cause.** Past the trained depth, latent CoT models lose accuracy mainly because repeated latent entries dilute the
   answer's attention. The latent states deteriorating is not the main cause: on SIM-CoT, copies account for all of the
   loss. On CODI, the states' own drift contributes too.
2. **The fix.** A training-free soft dedup removes the drop on both models when a checkpoint runs past its trained depth.
   It matches the trained gate on SIM-CoT and beats it on CODI. It should not be applied to a model trained with the
   repeats, where it hurts.
3. **The limit.** One pass of answer-only fine-tuning at 12 steps, with or without a learned gate, does not get a 6-step
   GPT-2 checkpoint to use the extra steps. This is narrower than "GPT-2 cannot use more depth". PCCoT trains GPT-2
   continuous CoT from scratch with 12 latents and reports it "much better than using 6" under its own recipe
   ([arXiv 2506.18582](https://arxiv.org/abs/2506.18582), Appendix C.2).
4. **The learned gate is not a contribution.** The mechanism is prior art (TRIM-KV and others), it adds nothing over soft
   dedup on SIM-CoT, and its CODI result is confounded by a training bug.

Point 3 is what the literature predicted: every published method that stabilizes a loop past its trained depth preserved
peak accuracy without exceeding it. On novelty, the literature review proposed soft dedup and found no source that tests
it. Its nearest relatives are key-similarity eviction rules for text caches, such as R-KV and KeyDiff.

## Caveats

- **The hard-gate tie bug.** Both `gate12_hard` runs trained with it.
  - **The bug.** During training, the cut keeps every entry whose score ties the 6th-best. The gate scores are computed at
    bf16, so near-equal scores tie and more than 6 entries stay visible. Inference computes scores at full precision and
    keeps exactly 6.
  - **Why it would bite harder on CODI.** CODI's same-phase entries past step 6 are near-copies, so their scores are
    near-equal and ties should be far more common. Training would then have been closer to soft than intended, which
    matches the large cost of hard eviction on CODI. This is unconfirmed: checking it needs the trained gate's weights.
  - **The fix.** A stable top-6 cut with the gate computed at full precision. Codex's review drafted the cut; it has not been
    applied.
- **One seed.**
  - Arms differ by about 10 answers from noise alone: the SIM-CoT 6-step model trained on twice the data scored 206, not 215.
  - The SIM-CoT 6-step model's +11 over the base model is not significant either (p = 0.10).
- **Validation only.** `gsm8k_valid` has been used for discovery throughout. The held-out sets are not scored yet.
- **Exploratory comparisons.** Everything here except the planned decision rules is exploratory. Claims that accuracy holds
  at depth still need paired confidence intervals rather than non-significant p-values.
- **A weak rival.** The `logscale` rival sharpens attention by only about 3% at 24 steps, because its key count includes the
  question tokens.
