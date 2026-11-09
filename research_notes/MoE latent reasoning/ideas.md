# MoE latent reasoning: candidate designs

Step 2 of the idea round (literature, then design, then review), written 2026-11-08. Input: `literature_brief.md` in this folder, plus the project's code (`dsa/bench/models/codi.py`, `scripts/inner_loop/{train,looped,eval}.py`) and results (`results/inner_loop/README.md`). Nothing here has been run. Nothing here changes an existing plan.

**Tags.** [Brief]: from the literature brief, with its own verified/unverified status. [Proj]: this project's code or measurements. [Calc]: computed for this note (the script is a few lines of arithmetic on the brief's parameter table). [Inf]: my inference. [Search]: checked by a web search this session, at abstract or snippet level only.

---

## 0. Shared ground for every design

### 0.1 Starting point, data and recipe

- **Checkpoint:** SIM-CoT GPT-2 (`internlm/SIM_COT-GPT2-CODI@8685b42`). It scores 204/500 on validation [Proj].
- **Data:** the same 98,304-record GSM8K-Aug sample and order that `train.py` draws for seeds 7 and 8. No new data and no augmentation.
- **Recipe (from `train.py`, unchanged):**
  - loss: answer CE plus SIM-CoT's step loss, with the step decoder frozen;
  - batch 128, lr 4e-5 (2e-5·√(128/32)), 3% warmup, linear decay, gradient clip 2;
  - 768 updates, 6 latent steps.
- **Dense control:** this recipe on the dense model is Stage 1b's `ft6_step`. It scored **209 at seed 7 and 200 at seed 8** [Proj].
  - The pilot reuses those two numbers.
  - The full test reruns ft6 only if a 50-update check shows the dense code path has changed. The check compares the training losses with Stage 1b's `metrics.jsonl` at the same updates.
- **Evaluation:** validation only (500 questions; the test set is spent). Greedy decoding, scored with `scripts/inner_loop/eval.py`'s `plain` arm and the sparse hook described below.
  - Paired sign tests use `pool.py`'s convention.
  - The 173 questions with 4 or more rationale steps get a secondary readout, which no rule uses.

### 0.2 The conversion ("convert-f"), shared by Designs 1–4 and the pruned baseline

1. **Calibration.** Run the dense model in latent mode on 2,048 training records (never validation), with the answer teacher-forced.
   - Per layer, record the FFN input x (768-d) and the post-GELU activations a (3,072-d) at every position.
   - Tag each position by phase: Q (question tokens and `<bot>`), L (latent positions 1–6), A (`<eot>` and the answer tokens).
   - Subsample Q positions to about a quarter so that L and A are well represented.
2. **Importance.** Use MoEBERT's first-order Taylor score, |a_i · ∂L/∂a_i|, summed over 16 calibration batches. L is CE plus the step loss.
3. **Shared expert.** The top-s neurons by importance are always active: s = 384 at 25% FFN, s = 768 at 50%.
4. **Routed experts.** Split the remaining neurons into equal experts of 192 neurons each, by balanced k-means on co-activation features.
   - Each neuron's feature is its standardized activation over 8,192 sampled positions. This is MoEfication's co-activation idea; PADD clusters W1 rows instead.
   - At 25%: 2,688 neurons give 14 experts, top-2 active. At 50%: 2,304 neurons give 12 experts, top-4 active.
5. **Router.** Each layer gets a linear router (768 → E), fit offline.
   - It regresses each expert's contribution norm ‖W_out,eᵀ a_e(x)‖ (D2DMoE's target), using 300 Adam steps per layer, with L positions weighted ×4.
   - Router logits are computed in fp32 and ties go to the lower index. This is the lesson of the hard-gate bf16 tie bug [Proj].
   - **The router stays frozen during fine-tuning in the default arm.** That gives fewer knobs, no balancing loss and no router collapse. A trained router is a stage-2 variant (Section 1, variant 1b).
6. **GELU fix** (G-MoEfication's representative values). An expert that is not selected still contributes a constant c_e = W_out,eᵀ μ_e, where μ_e is the mean activation of its neurons over the calibration positions where it was not selected.
   - The constants are stored as frozen buffers. The FLOP cost is (k+1) vector adds of length 768.
7. **Prune-f (width baseline).** Keep the top f·3,072 neurons by the same Taylor importance, and physically remove the rest. The removed neurons' mean contribution is folded into c_proj's bias, so the baseline gets the same GELU fix.
   - Prune-25's kept set contains MoE-25's shared expert. MoE-25 is therefore "the top 384 neurons plus 2 of 14 routed experts", and Prune-25 is "the top 768 neurons". This is MoEBERT's "1 expert" ablation, done properly.
8. **LayerDrop-n (depth baseline).** Keep n of the 12 layers. Drop the layers with the lowest Block Influence, 1 − cos(h_in, h_out), measured on the calibration data in latent mode. Layer 0 is always kept.
   - This rule is preregistered. Alternate layers, as in DistilGPT2, is a sensitivity check only.
9. **Cost:** about 2–3 G4-minutes [Inf], covering forward and backward passes over 16 batches, 12 clusterings and 12 router fits. The converted objects are saved to Drive, so later runs only load them.

**Implementation.** The MoE runs as a masked-dense simulation. A hook replaces each block's `mlp.forward` with: a = GELU(c_fc(x)); mask from the shared set plus the router's top-k; out = c_proj(a ⊙ mask) + Σ_unselected c_e.

- The phase is set per model call. The question pass, each latent step and the answer pass are already separate calls in `train_step` and in `eval.py`'s `chain`/`readout`.
- Training and evaluation therefore cost dense FLOPs plus about 10% [Inf]. **All claims use analytic active parameters and FLOPs.**
- Three unit tests are needed:
  1. With all experts on, the hook equals the dense model.
  2. A gather-based sparse FFN equals the masked one to 1e-5 on 20 training questions.
  3. The FLOP counter matches the table below.
- There is no capacity factor, so no tokens are dropped.

### 0.3 Accounting [Calc]

Exact counts from the brief's parameter table (d = 768, 12 layers). FLOPs per token = 2 × active non-embedding params (attention-score FLOPs, about 2.5M at the mean context of 68, are left out). The validation averages, tokenized here, are 55.6 Q positions, 6 L positions and 6.1 A positions per problem. Per-problem FLOPs add the tied LM head on A positions (6.1 × 77.2M) and the projection MLP (6 × 2.4M); both are the same in every arm.

| Arm | Active non-emb params | % of dense | FLOPs per token | Total params | GFLOPs per validation problem |
|---|---|---|---|---|---|
| Dense SIM-CoT (and ft6) | 85.06M | 100% | 170.1M | 124.4M | 12.00 |
| MoE-50: shared 768 + top-4 of 12×192 | 56.84M | 66.8% | 113.7M | 124.6M | 8.18 (68%) |
| Prune-50: 1,536 neurons | 56.73M | 66.7% | 113.5M | 96.1M | 8.17 |
| LayerDrop-8 | 56.70M | 66.7% | 113.4M | 96.1M | 8.17 |
| **MoE-25: shared 384 + top-2 of 14×192** | **42.69M** | **50.2%** | **85.4M** | 124.6M | **6.27 (52%)** |
| Prune-25: 768 neurons | 42.56M | 50.0% | 85.1M | 81.9M | 6.25 |
| LayerDrop-6 (DistilGPT2's shape) | 42.53M | 50.0% | 85.1M | 81.9M | 6.24 |
| DistilGPT2, for reference [Brief] | 42.53M | 50.0% | 85.1M | 81.9M | — |
| D2 MoE-QL: Q at shared + top-1 (576), L dense, A at MoE-25 | Q 39.15M / L 85.06M / A 42.69M | per phase | 78.3 / 170.1 / 85.4M | 124.6M | 6.38 (53%) |

- The routers add 0.13M params (MoE-25) or 0.11M (MoE-50), and they are counted.
- **The MoE saves neither total params nor memory.** At these sizes no design claims wall-clock gains (MoEfication gets 1.2× on a GPU at 25%; MoEUT trains 1.7× slower [Brief]).
- LayerDrop really does save time and KV memory. That counts in the baseline's favor, and we should say so.
- **Phase shares** [Calc]. On validation, Q is 82.1% of positions, L 8.9% and A 9.0% (on training data: 78 / 11 / 11).
  - **Almost all FLOPs are spent reading the question, not thinking.**
  - Running the six latents through the full FFN costs about 0.5 GFLOPs per problem (4% of dense).
  - Taking one routed expert away from each question token pays for it, to within 2%.
  - This arithmetic drives Design 2.

### 0.4 Statistics and the bar

- Seed noise is about 9/500. Our bar is **+10/500 per seed, pooled over seeds 7 and 8, with p < 0.05 on the pooled paired sign test** (1,000 question-runs).
- "Equal accuracy" is a non-inferiority claim. With w wins and l losses pooled, the per-seed lower bound is (w − l − 1.645·√(w + l))/2. **MATCH** needs that bound above −12/500 per seed and a point estimate of at least −5 per seed.
  - At the discordance we expect between a converted model and its dense parent (about 50–70 per seed, so about 120 pooled [Inf]), this needs a point estimate of about −3 per seed or better.
  - If the point estimate falls in [−5, 0] but the bound misses, the result is INCONCLUSIVE, and seed 9 is the only extension.
- Single-seed pilots kill only on large gaps and send anything in between to a second seed.

### 0.5 The baselines a reviewer will demand, and DistilGPT2

All baselines start from the same checkpoint, use the same budget and recipe, and are matched on active params within 0.4%.

- **Width:** Prune-f.
- **Depth:** LayerDrop-n.
- **Ceiling:** the unpruned base (204), and ft6 at the same budget (209 / 200).
- **Training-free conversions:** MoE-f and Prune-f with no fine-tune. Each costs one evaluation.
- **Same treatment:** any loss a design adds (Design 3's teacher) is given to the Prune and LayerDrop arms too, before any comparison is claimed.

**DistilGPT2.** Training it with the full CODI recipe is out of budget. I would handle it in three ways:

1. **LayerDrop-6 is the depth baseline.** It is DistilGPT2's construction applied to our latent-trained checkpoint: 6 of 12 teacher layers, then a distillation-style fine-tune. Its active and total params match DistilGPT2's exactly.
   - At our budget it is a stronger start than DistilGPT2, because it inherits the latent training. The comparison is therefore conservative for the MoE.
2. **Optional floor arm, run only if a reviewer asks:**
   - real DistilGPT2 weights;
   - SIM-CoT's projection and three special-token rows copied in;
   - the same 768 updates;
   - Design 3's teacher loss, mapping student layer j to teacher layer 2j.
   - It costs about 4 minutes. Label it "budget-matched DistilGPT2", never "DistilGPT2 with the CODI recipe".
   - Expect it to land near the floor: our own CODI-recipe run from GPT-2 reached 7–9% after 256 updates [Proj].
3. **State the limitation in plain words:** no DistilGPT2 trained with the full latent recipe exists, and we did not train one.

### 0.6 Cost basis

- **Training:** 768 updates of `ft6_step` took 187.5 s on the G4 [Proj]. Assume about 3.5 minutes for a masked MoE arm and about 3.2 minutes for Prune or LayerDrop. A frozen dense teacher adds about 35% [Inf].
- **Evaluation:** about 1 minute per arm of 500 questions, inferred from Stage 1b's 30-minute total and not measured.
- **Setup:** each Colab session adds about 4 minutes, as in earlier notebooks.

---

## Design 1: MoEfied SIM-CoT with a short recovery fine-tune

**Mechanism.**
- Apply convert-25 to the SIM-CoT checkpoint: every layer's FFN becomes a shared expert of 384 neurons plus top-2 of 14 routed experts of 192.
- The router is fit offline and frozen, and the GELU constants are in place.
- Fine-tune all weights except the router and constants with the plain `ft6_step` recipe, with no teacher.
- Arms: MoE-f, Prune-f, LayerDrop-n and ft6.
- Stage-2 variants, run only after a GO:
  - **1b:** the router is trained with a straight-through gate (expert output × g_e/g_e.detach(), so the forward value is unchanged at initialization), aux-loss-free bias balancing, and a router lr 10× the base.
  - **1c (fallback, if the conversion loss is too large):** sparsify then convert. The first 384 of the 768 updates add a group-Hoyer penalty on per-expert activation norms (D2DMoE's recipe, scaled down); the conversion happens at update 384.

**Hypothesis.** A latent reasoner keeps its accuracy at half the active non-embedding params if it keeps all its FFN neurons and routes among them. A width-pruned or depth-pruned model of the same active size cannot.

Why it could hold:
- MoEBERT matched BERT-base at 25% FFN, with its "1 expert" (pruned) ablation clearly worse.
- MoEfication keeps more than 95% at 10–30% FFN.
- ICoT finds that implicit reasoning needs depth, which predicts LayerDrop will hurt.

Why it could fail:
- GSM8K is a hard task, and small models need more neurons (MoEfication: T5-Small needs 30–40%).
- Math accuracy tracks active params (Mixture of Parrots; Optimal Sparsity).
- GELU is not sparse, and a non-sparsified GELU model collapses under conversion (D2DMoE).
- Our budget is about 1/60 of D2DMoE's sparsification fine-tune.

**Novelty delta, honestly.** The mechanism is MoEBERT plus MoEfication plus G-MoEfication, with nothing new in it. What is new is the object:
- no trained continuous-CoT model has been put on an MoE or distilled into one [Brief];
- we would be the first to report whether six self-generated continuous thoughts survive sparsification and how they route;
- we would give the first GSM8K number for a GPT-2-scale MoE [Brief].

On its own, Design 1 is an application paper. Its value as a finding comes from the baselines and from Designs 2 and 4.

**Active params and FLOPs.** See the table in Section 0.3.
- MoE-25: 42.69M active (50.2%), 85.4M FLOPs per token, 6.27 GFLOPs per validation problem (52%).
- MoE-50: 56.84M, 113.7M, 8.18 GFLOPs (68%).

**Cheapest first test (the pilot; about 15 G4-minutes plus setup; validation only; seed 7).**
- Load the checkpoint and run convert-25: about 3 minutes.
- Train MoE-25 (about 3.5 minutes) and Prune-25 (about 3.2 minutes) with the `ft6_step` recipe.
- Evaluate five arms, about 5 minutes:
  - MoE-25 trained;
  - Prune-25 trained;
  - MoE-25 with no fine-tune;
  - Prune-25 with no fine-tune;
  - MoE-25 trained, with all 14 experts at the latent positions only (k_L = 14). This is the free lead for Design 2.
- Optional, one more minute: MoE-25 trained with k_L = 0, meaning the latents get only the shared expert.
- References: `ft6_step` at seed 7 = 209, base = 204.

The pilot uses 25%, not 50%, for two reasons. At 50%, MoE and Prune might both recover to the ceiling, and the pair would then say nothing about routing. And 25% is the DistilGPT2-sized point that the main claim is about.

**Kill rule (pilot, preregistered).** M = MoE-25 trained, P = Prune-25 trained, both at seed 7.
1. **RERUN-50.** If M < 150, both collapsed, and the 25% pair is uninformative. Run the same pair at 50% (about 8 minutes) and apply rules 2–5 to it. This rule takes precedence over the others.
2. **KILL.** If M − P < +5, routing adds nothing visible over importance pruning, even where it has the most room. The MoE line stops, with a one-paragraph note: "a pruned latent reasoner is as good as a routed one at 25% FFN".
3. **GO-25.** If M − P ≥ +10 and M ≥ 194 (within 15 of 209), run the full test at 25%.
4. **GO-50.** If M − P ≥ +10 and 150 ≤ M < 194, routing has value but 25% is too lossy at this budget. Run the full test at 50%.
5. **AMBIGUOUS.** If +5 ≤ M − P < +10, run the same pair at seed 8 and apply full-test rule 1 below to the pooled pair. If it fails, KILL.

**Full test (about 35 G4-minutes).** At the fraction the pilot chose, seeds 7 and 8:
- MoE, Prune and LayerDrop arms: 6 trainings of about 3.3 minutes each, plus 6 evaluations, plus loading the conversion and setup;
- ft6 reused (209 / 200).

The rules, pooled over both seeds:
1. **Routing value:** MoE − Prune ≥ +10 per seed, p < 0.05.
2. **Depth baseline:** MoE − LayerDrop ≥ +10 per seed, p < 0.05.
3. **Equal accuracy:** MATCH against ft6 under Section 0.4.

Verdicts:
- **MATCH** (the headline) if 1–3 all hold.
- **BEST-COMPRESSED** if 1 and 2 hold but 3 fails.
- **NULL** if 1 fails. This is written up as a negative, and the line stops.

**What an interesting positive looks like.** MoE-25 at DistilGPT2's active size (42.7M, 52% of dense per-problem FLOPs) passes MATCH against ft6. Prune-25 and LayerDrop-6 each trail it by 10 or more per seed.
- That would be "a continuous-thought reasoner at half its active params with no accuracy loss, where pruning and layer-dropping both lose".
- It would be the first such number, and the ICoT depth result would explain why LayerDrop loses.
- The training-free arms would show how much of the recovery is the fine-tune's doing.

**Main risks.**
- The GELU conversion loss may be too large to recover in 15M tokens (fallback 1c).
- The ceiling effect at 50% (this is why the pilot uses 25%).
- The frozen router goes stale while the weights move. The drift is small at lr 4e-5 over 768 updates [Inf]; variant 1b covers it.
- ST-MoE says sparse models want smaller batches and a higher lr than dense ones. Our batch 128 follows the user's preference, and the frozen router sidesteps the routing half of the problem.
- A clean positive is still "MoEBERT works on a latent reasoner". That is useful but expected, so the finding has to come from Designs 2 and 4.
- Effects below 10/500 cannot be resolved with two seeds.

---

## Design 2: Phase-budgeted experts ("think dense, read sparse")

**Mechanism.**
- Same convert-25 model, but the routed top-k depends on the phase of the model call:
  - k_Q for question tokens and `<bot>`;
  - k_L for the six latent positions;
  - k_A for `<eot>` and the answer tokens.
- The shared expert is always on.
- Arms, all fine-tuned with the D1 recipe:
  - **MoE-QL:** k_Q = 1, k_L = 14 (the full FFN), k_A = 2.
  - **Prune-QL:** the same per-phase widths (576 / 3,072 / 768 neurons) with fixed importance-pruned masks and no router. This control separates the phase budget from the routing.
  - **MoE-LQ (optional, scientific only):** k_Q = 2, k_L = 0 (the latents get only the shared 384 neurons), k_A = 2. It asks directly whether continuous thoughts need fewer active params than reading.
- The uniform comparisons are D1's MoE-25 and Prune-25 at the same seed.
- Stage 2: a dynamic-k router (D2DMoE's τ·max threshold per token). Report the mean k it chooses per phase, which answers the same question from inside the model.

**Hypothesis.** At a fixed per-problem FLOP budget, moving active params from question tokens to the six latent positions raises accuracy. Equivalently, the question prefill tolerates sparsity better than the thoughts do.

Why it could hold:
- In CODI and SIM-CoT the reasoning lives at the latent positions: CODI distills to the answer position, and SIM-CoT supervises each latent as a step.
- Most of the 55 question positions are ordinary words.
- Reasoning accuracy tracks active compute (Optimal Sparsity, [Search]).
- Harder inputs use more experts when allowed to (Harder Tasks Need More Experts, 2403.07652 [Search]: more experts on BBH than HellaSwag).
- **The trade is nearly free in FLOPs** (Section 0.3): MoE-QL costs 6.38 against 6.27 GFLOPs per problem. The +1.8% favors D2 slightly, so no bar is lowered.

Why it could fail:
- The question tokens carry the numbers, and their keys and values are read at every latent step.
- **Idea A's null is a strong prior against "the latents are starved of compute"** [Proj]: looping layers 4–7 inside each latent step found no use (the gate stayed at about 0.01).
  - That was in the dense model, though. Under 25% FFN, which phase is starved is an open question.

**Novelty delta, honestly.** The mechanism, a different compute budget for different token types, is known in several forms:
- per-token adaptive k (Harder Tasks…, 2403.07652; AdaMoE, 2406.13233 [Search]);
- a heavy branch only for routed tokens (CoLT5, 2303.09752, from memory);
- modality-specific parameters with the same compute per token (Mixture-of-Transformers [Brief]);
- iteration-specific LoRA and extra passes only on hard tokens in latent iterations (Think-at-Hard [Brief]);
- per-timestep channel subsets (Denoising Task Routing [Brief]).
- SoftCoT (2502.12134 [Search]) has a *small* assistant model generate soft thoughts for a large frozen reader. That is the MoE-LQ direction, done with two models instead of one routed model.

What I could not find, and the brief did not find either: a FLOP-matched comparison of thinking compute against reading compute inside one continuous-CoT model, together with a routing-free control. The mechanism is trivial; the measurement answers the user's question and the brief's open questions 1 and 3.

**Active params and FLOPs.**
- MoE-QL: Q positions 39.15M active (78.3M FLOPs), L positions 85.06M (170.1M), A positions 42.69M (85.4M). That is 6.38 GFLOPs per validation problem, 53% of dense.
- Prune-QL: the same, minus the router, so 6.36.
- MoE-LQ: 6.18 GFLOPs, 51.5%.
- Report active params per phase and the position-weighted mean (43.5M for MoE-QL [Calc]). Do not report a single "active params" number.

**Cheapest first test.**
- *Free lead:* D1's pilot evaluates MoE-25 trained with k_L = 14, and optionally k_L = 0, at test time.
- *Trained test (about 10 G4-minutes; seed 7):* load the convert-25 objects from Drive (1 minute), train MoE-QL and Prune-QL (about 3.5 minutes each), and evaluate both (2 minutes).
  - If the free lead shows k_L = 0 within 2 of trained MoE-25, add MoE-LQ (4.5 minutes). That decision is fixed now.

**Kill rule (preregistered).** Compared at seed 7 with D1's MoE-25 and Prune-25:
- **KILL** if MoE-QL − MoE-25 < +5 and Prune-QL − Prune-25 < +5. Moving compute into the thoughts buys nothing.
- **GO** if either difference is at least +10. Run seed 8 and require the pooled difference to be at least +10 per seed, with p < 0.05.
- **Attribution:** if Prune-QL ≥ MoE-QL − 4, the gain belongs to the phase budget, not to routing. Report it as "phase-budgeted pruning". It is a positive about latent reasoning, but no longer an MoE claim.
- For MoE-LQ, if run: "latents need fewer active params" is claimed only if, pooled over two seeds, MoE-LQ − MoE-25 ≥ −5 per seed and the MATCH-style lower bound against MoE-25 is above −12.

**What an interesting positive looks like.** Either answer is interesting.
- (a) **MoE-QL** (53% of dense FLOPs) passes MATCH against ft6 where uniform MoE-25 (52%) failed. The statement would be: "a latent reasoner can read the question with a fifth of its FFN as long as it thinks with all of it."
- (b) **MoE-LQ** holds MoE-25's score while its thoughts run on the 384-neuron shared expert (12.5% FFN). The statement would be: "continuous thoughts are cheap and the reading is what costs". This is the one-model analogue of SoftCoT's direction.
- In both cases, the dynamic-k stage reporting the k it chooses per phase would make a clean figure.

**Main risks.**
- Phase switches at `<bot>` and `<eot>` change the input distribution; 768 updates may be too few to re-balance work across phases.
- The FLOP match is approximate and depends on question length: longer questions save more.
- Any effect has to pass through only 6 positions. That gives high leverage but also high fragility.
- Idea A's null.
- If D1 kills, D2 survives only as Prune-QL against Prune-25, which is no longer about MoE.

---

## Design 3: Cross-architecture trajectory distillation (CODI's teacher-to-student, dense to MoE)

**Mechanism.**
- The teacher is the frozen dense SIM-CoT base. It runs in latent mode on the same batch, generating its own latents. The student is D1's MoE-25.
- Both see the same question and answer tokens, so positions line up exactly. This differs from CODI's explicit-versus-latent alignment.
- Loss: L = CE + L_step + λ·L_traj.
  - L_traj = the mean over the 12 layers and the positions P of smoothL1(h_s, h_t)/std(h_t), with the teacher detached.
  - This is CODI's normalization, as in `dsa/training/objectives.py::answer_distillation`. Here it is applied at a set of positions rather than one.
  - λ = 1, as in this project's CODI recipe.
  - Check λ against CE at update 0, where L_traj equals the conversion error and is not zero.
- Arms:
  - **T0:** no teacher (D1's MoE-25).
  - **T1:** P = {the token before the answer number}, i.e. CODI's single site.
  - **T2:** P = {`<bot>`, latents 1–6} ∪ T1's site, i.e. the latent trajectory.
  - Stage 2 only:
    - **T2-TF:** scheduled latent teacher forcing. With probability p, annealed from 0.5 to 0 over the first half of training, the student's input at latent step k is the teacher's latent; at test the student runs free.
    - **T2+KL:** add a KL term on the answer logits.
- If T1 or T2 wins, the full test gives the same teacher to Prune-25 and LayerDrop-6 before claiming anything. For LayerDrop, each student layer is mapped to the teacher layer it came from.

**Hypothesis.** The dense model's latent trajectory is a dense, per-step target for what each thought should contain. CE and the step-decoder loss constrain only what the latents decode to, and SIM-CoT's decoder reads gold steps only at slots 0–2 [Proj]. So a hidden-state target at the latent positions transfers more reasoning to the sparse student than CODI's single answer-position target does.

Support:
- MoEBERT: layer-wise hidden-state distillation was needed (last-layer-only was not enough), and it was worth about 1 point.
- At 25% the gap between student and teacher is largest, so the teacher has the most to give.

**Novelty delta, honestly.**
- Dense-to-MoE hidden-state distillation from the same weights is MoEBERT.
- Dense-to-MoE reasoning distillation exists for explicit CoT with logit KD (PADD).
- Single-position hidden-state distillation between two modes of one model is CODI.

What is new:
- The distillation sites here are the student's **own self-generated recurrent inputs**, so the loss shapes a trajectory rather than a map from fixed inputs.
- The teacher-forcing knob exists only when the teacher has latents of its own, i.e. across architectures.
- T1 against T2 tests whether CODI's single site is enough across architectures (the brief's open question 4). The brief found no cross-architecture CODI.

**Active params and FLOPs.** Inference is identical to MoE-25 (42.69M; 6.27 GFLOPs per problem). Training costs about 35% more per update for the teacher's forward pass [Inf].

**Cheapest first test (about 6 G4-minutes for T2 alone, about 12 with T1; seed 7).** Load convert-25. Train T2 (about 4.7 minutes) and evaluate it (1 minute). T0 is D1's pilot arm, so it costs nothing extra.

**Kill rule (preregistered).**
- **KILL** if T2 − T0 < +5, and also T1 − T0 < +5 if T1 is run.
- **GO** if either difference is at least +10. Run seed 8, with the pooled bar at +10 per seed and p < 0.05.
- **Mechanism check, required for a GO to count as "transfer":** on validation, the student-to-teacher hidden L1 at the latent positions must be lower for T2 than for T0. If accuracy rises without that, report the gain as regularization.

**What an interesting positive looks like.** T2 > T1 ≥ T0, each step at +10 or more. In words: the latent trajectory, not CODI's answer site, carries latent reasoning across architectures. Best case, the teacher closes the gap, so that MoE-25 + T2 passes MATCH against ft6 where MoE-25 alone did not. That would serve both of the user's aims at once.

**Main risks.**
- The expected effect is about 1 point (MoEBERT), which is below our floor of 10/500. This is the main reason it ranks third.
- The student starts as a copy of the teacher, so L_traj may only say "stay near the dense function", which CE already pushes toward.
- DeepSpeed-MoS found that distilling for the whole run hurt [Brief, unverified]. Stage 2 can stop the teacher at mid-run.
- **The secondary aim cannot be fully tested.** Because the student is initialized from the teacher, this is *recovery* distillation, not distillation of latent reasoning into an MoE from scratch. Training from scratch is out of budget, and the write-up has to say that.

---

## Design 4: Step-indexed latent experts, and what latents' routing looks like

**Mechanism.** In MoE-25, latent positions bypass the learned router.
- Latent step t uses a fixed routed set S_t of 2 of the 14 experts: the two with the largest mean contribution norm at latent step t in the calibration data. This is a deterministic hash on step index, in the spirit of Hash Layers and Denoising Task Routing.
- Q and A positions keep the router.
- Stage 2 variant: per-step routers, each fit offline on step-t data (LOOM-like).

**The first part is free analysis** from the routing logs of D1's pilot evaluation, per layer:
1. the Jaccard overlap of the selected experts between consecutive latent steps, against consecutive question tokens (LOOM's "expert selection collapse");
2. the entropy of expert use at latent positions against token positions (ST-MoE's sentinel-token experts);
3. whether the routing at latent step t predicts the operation of gold step t (+, −, ×, ÷) above the majority-class rate, with a linear probe on the router's top-k indicators and 5-fold cross-validation on training-set calibration data.

**Hypothesis.**
- Latents are off-manifold inputs (projected hidden states, not token embeddings), so a router fit mostly on token positions may route them noisily.
- SIM-CoT trains latent t to decode as step t, so step identity may be a better routing key than the latent's content.
- Fixed sets also remove routing jitter inside the recurrence.

Against:
- SIM-CoT's latents past step 2 do not decode as gold steps, and they settle (cos about 0.95 from step 6) [Proj]. Step identity may carry little.
- Latents are only 9% of positions.

**Novelty delta, honestly.**
- The mechanism is an application:
  - hash routing (Hash Layers; MoEBERT's default);
  - per-step channel subsets (DTR);
  - per-loop routers against collapse (LOOM, though there the recurrence runs through depth);
  - iteration-specific LoRA (Think-at-Hard).
- The routing analysis is new. The brief found nobody who has routed Coconut, CODI or SIM-CoT latents through an MoE, and no analysis of routing by arithmetic step or operation.

**Active params and FLOPs.** The same as MoE-25, minus the router's FLOPs at latent positions (negligible).

**Cheapest first test.**
- The analysis costs nothing: it needs only the logs from D1's evaluation and about a minute of CPU.
- The trained arm, MoE-25-step, costs about 4.5 G4-minutes at seed 7. **It runs only if the analysis shows latents routing unlike tokens:** latent-position entropy below 0.5× token entropy, or consecutive-step Jaccard above 0.8.

**Kill rule.**
- KILL the trained line if MoE-25-step − MoE-25 < +5 at seed 7.
- GO if it reaches +10, with seed 8 pooled at +10 per seed and p < 0.05.
- The analysis is reported either way, as descriptive results with no claim of a gain.

**What an interesting positive looks like.** The likely positive is descriptive rather than an accuracy gain. Two examples:
- latents collapse onto one or two dedicated experts at every step and every layer, which would make them sentinel-like;
- routing at latent steps 1–2 decodes the operation well above chance, while later steps do not, matching the step decoder's slot-by-slot fall-off.

**Main risks.**
- Low leverage: there are 6 positions, and our noise floor is 10/500.
- Sets chosen from the dense model's activations may be wrong after the fine-tune.
- The analysis is only as good as the frozen offline router.

---

## Considered and dropped

- **Sparse upcycling (copy the FFN into E experts).** It adds total params and does not cut active ones (Komatsuzaki; Drop-Upcycling), so it fails the main aim.
- **Attention experts (SwitchHead, MoA) to get past the 67% ceiling of FFN-only MoE.** That is too much new machinery for a 768-update recovery, and MoEUT trains them from scratch.
- **Mixture-of-Depths or early exit on latent steps.** System-1.5 already does depth shortcuts for latent CoT, and it is not an MoE.
- **Expert-choice routing.** It is non-causal across positions.
- **CODI from scratch on an MoE, or DistilGPT2 with the full recipe.** Unaffordable; Section 0.5 says how we handle it.
- **Sparsify-then-convert as a design of its own.** D2DMoE already did it on GPT-2, so it is kept only as D1's fallback, variant 1c.
- **An MoE in the projection MLP.** It has 1.2M params and runs at 6 positions, so the saving is negligible.
- **Anything that adds or rewrites training data.** Excluded.

---

## Ranking and recommended first test

1. **Design 1 (MoEfied SIM-CoT).** It is the direct test of the main aim, and every other design reuses its conversion and its MoE-25 and Prune-25 arms. If D1 kills, D2 survives only as a pruning result, and D3 and D4 lose their reason to run.
2. **Design 2 (phase budget).** It is the one place where the latent recurrence makes MoE structurally interesting, and it answers the user's question directly. It is cheap (about 10 minutes) and FLOP-matched almost for free, because the latents are only 9% of positions. Idea A's null is the main prior against its forward direction.
3. **Design 3 (trajectory distillation).** It addresses the secondary aim, and it is the literal "CODI teacher, MoE student". The expected effect (about 1 point, from MoEBERT) is below our resolution, and it can only test recovery, not transfer from scratch.
4. **Design 4 (step-indexed latent experts).** It is mostly a free analysis. The trained arm is gated on that analysis and has low leverage.

**Recommended first test: the Design 1 pilot at 25% FFN, seed 7.**
- MoE-25 against Prune-25, both fine-tuned with the `ft6_step` recipe from the SIM-CoT checkpoint.
- Five validation evaluations: both trained arms, both training-free conversions, and MoE-25 with dense latents at test time.
- References: `ft6_step`@seed 7 = 209 and base = 204.
- Cost: about 15 G4-minutes plus about 4 minutes of setup.
- The rules are the five in D1's kill rule. Write them into the run's README before it starts.
- The run costs about one Stage-1b arm pair. It decides whether routing is worth anything at DistilGPT2's active size before any credit goes to D2–D4. It also produces D2's free lead and all of D4's analysis.

**If the pilot passes, the follow-up budget is:**

| Step | G4-minutes |
|---|---|
| D2 trained test | about 10 |
| D3 T2 | about 6 |
| D1 full test | about 35 |
| **Total, plus one setup per session** | **about 50** |

Each next step depends on the previous verdict.

**Open choices for the reviewer:**
- expert granularity: 14×192 with a shared expert, against MoEBERT's 4×768 top-1;
- the router: frozen against trained, for the pilot;
- the MATCH margin: −12/500 per seed;
- whether ft6's Stage 1b numbers may be reused, or must be rerun in the same session.
