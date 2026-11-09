# MoE latent reasoning: review

Step 3 of the idea round (literature, then design, then review), written 2026-11-08. I read `literature_brief.md` and `ideas.md` in this folder, `dsa/bench/models/codi.py`, `scripts/inner_loop/{train,looped,eval,pool}.py` and `results/inner_loop/` (README, Stage 1b and seed-8 eval files). Nothing was run except a short CPU script over the committed eval JSONL files and a 5,000-record sample of `data/gsm_train.json`. The script is in my scratchpad and is not part of the repo.

**Tags.** [Checked]: I opened the source this session (arXiv abstract, HTML/ar5iv full text, or project file). [Calc]: recomputed here. [Proj]: project files. [Brief]: taken from the brief and not rechecked. [Inf]: my inference.

This is a new line. It changes no agreed plan, and nothing here belongs in `paper/main.tex`.

---

## Bottom line

| Design | Verdict | One-line reason |
|---|---|---|
| 1. MoEfied SIM-CoT + recovery fine-tune | **REVISE**, and do not run as written | The mechanism is fully covered. The pilot's main rule (MoE vs pruning) mostly measures total capacity, and its decision tree spends 35 more minutes on the expected outcome. |
| 2. Phase-budgeted experts | **REVISE** into an evaluation-only phase map | It asks the only question in the round that is specific to latent reasoning, and it can be asked with no training and no seed noise. |
| 3. Cross-architecture trajectory distillation | **REJECT** | The expected effect (about 1 point, from MoEBERT) is below our 10/500 resolution. It is recovery from the same weights, not distillation into an MoE. |
| 4. Step-indexed latent experts | **REJECT** the trained arm. Keep the descriptive statistics as a free by-product. | Hash routing, per-step routing and per-loop routing are all published. It touches 6 positions, against a 10/500 noise floor. |

**What I would put to the user: one evaluation-only probe, SPARSE-0 (Section 9).** It costs about 15 G4-minutes and trains nothing. On the dense SIM-CoT checkpoint, it measures how much accuracy survives per-token FFN sparsity at 50% and 25% of the FFN, under oracle selection and under a static pruned mask, and separately at the thinking and reading positions.

- Oracle selection is an upper bound on any router, so a failure closes the MoE line cheaply and cleanly.
- A pass picks the fraction for a revised, two-seed Design 1 run.
- My honest prior is low: the literature predicts the outcome, and only about 9% of the compute is latent. If credits are needed elsewhere, parking this line loses nothing time-sensitive.

**The user's secondary aim cannot be tested at this budget, by any design here.** That aim is whether latent reasoning can be distilled into an MoE. Every design starts the MoE from the latent model's own weights, so each one tests recovery, not distillation. The write-up should say so.

---

## 1. Novelty

**What I checked myself.** I opened LOOM (2610.01153), Think-at-Hard (2511.08577), MoEfication (2110.01786, ar5iv full text), MoEBERT (2204.07675, ar5iv) and D2DMoE (2310.04361, HTML). I also opened two looped-MoE papers the brief missed: Sparse Layers… (2605.09165) and Loopie (2607.16051). And I ran five searches pairing MoEfication, conversion, sparsity and pruning with continuous CoT, Coconut, CODI and latent tokens.

**Has anyone MoEfied a latent or recurrent reasoning model? Not that I can find.** Here is what exists:
- **LOOM** loops the shared blocks of an MoE LLM. It names "expert selection collapse" (the routers pick the same experts on every loop) and fixes it with per-loop routers. Sizes are 100M–1.7B. It reports perplexity and zero-shot accuracy, with no GSM8K and no latent tokens. The abstract does not say whether the models are converted or trained from scratch [Checked].
- **Sparse Layers Are Critical to Scaling Looped LMs** (2605.09165, NeurIPS 2026) [Checked abstract]:
  - Looped-MoE scales better than dense looped models.
  - The authors attribute this to "routing divergence": different experts fire on each pass.
- **Loopie** (2607.16051) is a looped MoE at 20B-A2B and 6B-A0.6B, with reasoning post-training [Checked abstract].
- **Training-Free Looped Transformers** (2605.23872): on MoE backbones, it loops each layer separately so that routing stays fixed [search snippet only].
- **Dreamer** trains depth-recurrent expert attention from scratch [Brief].
- **Think-at-Hard** adds depth-aware LoRA and a decider on latent iterations, with no MoE [Checked].
- **Pruning of reasoning models** has been studied for explicit CoT only (2608.14003, 2605.09008, and an ICLR 2026 pruning-and-faithfulness paper) [search snippets].

**None of these puts a continuous-thought model (latents fed back as input embeddings) on an MoE, or sparsifies one.** On this point the brief's bottom line stands.

**Per design:**
- **Design 1.** Nothing in the mechanism is new. MoEfication and MoEBERT invented the conversion, D2DMoE has already run it *on GPT-2*, G-MoEfication supplies the GELU fix, and LLaMA-MoE runs it at scale. Only the object is new, and that novelty is thinner than it looks. On validation, 82% of positions are question tokens and only 9% are latent [Calc, matches ideas.md §0.3], so a uniform MoE-25 result is mostly "GPT-2 reading GSM8K questions with a quarter of its FFN". As written, it is not worth a slot. The question it should be asking is answered more cheaply by SPARSE-0.
- **Design 2.** A different compute budget per token type is known (Mixture-of-Transformers, CoLT5, AdaMoE, Think-at-Hard, Denoising Task Routing). What is new is the *measurement*: within one continuous-CoT model, how sensitive the six thoughts are to losing FFN capacity, compared with the question tokens. That is worth a slot in evaluation-only form.
- **Design 3.** MoEBERT already does layer-wise hidden-state distillation into an MoE built from the same weights. Only the distillation sites are new, and the effect cannot be resolved (Section 7).
- **Design 4.** Hash routing, step-indexed routing and per-loop routing are all covered: Hash Layers, MoEBERT, DTR, LOOM, 2605.09165, and Chain-of-Experts' per-iteration routers. Only a descriptive routing analysis would be new.

---

## 2. Arithmetic (all recomputed by hand)

Inputs: d = 768, d_ff = 3,072, 12 layers. One FFN neuron costs 768 + 1 + 768 = 1,537 params. Each layer also has a 768-param c_proj bias, attention costs 2,362,368 params per layer, and LayerNorms total 38,400.

| Claim | Recomputed | Verdict |
|---|---|---|
| Dense non-embedding | 28,348,416 + 56,669,184 + 38,400 = 85,056,000 | correct |
| MoE-25: 384 shared + top-2 of 14×192 = 768 active neurons per layer | 12 × (768·1,537 + 768) = 14,174,208 FFN; plus attention and LN; plus router 12 × 768 × 14 = 129,024 → **42,690,048** | correct (42.69M, 50.2%) |
| Prune-25 / LayerDrop-6 / DistilGPT2 | 42,561,024 / 42,528,768 / 42,528,768 | correct |
| MoE-50 / Prune-50 / LayerDrop-8 | 56,836,608 / 56,726,016 / 56,704,512 | correct |
| Total params of Prune-25 and LayerDrop-6 | 124.44M − 12·2,304·1,537 = 81.94M; 124.44M − 6·7,087,872 = 81.91M | correct |
| Per-problem FLOPs, dense | 67.7 positions × 170.11M + 6.1 × 77.19M (LM head) + about 7 × 2.36M (projection) = 12.00 G | correct |
| MoE-25 | 67.7 × 85.38M + 0.471 + 0.016 = **6.27 G (52%)** | correct |
| D2 MoE-QL (Q at 576 neurons: 39.15M) | 55.6 × 78.30 + 6 × 170.11 + 6.1 × 85.38 + 0.487 = 6.38 G; position-weighted mean 43.5M | correct |
| D2 MoE-LQ | 6.27 − 6 × (85.38 − 70.96) = 6.18 G | correct |
| "Dense thoughts cost about 0.5 G, 4%" | 6 × 84.73M = 0.51 G = 4.2% | correct |
| Brief's expert-granularity table | 35.55 / 42.60 / 42.71 / 42.86 / 43.45 / 56.8M | correct |

**The "42.7M active, about DistilGPT2" claim** holds for non-embedding params only.
- Counting the token embedding (tied to the LM head, which runs on every answer token) and the position table, MoE-25 has **82.1M active against 124.4M dense, i.e. 66%, not 50%**.
- That matches DistilGPT2's 81.9M total. So "DistilGPT2-sized" is still right, but "half the active params" must say "non-embedding".
- The per-problem FLOP figure (52%) already includes the LM head and is the more honest headline.

**Accounting note for any measured number.** HF's `GPT2LMHeadModel` computes logits at every position of the question pass (`codi.py`, `train.py`). That is about 55.6 × 77.2M ≈ **4.3 GFLOPs per problem of unused LM-head work**, more than the whole latent phase. The analytic counts rightly leave it out. Any measured FLOP or wall-clock figure must slice the hidden states to the last position first, or this overhead will swamp the MoE's savings.

---

## 3. Is the test fair and informative?

### 3a. MoE against width-pruned dense as the primary comparison: no

- **It confounds routing with total capacity.** MoE-25 keeps 124.6M total params and Prune-25 keeps 81.9M, so at equal active params the MoE has 3.5× as many FFN neurons to choose from and to train.
- **MoEBERT shows how far capacity alone goes** [Checked]:
  - its default router is a *fixed random hash* of the token id;
  - the paper says that across routing variants "all the methods yield similar performance";
  - it also discards about half the FFN (each expert has 512 shared and 256 own neurons, so 1,536 neurons are kept in total).
  - So its evidence that MoE beats pruning is evidence for "more total neurons, trained with distillation", not for learned routing.
- **The likely outcome is therefore already known.** MoE − Prune will come out large and positive at 25%, and that is MoEBERT's result again.
- **The user's aim is non-inferiority against the dense model.** The aim is equal accuracy with fewer active params, so the primary endpoint has to be MoE-f against ft6. MoE against Prune is the secondary question, "is an MoE needed at all?"
- **What the pilot's rules do with this.**
  - Under the designer's rules, the most likely result (both degrade, MoE less) triggers GO-50. That leads to a 35-minute full test whose most likely verdict is BEST-COMPRESSED, an expected replication.
  - The opposite case also happens: if MoE and Prune both reach the ceiling, KILL fires. Yet "pruning half the FFN costs nothing" is a positive for the user's primary aim (just not for MoE).
  - The rules need to separate these cases (Section 9 and D1-R).
- **Missing control.** A hash-routed MoE is needed, on the same experts with a fixed hash of token id at Q and A and of step index at the latents, before any gap is credited to the router.

### 3b. Are 768 updates enough to recover? Probably not at 25%. What happens if both collapse?

- **The data budget is smaller than the brief says.** The fine-tune sees 98,304 records of about 57 positions each: training questions average 180.8 characters (35.4 words, so about 44 GPT-2 tokens with `<bot>`), plus 6 latents and about 7 answer tokens [Calc from a 5,000-record sample]. That is **about 5.6M real tokens, not the brief's 15M**.
- **Compared with the literature:**
  - D2DMoE used about 1B tokens just to sparsify GPT-2 before conversion [Checked], about **170× our budget, not 60×**;
  - MoEBERT trained 4–10 epochs per GLUE task [Checked];
  - the D2DMoE paper says a non-sparsified GELU model "collapses" under conversion [Checked].
- **What I expect** [Inf]:
  - 25% FFN on a GELU GPT-2 with a linear router recovers only partly in 768 updates;
  - 50% may recover close to dense.
- **The designer's RERUN-50 rule handles a double collapse correctly**, but only after spending about 15 minutes to learn what a training-free bound could have predicted. SPARSE-0 predicts it first.

### 3c. Is a router fit offline and frozen sound? Partly.

- **Offline fitting is standard practice:**
  - MoEfication trains a 2-layer tanh MLP on about 500K sampled representations, supervised only by activations (Adam lr 1e-2, batch 512, 10 epochs) [Checked];
  - D2DMoE trains 2-layer MLP routers (hidden size 128) by MSE regression on each expert's output norm, for 2,000 steps on 8–16M tokens [Checked].
- **Two departures from that practice matter.**
  1. **The router is linear.** D2DMoE's Appendix F says that **linear routers "do not perform well with our method"** [Checked]. Use the 2-layer MLP and count its params: 12 × (768·128 + 128·14) ≈ 1.2M, about 3% of the active params.
  2. **The router stays frozen while every weight trains.** MoEfication's only post-conversion fine-tune ("parameter calibration") updated **only W2 and b2**, precisely so that the routers, which read the W1 side, stay valid [Checked]. The designer trains c_fc and every upstream layer, so the neurons each expert holds change underneath a fixed router.
- **The frozen router is still acceptable for a pilot.** It behaves like a content-based hash, and hash routing worked in MoEBERT and in Hash Layers. Two cheap fixes:
  - (a) also freeze c_fc, and fine-tune the attention, c_proj and the projection (ST-MoE found that updating only the non-MoE params works about as well as updating everything [Brief]);
  - (b) log the router's agreement with the oracle top-k before and after the fine-tune.
- **The GELU constants have the same staleness problem.** c_e = W_out,eᵀ μ_e is frozen while W_out trains. Meanwhile Prune's folded constant sits in c_proj's bias, which trains. Store μ_e as a buffer and compute W_out,eᵀ μ_e in the forward pass, so that both arms track their own weights.

### 3d. Does the router run per latent step, and will it transfer to latent positions?

- **Yes, it runs per step.** Each latent step is a separate one-position forward call (`latent_step` in `looped.py`), and the hook routes every position at every layer.
- **The designer already reduced the transfer risk.** Calibration runs in latent mode, and the latent positions are tagged and weighted ×4, so the router is fit on real latent FFN inputs, not only on tokens.
- **Two residual risks:**
  - The FFN input is `ln_2` of the residual, so its scale is normalized, but its direction is not.
  - After fine-tuning, the latents are the model's own outputs and move further than token positions do. The frozen router therefore sees the largest shift exactly where errors compound over 6 steps.
- **`<bot>` is in the wrong phase.** It is tagged as Q, but its output becomes latent 1. Tag it with the latents.

---

## 4. Statistics

**How big is the seed noise, measured?**
- Same-recipe gaps between seeds 7 and 8 are 9 (ft6_step), 1 (ft6_ce), 4 (loop k1) and 5 (loop k3): an RMS of 5.6 for a between-seed gap, so about 4 per single run [Calc, from `results/inner_loop`].
- So "about 9/500" is the worst observed gap, and rules built on it are conservative.
- Discordant pairs between closely related fine-tunes run 25–37 per seed: ft6_step@7 against @8 is +17/−8, and ft6_step against the base is +21/−16 and +16/−20.
- A converted model will be more discordant (60 or more [Inf]).

**Is the single-seed rule "M − P < +5 → KILL" meaningful?** Not in the 5–10 range.
- With about 80 discordant pairs between MoE and Prune [Inf], the question-level SD of M − P is about 9, on top of a training-noise SD of about 5.6.
- A true +10 effect is then killed roughly 20–30% of the time, and a true 0 reaches GO 5–15% of the time.
- The rule can only resolve effects of 20 or more, and the effect it would resolve is the expected capacity gap (3a).

**A sounder scheme:**
1. **Prefer comparisons within one model and with no training.** Every arm of SPARSE-0 is the same deterministic model under a different mask, so there is no seed noise, and the exact paired sign test on 500 questions is the whole story.
2. **Make the primary endpoint for any trained arm MATCH against ft6, pooled over two seeds.** Keep the designer's §0.4 rule: a per-seed lower bound above −12/500 and a point estimate of at least −5.
3. **Kill on a single seed only for large deficits.** Stop after seed 7 if MoE-f@7 ≤ 192, i.e. more than 12 below ft6's two-seed mean of 204.5.
   - Take a model whose true mean sits at the MATCH edge (about 201.5).
   - With a per-run SD of 4 to 5.6, it falls to 192 or below with probability about 1–5%. So the early kill rarely throws away a model that would pass.
4. **Require two more checks before crediting routing:** MoE − Prune ≥ +10 per seed, pooled p < 0.05, **and** MoE − hash-MoE ≥ +5 per seed pooled. Otherwise the gap is reported as capacity.

---

## 5. Our own evidence

- **Accuracy follows problem length, so the score is mostly short problems** [Calc, base eval file].
  - Correct answers by number of rationale steps: 2 steps 103/155, 3 steps 58/140, 4 steps 18/91, 5 or more 11/82.
  - So **175 of the base's 204 correct answers come from problems with 3 steps or fewer**, and the 4+-step readout holds only 29 correct answers.
  - Any loss of "deep" reasoning from sparsification is mostly invisible on this metric, and the secondary readout has no power. A MATCH would mostly certify that short problems were preserved.
- **The inner loop's null** (gate about 0.01 across three trainings; answer loss flat on matched batches) says the dense latent step is not limited by compute in a way more passes could fix.
  - That is direct evidence against D2's forward direction, "give the thoughts more", at least in the dense model.
  - It says nothing about how much capacity can be *removed* from the thoughts, which is what SPARSE-0 measures.
- **The fine-tune adds no accuracy to the dense model.** ft6 runs score 199–215, and training CE is flat at 0.15–0.18.
  - A damaged model has a real gradient to follow, so recovery is possible.
  - But the reference itself moves by about ±5 between seeds: use the two-seed ft6 mean (204.5), not 209.
- **Most errors are planning errors** (Stage 0: 188 plan, 78 arithmetic, 30 readout). An FFN cut is likely to show up first as more arithmetic slips [Inf], which would be readable in the decoded steps if anyone wanted it.
- **Design 4's argument against itself uses the wrong steps.** "Latents settle, cos ≈ 0.95" was measured past step 6 (steps 6–24, `simcot-past-depth`). It says nothing about whether steps 1–6 are alike.

---

## 6. Risks the designer missed

1. **Capacity confound** in MoE against Prune. There is no hash-routing control (3a).
2. **The decision tree spends on the expected outcome:** GO-50 → a 35-minute full test → most likely BEST-COMPRESSED.
3. **Only 9% of positions are latent.** A uniform MoE result is mostly about reading the question, so the "latent reasoner" framing overstates what is tested. The designer computed the phase shares but did not carry them back into Design 1's claim.
4. **The router is linear** (D2DMoE says linear routers underperform), and the router and constants are frozen while c_fc trains. MoEfication avoided this by tuning only W2 and b2 (3c).
5. **Asymmetric GELU fix:** frozen constants in the MoE arm against a trainable folded bias in Prune.
6. **The hook must be per instance.** The step decoder in `train.py` is also a `GPT2LMHeadModel`. Patching the `GPT2MLP` class would sparsify the frozen decoder too and silently change the step loss.
7. **The training budget is about 5.6M tokens, not 15M.** That is about 1/170 of D2DMoE's sparsification, not 1/60.
8. **Floor effect on long problems** (Section 5).
9. **The LM head is computed on every position** in the measured compute (Section 2).
10. **Top-k ties under bf16 autocast in training.** Compute the router, oracle and static scores in fp32, as the designer already does for the router, and break ties by index.
11. **The 500 validation questions have now served many decisions.** That is mild adaptivity, unavoidable because the test set is spent. Say so in any write-up.

---

## 7. Verdicts, with the exact changes

### Design 1: REVISE; run only after SPARSE-0 says GO ("D1-R")

Changes:
1. **Fraction.** SPARSE-0 picks it: 25% if the oracle at 25% scores at least 194, else 50%. The pilot does not start at 25%.
2. **Router.** A 2-layer MLP (hidden size 128) fit offline as the designer describes, kept frozen, with its params counted.
   - Option: also freeze c_fc, so the router's targets stay valid (MoEfication's calibration practice).
3. **GELU constants.** Computed in the forward pass from trainable W_out and frozen μ_e.
4. **Phases.** `<bot>` is tagged with the latents.
5. **Arms, at seeds 7 and 8:**
   - MoE-f, Prune-f, and **Hash-MoE-f**: the same shared set and experts, with the expert chosen by a fixed hash of token id at Q and A and of step index at the latents;
   - ft6_step is reused (209 / 200; the per-question files exist), after the designer's 50-update loss check;
   - LayerDrop is dropped from the first run. ICoT and our own depth results make it an easy arm to beat, and it costs about 7 minutes.
6. **Rules, pooled with `pool.py`'s convention:**
   - early kill if MoE-f@7 ≤ 192;
   - **primary:** MATCH against ft6_step (§0.4);
   - **secondary:** MoE − Prune ≥ +10 per seed with p < 0.05, and MoE − Hash ≥ +5 per seed.
7. **Verdicts:**
   - **MoE-MATCH:** primary and both secondaries hold;
   - **PRUNE-SUFFICES:** MATCH holds, but Prune also passes MATCH;
   - **CAPACITY-NOT-ROUTING:** MoE − Prune holds but MoE − Hash fails;
   - **NEGATIVE:** no MATCH, and the line closes.
8. **Cost:** 6 trainings at about 3.5 minutes, 6 evaluations at about 1.2 minutes, plus conversion and setup: **about 35 G4-minutes**, spent only on a GO.

### Design 2: REVISE into SPARSE-0's phase arms

- The trained MoE-QL arm runs only if SPARSE-0 finds that the thoughts are the sensitive phase (rule R4) **and** D1-R reaches at least CAPACITY-NOT-ROUTING.
- It would then run at two seeds, at D1-R's fraction, with the bar at +10 per seed and pooled p < 0.05: about 15 G4-minutes.
- The "free lead" in the designer's pilot (k_L = 14 at test time in a model trained at k_L = 2) is out of distribution. Drop it: SPARSE-0's arms answer the same question cleanly.

### Design 3: REJECT

- **The effect is too small to see.** MoEBERT's distillation was worth about 0.7–1.1 points, i.e. 4–6/500, which is below the bar.
- **There is little for the teacher to give.** The student starts as a copy of the teacher, so L_traj is just the conversion error, and CE already pushes the student in the same direction.
- **Fairness doubles the cost.** The same teacher would have to be given to Prune and LayerDrop.
- **It still would not test the secondary aim.** Training an MoE latent reasoner from a non-latent start is the real test, and our own CODI-recipe run from GPT-2 reached only 7–9% after 256 updates.

### Design 4: REJECT the trained arm

- Keep the descriptive statistic as a by-product of SPARSE-0's oracle masks: per-layer Jaccard overlap of the active-neuron sets between consecutive latent positions, against consecutive question tokens.
- It costs nothing and makes no claim of a gain.

---

## 8. Corrections

### To `literature_brief.md`

1. **Fine-tune size.** About 5.6M real tokens (98,304 records × about 57 positions), not "about 15M". D2DMoE's 1B tokens is about 170× ours, not 60×.
2. **Think-at-Hard, per its abstract today:**
   - 93% of tokens skip the extra iteration (not 94%);
   - +3.8–4.4 over always-iterate at equal params;
   - with under 3% more params (LoRA plus the decider): +5.3–6.2 over always-iterate and +6.1–6.8 over single-iteration Qwen3;
   - not "+8.1–11.3". The abstract does not name the base model. The tag can now be [verified].
3. **MoEBERT:**
   - each of the 4 experts is 512 shared neurons plus 256 of its own, and the least important neurons are discarded, so about 1,536 of the 3,072 neurons survive;
   - the default router is a fixed random hash, and "all the methods yield similar performance";
   - training is 4–10 epochs per task;
   - the "1 expert" ablation appears only in a figure.
   - Its MoE-beats-pruning evidence is therefore about capacity, not routing.
4. **MoEfication:**
   - the router's label is described inconsistently in the paper (a sum of positive activations in §3.3, cross-entropy to a greedy ground-truth selection in §4.1 and Appendix D);
   - the post-conversion calibration tuned only W2 and b2;
   - there are no GPT-2 or decoder-only results.
5. **D2DMoE:**
   - linear routers "do not perform well" (Appendix F);
   - routers are 2-layer MLPs (hidden 128) trained by MSE on expert output norms, for 2,000 steps on 8–16M tokens;
   - the paper gives no numbers at 50% or 25% compute for GPT-2, only a figure.
6. **"~42.6M active = DistilGPT2" is non-embedding only.** With embeddings it is 82.1M, against 124.4M dense (66%), and DistilGPT2's total is 81.9M.
7. **Add these to the looped-MoE prior work:** 2605.09165 (Looped-MoE, routing divergence across loops), 2607.16051 (Loopie) and 2605.23872 (Training-Free Looped Transformers on MoE backbones). None of them uses continuous CoT.

### To `ideas.md`

1. **§0.2 step 5:** make the router a 2-layer MLP, not linear (D2DMoE, Appendix F).
2. **§0.2 step 6:** the GELU constants must track the trainable W_out. As written, they are frozen while Prune's equivalent bias trains.
3. **§0.2 step 1:** tag `<bot>` with the latents.
4. **§0.6 and Design 1's risks:** "about 1/60 of D2DMoE's budget" is about 1/170.
5. **§0.4:** "seed noise about 9/500" is the worst same-recipe gap. The RMS gap is 5.6, so the rules built on 9 are conservative, which is fine.
6. **Design 4's "against" paragraph:** the cos ≈ 0.95 settling was measured past step 6. It is not evidence about steps 1–6.
7. **All parameter and FLOP figures check out** (Section 2). The only caveat is the non-embedding qualifier on "50% active".

---

## 9. The single first test: SPARSE-0 (evaluation only)

**Question.** How much FFN width per token does a latent reasoner need, in which phase, and does per-token selection beat a fixed pruned mask? Everything is measured on the existing SIM-CoT GPT-2 checkpoint, with nothing trained.

**Setup.**
- Checkpoint `internlm/SIM_COT-GPT2-CODI@8685b42`; 500 validation questions; greedy; 6 steps; fp32 (the default in `codi.py`); the `plain` chain from `eval.py`.
- A hook on each of the 12 blocks' `mlp`, per instance. It computes a = GELU(c_fc(x)) and keeps a mask of neurons; every dropped neuron i is replaced by its representative value μ_i (G-MoEfication). The output is c_proj(a ⊙ m + μ ⊙ (1 − m)).
- **Phase tags, set per model call:**
  - **Q:** question tokens;
  - **T:** `<bot>` and the 6 latent positions;
  - **A:** `<eot>` and the generated tokens.
- **Calibration**, on 2,048 training records in latent mode with the answer teacher-forced, never on validation. It computes:
  - per-neuron Taylor importance, as in ideas.md §0.2 step 2;
  - μ_i, the mean of neuron i's activation over the positions where it falls outside the oracle top-k.
  - It takes about 2 minutes.
- Scores are computed in fp32, with ties broken by index.

**Arms.** The first is a validity check; the other six are full 500-question evaluations.

| Arm | What it keeps |
|---|---|
| ID | everything (the hook with nothing dropped). It must reproduce the base, 204, answer for answer on 500/500 |
| O50 / O25 | per token and per layer, the top 1,536 / 768 neurons by \|a_i\|·‖w_out,i‖, at every position. This **oracle** is an upper bound on any router |
| S50 / S25 | a fixed set of the top 1,536 / 768 neurons by Taylor importance, at every position: a training-free Prune-f |
| O25-T | the oracle at 25% at T positions only; Q and A dense |
| O25-Q | the oracle at 25% at Q positions only; T and A dense |

- Also logged, for free: the per-layer Jaccard of the oracle-active sets between consecutive T positions, and between consecutive Q tokens (Design 4's descriptive statistic).
- **Cost:** about 7 arm-evaluations at roughly 1 minute each (the designer's estimate, inferred and not measured), plus 2 minutes of calibration and about 4 minutes of setup: **about 15 G4-minutes, with no training**. The arms can be sharded.
- **New code:** the hook, the calibration and three unit tests (ID equals the base; masked equals gathered to 1e-5 on 20 questions; the FLOP counter matches Section 2).

**Rules, preregistered.** B = 204. Every comparison is paired on the same 500 questions and uses pool.py's exact two-sided sign test. There is no seed noise, because every arm is the same deterministic model.

- **R0, validity.** ID ≠ base on any question → fix the code. This gives no verdict.
- **R1, STOP: no headroom.** O50 ≤ 184: even oracle selection at half the FFN costs 20 or more. The MoE line closes.
  - Statement: "a GELU GPT-2 latent reasoner does not tolerate per-token FFN sparsity at 50%, even with oracle selection, before any router error".
- **R2, STOP: no routing value.** O50 − S50 < +10 **and** O25 − S25 < +10: per-token selection beats a fixed mask by less than 10 at both fractions, so no router can do better training-free. The MoE line closes.
  - If S50 ≥ 194, record "static pruning to 50% of the FFN costs at most 10 training-free". That is a pruning note, not an MoE result.
- **R3, GO.** Neither stop fires. Run D1-R (Section 7) at 25% if O25 ≥ 194, else at 50%. About 35 G4-minutes.
- **R4, phase**, reported in every case; it gates D2's trained arm only under GO:
  - "Thinking is the sensitive phase" if O25-T ≤ O25-Q − 10 with p < 0.05. Sparsifying the 7 thinking positions then costs at least 10 more than sparsifying the ~55 reading positions.
  - "Reading is the sensitive phase" in the mirror case.
  - Otherwise, "no phase asymmetry at 25%".

**What a positive result would show:**
- an upper bound on what per-token FFN routing can keep, training-free, in a continuous-thought model at 50% and 25% FFN;
- how much of that requires per-token selection rather than a fixed mask;
- whether the six thoughts or the question reading are more fragile to losing FFN capacity, a measurement nobody has published.
- It would also tell D1-R where to run, so the follow-up is not wasted on a collapsed pair.

**What a positive result would *not* show:**
- that a learned router reaches the oracle;
- that a fine-tuned MoE matches the dense model;
- any wall-clock or memory saving (the masked simulation computes everything; real savings need gather kernels, and at batch 1 GPT-2 latency is dominated by kernel launches);
- anything about distilling latent reasoning into an MoE (the weights are the same);
- anything beyond GPT-2 with SIM-CoT.
- The phase result is training-free, so a fine-tuned model may move capacity between phases.
- Because 175 of the 204 correct answers come from problems with 3 steps or fewer, "no loss" mostly means "short problems preserved".

**My recommendation.** Propose SPARSE-0 as the only next step for this line, framed as a cheap gate, with D1-R and D2 strictly conditional on it. Cap the line at about 15 + 35 + 15 = 65 G4-minutes in the best case.

If credits are needed for higher-priority work, park the line. The literature already predicts a lossy conversion at 25% on a GELU GPT-2, and the two outcomes the MoE line most plausibly reaches are both expected rather than new:
- "MoE beats pruning, through capacity";
- "pruning to 50% suffices".

---

## Sources

Opened this session:
- MoEfication (ar5iv full text): https://ar5iv.labs.arxiv.org/html/2110.01786
- MoEBERT (ar5iv full text): https://ar5iv.labs.arxiv.org/html/2204.07675
- D2DMoE (HTML v4): https://arxiv.org/html/2310.04361v4
- LOOM, abstract: https://arxiv.org/abs/2610.01153
- Think-at-Hard, abstract: https://arxiv.org/abs/2511.08577
- Sparse Layers are Critical to Scaling Looped Language Models, abstract: https://arxiv.org/abs/2605.09165
- Loop the Loopies! (Loopie), abstract: https://arxiv.org/abs/2607.16051

Search results only (snippet level, not opened):
- Training-Free Looped Transformers: https://arxiv.org/html/2605.23872v1
- Batch-wise Adaptive Pruning for reasoning models: https://arxiv.org/pdf/2608.14003
- Relative Kinetic Utility pruning: https://www.emergentmind.com/papers/2605.09008
- When Pruning Breaks Reasoning (ICLR 2026): https://iclr.cc/virtual/2026/10016646
- Do Latent-CoT Models Think Step-by-Step? (CODI mechanistic study, no MoE): https://arxiv.org/html/2602.00449v1
- MUX, multiplexed latent tokens (no MoE): https://neurips.cc/virtual/2026/poster/151022
- Hard-routed mixtures of reasoning LoRAs (explicit CoT): https://arxiv.org/pdf/2606.31413

Project files used:
- `results/inner_loop/README.md`
- `results/inner_loop/stage1b/evals/{base-simcot,ft6_step,ft6_ce}/gsm8k_valid.jsonl` and `results/inner_loop/stage1b-seed8/evals/{ft6_step,ft6_ce}/gsm8k_valid.jsonl` (per-length accuracy and discordance)
- `results/inner_loop/stage1b-seed8/pooled.txt`
- `results/inner_loop/stage1b/training/ft6_step/metrics.jsonl`
- `scripts/inner_loop/{train,looped,eval,pool}.py`
- `dsa/bench/models/codi.py`
- `data/gsm_train.json` (5,000-record sample, for question length)
