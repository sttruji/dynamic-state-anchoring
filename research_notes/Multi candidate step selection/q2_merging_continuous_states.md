# Q2: Merging several continuous reasoning states into one

Working notes, 2026-11-06. Web research only. Read level is marked per source:
[full text] = arXiv HTML or PDF text read for the facts quoted; [abstract only]; [unverified] = not confirmed this pass.
"Inference:" marks my own reasoning, not something a paper reports.

Setting recap: SIM-CoT GPT-2 (CODI-style), 6 latent steps, each step = one new position whose input is a
projection of the previous final hidden state, adding one KV entry per layer. Proposal: K=4 candidates per step
(cand 0 = dropout-off, others = MC-dropout reruns), a small RL selector, frozen generator. Question: if the selector
picks several, how to merge (next latent + per-layer KV) into one state.

---

## 1. Superposition views of continuous thought

### Zhu et al., "Reasoning by Superposition" — arXiv 2505.12514, NeurIPS 2025 [full text, main body]
- Theory: a 2-layer transformer with D continuous-thought steps (D = graph diameter) solves directed graph
  reachability; the best known constant-depth discrete-CoT construction needs O(n^2) steps.
- Construction (Lemma 2): thought c is `[t_c] = (1/sqrt|V_c|) * sum_{v in V_c} u_v`, a *normalized* superposition
  (unit norm, not a plain mean) of all vertices reachable within c steps. Weights uniform in theory.
- Experiments (ProsQA): 2-layer Coconut "near-perfect"; 2-layer CoT and No-CoT "about 75%" (chance 50%);
  12-layer CoT* 83%. Layer-2 attention concentrates on reachable edges (approx. 1.07/0.92/0.72/0.66 at steps 1-4 vs
  0.07-0.20 for non-reachable; table partly garbled in extraction). Trained weights are not uniform: biased to
  frontier and optimal nodes.
- Relevance: the precedent for "one vector holds several candidates" uses sqrt-normalization and near-orthogonal
  embeddings; superposition was *learned* with a Coconut curriculum, not imposed on a frozen model.

### Zhu et al., "Emergence of Superposition" — arXiv 2509.23365, ICLR 2026 (per arXiv) [abstract only]
- Training-dynamics analysis of the same 2-layer setting: the index-matching logit first grows, then stays
  bounded; the bounded logit keeps comparable weight on several plausible traces under uncertainty, so
  superposition emerges from training. No numbers taken.
- Relevance: superposition is a trained equilibrium (bounded logits). A frozen CODI/SIM-CoT model trained to emit
  one step per latent has no such pressure.

### Coconut, Hao et al. — arXiv 2412.06769 (COLM 2025, venue from memory) [full text]
- Analysis is Sec. 4 ("Continuous Space Enables Latent Tree Search"), not Sec. 5. Continuous thoughts
  "encode multiple candidate paths"; pattern "resembles a breadth-first search". Implicit value function = predicted
  probability of a node: step 1 top value 0.33 ("lempus"); step 2 top value 0.87 ("rorpus"), "not greedy search".
  Lower-height nodes (closer to leaves) get more definitive values. The top-1/2/3 value gap narrows -> widens over
  thoughts (parallel early, focused later).
- Table 1: GSM8K CoT 42.9, No-CoT 16.5, iCoT 30.0, Pause 16.4, Coconut 34.1 (8.2 tokens);
  ProsQA CoT 77.5, No-CoT 76.7, iCoT 98.2, Coconut 97.0; ProntoQA Coconut 99.8.
  w/o curriculum: GSM8K 14.4, ProsQA 76.1. Per-k ProsQA numbers only in Fig. 3 (not extracted).
  GSM8K: c=3 thoughts per stage gave "a slight performance drop accompanied by increased variance".
- Relevance: BFS-like superposition is reported for ProsQA (branching graph), not for GSM8K arithmetic.

### CoT2, Gozeten et al., "Continuous Chain of Thought Enables Parallel Exploration and Reasoning" —
### arXiv 2505.23648, ICLR 2026 (per arXiv) [full text]
- Continuous token z_t = E^T alpha_t (softmax-weighted embedding mix); deterministic.
- **Multi-token sampling (MTS)**: sample K discrete tokens i.i.d. from alpha_t, feed the *uniform average*
  z_t = (1/K) sum_r e_{i_r}. K=1 is discrete CoT. Alternative: Dirichlet sampling alpha_hat ~ Dir(gamma*alpha_t)
  (unscaled gamma "often causes training instability").
- **GRPO over continuous tokens**: ratio per continuous step = geometric mean of the K per-sample prob ratios,
  r_t = (prod_r alpha_{t,i_r} / prod_r alpha^old_{t,i_r})^(1/K), "to keep the ratio on the same scale" as discrete;
  Dirichlet variant uses the density ratio. Sparse 0/1 reward, shared advantage, mu=1.
- Numbers (Table 1, MNNS, 1-layer GPT-2, val %): d=24: SFT 39.76, GRPO K=1 49.01, K=3 52.60, K=6 49.69;
  d=32: SFT 43.50, K=1 51.61, K=3 55.66, K=6 50.38 -> moderate K best.
  Table 2 (4-layer GPT-2, d=32): ProsQA CoT2 SFT 93.37 -> 94.21 (K=12) vs discrete 68.50 -> 72.76;
  ProntoQA CoT2 75.36 -> 77.64 vs discrete 59.58 -> 74.03.
  GSM8K (Qwen3-0.6B, single run, "preliminary"): CoT2-MTS vs discrete CoT at max len 160: 41.3 vs 37.0;
  128: 36.2 vs 34.3; 96: 27.1 vs 27.2; 80: 27.0 vs 21.5. MTS K=4, G=4.
- Pitfalls reported: d=16 cannot fit budget B=16 (capacity limit, theory d = Omega(B log(v/B)));
  scaling logits by 1/K instead of averaging "leads to a distribution shift relative to the SFT-trained model";
  gains saturate in K; RL objective "saturates to near zero" at large K.
- Relevance: closest precedent for "sample K, average, train selector with RL". Their averaging is of *discrete
  vocabulary embeddings* (on-manifold points) and the model was *trained* (CSFT) to consume superpositions.

---

## 2. Probability-weighted embedding mixtures

### Soft Thinking, Zhang et al. — arXiv 2505.15778 [full text]
- Concept token = full next-token prob vector; input = sum_k p[k] e(k) after top-n filtering (n=15 QwQ, 10 R1).
  Training-free. "Cold Stop": end thinking after k consecutive low-entropy steps (H < tau).
- Math avg (MATH500, AIME24, GSM8K, GPQA-D): QwQ-32B 83.84 -> 86.32 (+2.48), length -11.6%;
  R1-Distill-Qwen-32B 81.32 -> 83.03, -22.4%; R1-Distill-Llama-70B 81.31 -> 82.42, -17.9%.
- Ablation (QwQ-32B, Table 3): **COCONUT-TF (feed raw last hidden state) 0.0 / 0.0** on AIME24 / LiveCodeBench,
  always hitting 32,768 tokens; **Average Embedding (mean of top-5 embeddings) 6.66 / 7.49**; Soft Thinking w/o
  Cold Stop 73.33 / 56.98; with Cold Stop 83.33 / 62.72.
- Pitfalls: concept tokens are OOD; without Cold Stop the model repeats to max length; Cold Stop "does not
  fundamentally resolve the OOD problem".
- Relevance: strongest single data point that *uniform* averaging of distinct candidates in a frozen model is
  catastrophic, while probability-weighted mixing that stays near the top-1 embedding is fine.

### Wu et al., "LLMs are Single-threaded Reasoners: Demystifying ... Soft Thinking" — arXiv 2508.03440,
### preprint ("working in progress") [full text]
- Probing ~10^6 steps of QwQ-32B: the next prediction from a soft input matches the top-1 token's prediction
  (JS divergence concentrated near 0) and is far from the 2nd token's; logit lens: top-1 path -> 1.0 in late
  layers. "Greedy Pitfall": soft thinking ≈ greedy decoding.
- Table 1 averages (8 benchmarks): R1-Distill-Qwen-32B greedy 71.57 / sampling 78.50 / soft 72.13;
  QwQ-32B 82.64 / 82.35 / 80.06; Skywork-OR1-32B 76.85 / 82.99 / 79.21. Vanilla soft < token sampling for all three.
- Fix = randomness: Dirichlet (alpha=4) avg 78.36 / 81.39 / 83.12; Gumbel-Softmax (tau=0.5) 79.55 / 83.04 / 83.41
  (only Gumbel beats sampling on all three).
- Relevance: a deterministic weighted merge collapses onto the dominant component. In our case the dominant
  component would be candidate 0 (dropout-off), so a score-weighted merge risks reducing to "always cand 0".

### Butt et al., "Soft Tokens, Hard Truths" — arXiv 2509.19170 (Meta), preprint [full text]
- Hard / soft (mixture at tau=0.5 + Gaussian noise on the input embedding) / fuzzy (tau=0.0001 + noise).
  Noise sigma = 0.33 x RMS token-embedding norm. RL = RLOO, G=32. Log-prob of a continuous token = Gaussian
  density of the noisy input around the noise-free mixture.
- Table 1 (greedy pass@1 / pass@32): Llama 3B GSM8K-trained: hard 75.9/94.1, fuzzy 76.7/97.4, soft 77.2/97.9.
  Llama 8B GSM8K-trained, evaluated on MATH-500: hard 20.2/45.4 vs soft 44.7/83.9 (hard training collapsed).
  Qwen 3B MATH-trained: hard 84.0/97.2, soft 82.9/97.6 on GSM8K.
- Soft *inference* does not help: soft-trained Llama 3B GSM8K 77.2 (hard decode) vs 74.5 (soft decode).
  Noise scale 3.0 -> learning collapse. Soft/fuzzy training preserves OOD NLL better than hard.
- Relevance: (a) template for an RL log-prob of a continuous perturbation (Gaussian density) — analogous to
  scoring MC-dropout candidates; (b) mixing helped as an *exploration device during training*, not at inference.

### Zhuang et al., Mixture of Inputs, "Text Generation Beyond Discrete Token Sampling" — arXiv 2505.14827,
### preprint [full text]
- Input = posterior mean of a Dirichlet prior (alpha = H(p) p) updated with the sampled token:
  w_i = (H p_i + (beta + 1 - H) y_i) / (beta + 1). High entropy -> closer to p; low entropy -> one-hot.
- Table 1 avg (AIME, CountDown4, GPQA-D, LiveCodeBench): QwQ-32B 72.86 -> 74.15; Nemotron-49B 53.09 -> 55.45;
  Gemma-3-27B 40.09 -> 41.65; DAPO-Qwen-32B 58.28 -> 60.33 (avg +1.8 over 16 pairs; DAPO AIME -0.23, p=0.8).
- **Direct Mixture (h = sum p_i e_i, no sampled token)**: QwQ 60.96, Nemotron 46.97, DAPO 47.90 (all below
  standard), Gemma 41.36 (above). Nemotron LiveCodeBench 39.92 -> 16.04.
- Pitfalls: beta is task dependent; feeding mixed hidden states "cannot make sense" without retraining.
- Relevance: anchoring the mixture on one actually-sampled candidate (entropy-gated) is what makes it safe.

### Tang et al., "Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge" — arXiv 2601.08808 (Jan 2026),
### code GMLR-Penn/Multiplex-Thinking [full text]
- Sample K tokens, average one-hots s = (1/K) sum z_j, reweight by LM-head probs over the sampled support,
  embed: c = E^T (s ⊙ w). Log-prob for RL = sum over steps and over the K samples. GRPO, 8 rollouts.
- Pass@1 (avg 64 runs), 7B: Discrete CoT / Stochastic Soft Thinking / Discrete RL / Multiplex:
  AIME24 15.7 / 20.3 / 17.2 / 20.6; AMC23 42.4 / 47.9 / 44.7 / 50.7; MATH500 71.6 / 76.5 / 74.1 / 78.0.
  1.5B gains smaller (MATH500 66.3 -> 67.5). Best in 11/12 settings.
- K ablation: AMC23 44.7% (K=1) -> 49.6% (K>=2); K=4 not tested; rest only in a figure.
- **Uniform vs probability-reweighted average: nearly identical** (7B AIME24 19.9 vs 20.6, AIME25 20.0 vs 19.7,
  MATH500 77.6 vs 78.0).
- Training-free Multiplex-I 7B: MATH500 76.4 (vs Discrete CoT 71.6).
- Pitfalls: heuristic early stopping -> "training instability and the generation of incoherent content";
  authors cite "interference between superposed reasoning paths", needing capacity (gains larger at 7B than 1.5B).
- Relevance: when the K samples are near-duplicates the merge acts like a discrete token; the weighting scheme
  matters little. The K-sample sum log-prob gives a ready RL formulation for "select a subset and merge".

### Other mixture papers found
- **SofT-GRPO** (Zheng, Gu, Liu, Teh, Lee) — arXiv 2511.06411 [abstract only]: Gumbel noise + Gumbel-Softmax keeps
  soft tokens in embedding space; reparameterized policy gradient. vs discrete GRPO (1.5B-7B): Pass@1 +0.13 avg,
  Pass@32 +2.19 avg. Search summaries say Dirichlet/Gaussian noise variants were worse or less stable [unverified].
- **HRPO**, "Hybrid Latent Reasoning via RL" (Yue et al.) — arXiv 2505.18454, NeurIPS 2025 [full text, partial]:
  learned gate e = a ⊙ e_tok + sqrt(1 - a^2) ⊙ (i ⊙ h), a = exp(-c softplus(Λ) r), c=8, initialized a≈1 (token
  dominates). Hidden state projected as W_e^T p / ||p|| (normalized by ||p|| to "preserve the scale and variance").
  Raw hidden states -> "nonsensical rollouts with zero reward"; pure interpolation tracked HRPO a few hundred
  steps then collapsed. GSM8K Qwen2.5-1.5B GRPO 0.711 vs HRPO 0.720; 3B 0.834 vs 0.845.
  Relevance: variance-preserving gate (a, sqrt(1-a^2)) and ||p||-normalization are direct fixes for norm shrinkage.
- **Superposed Decoding** (Shen et al.) — arXiv 2405.18400, NeurIPS 2024 [full text, partial]: feed
  x = sum gamma_i z_i (gamma = normalized draft probs) of k drafts' last token embeddings into one pass of a frozen
  LLM. Llama-2-7B is "sufficiently linear up to 10 timesteps" (mean cosine >= 0.6 to separate runs) but linearity is
  "imperfect"; needs n-gram filtering; per-draft perplexity rises with draft index (5.03, 7.97, 10.05; best 4.63).
  Relevance: frozen models tolerate probability-weighted input superposition only approximately, over few steps.
- **Parallel Test-Time Scaling for Latent Reasoning Models** (You et al.) — arXiv 2510.07745, ACL 2026
  [full text, partial]: MC dropout and additive Gaussian noise for sampling Coconut/CODI/CoLaR trajectories,
  plus a LatentRM (step-wise contrastive, N=8, p=0.2). Aggregation = majority vote, best-of-N, beam search;
  **no latent averaging tried**. Coverage peaks at moderate diversity; MC-dropout coverage "shows a sharp decline"
  at high diversity. COCONUT best-of-8 w/ LatentRM 35.4 vs majority vote 33.6 (GSM-Test).
  Relevance: closest prior setup to ours (MC-dropout candidates on CODI); merge is open there.
- "Concept token" papers: Soft Thinking coins the term; I did not find another concept-token *merging* paper this
  pass (Large Concept Models is sentence-level, not merging) [not searched deeply].
- LTPO, "Thinking on the Fly" — arXiv 2510.04182, ICLR 2026 poster [search summary only]: test-time policy-gradient
  refinement of latent thought vectors with a Gaussian perturbation policy and a confidence reward. Related to
  "reward-weighted merge = one ES step" below.

---

## 3. Averaging pitfalls

### Model soups, Wortsman et al. — arXiv 2203.05482, ICML 2022 [full text, pp. 3-6]
- CLIP ViT-B/32 on ImageNet (Table 3): best individual 80.38; **uniform soup 79.97 (< best single)**;
  greedy soup 81.03; learned soup 80.89; ensemble 81.19. Dist. shifts: uniform 51.45 vs best single 47.83.
- Uniform soup only beats the best model "when all individual models achieve high accuracy"; "there can be an
  error barrier between fine-tuned models", mainly with high learning rates. Greedy soup adds a model only if
  held-out accuracy does not drop.
- Fig. 3: averaging gain grows with the angle between solutions (more orthogonal fine-tunes -> larger gain).
- Relevance: averaging works inside one basin; a selection gate (greedy soup) is needed to exclude members that
  break the average. Analogue for us: merge only candidates that pass a similarity / score test.

### REPAIR, Jordan et al. — arXiv 2211.08403 (ICLR 2023, venue from memory) [abstract only]
- "Variance collapse": activations of interpolated networks shrink in variance; rescaling preactivation statistics
  reduces the barrier 60-100% (ResNet50 ImageNet 74%, ResNet18 CIFAR10 90%).
- Relevance: averaging hidden states lowers their variance/norm; renormalize after merging.

### Norm shrinkage of convex combinations (derivation, inference)
- For m vectors with equal norm r and mean pairwise cosine c: ||mean|| = r * sqrt(1/m + (1 - 1/m) c).
  m=2, c=0.95 -> 0.987 r; m=4, c=0.5 -> 0.79 r; m=4, c=0 -> 0.5 r. Near-duplicates barely shrink; distinct ones
  shrink a lot and land off the shell the model was trained on.
- slerp: White 2016, "Sampling Generative Networks" — arXiv 1609.04468 [abstract only]: slerp "prevents diverging
  from a model's prior distribution" and gives sharper samples than lerp.
- MiniCache (below) uses slerp on directions + stored norms; its appendix "Mean" (plain averaging) baseline is far
  worse at aggressive merge depth.
- Zhu et al. superposition uses 1/sqrt(|V|) (unit norm), not 1/|V|.

### Mode averaging / blur
- Mathieu, Couprie, LeCun — arXiv 1511.05440, ICLR 2016 [abstract only]: MSE gives "inherently blurry
  predictions". (Mechanism = averaging over modes; stated from general knowledge, not from the fetched page.)
- Soft Thinking Average Embedding (6.66 vs 83.33 AIME) and MoI Direct Mixture are the LLM-side evidence that
  averaging distinct discrete choices produces an input that is neither.

---

## 4. KV-cache / token merging operators

### ToMe, Bolya et al., "Token Merging: Your ViT but Faster" — arXiv 2210.09461 (ICLR 2023, venue from memory)
### [full text]
- Bipartite soft matching on cosine similarity of keys (averaged over heads); merge r pairs per layer.
- Merge operator: size-weighted average of features (sizes s add up).
- **Proportional attention**: A = softmax(QK^T/sqrt(d) + log s) — "the same operation as if you'd have s copies of
  the key". Without it, "if we merge two tokens with the same key, that key has less effect in the softmax".
- Ablations (ViT-L/16 MAE off-the-shelf, r=8, baseline 85.96): weighted avg 84.25; keep one 81.01; max pool 83.50;
  avg pool 83.57; random prune 79.22; attention prune 79.48. Similarity from keys 84.25 > Q 84.04 > V 83.80 > X 83.70.
  Proportional attention: MAE 84.25 (off) vs 83.84 (on); AugReg 82.15 (off) vs 83.51 (on): "necessary for supervised
  models (AugReg, SWAG, DeiT), but not for MAE" (MAE was pretrained with token dropping; gap disappears after training).
- Headline: off-the-shelf ~2x throughput, ViT-L/16 MAE -2.27 points; trained ViT-L -0.6 / ViT-H -0.4.

### CaM, Zhang et al., "Cache Merging for Memory-efficient LLMs Inference" — ICML 2024, PMLR 235:58840-58850
### [full text]
- Merges **values only**; the evicted key is dropped. Evicted value V_i is spread into the next m local values:
  V_k <- V_k + M * V_i / m, M ~ Bernoulli(clamp(Abar_i / avg(Abar_{j:j+m}), 0, 1)) (cumulative attention).
  Theorem: V_j <- V_j + (A_i/A_j) V_i makes the current-step output exact; a wrong ratio can be worse than eviction.
- 20% cache budget, LLaMA-7B: StreamingLLM -> +CaM: RTE 49.5 -> 54.6, COPA 64.0 -> 68.0, OpenBookQA 30.0 -> 31.8;
  H2O COPA 47.0 -> 56.0. Ablation: w/o merge mask RTE 48.1 (below baseline 49.5).
- Additive (not convex) value merge = mass-preserving by construction.

### KVMerger, Wang et al., "Model Tells You Where to Merge" — arXiv 2407.08454, preprint [full text]
- Merging sets from key cosine similarity within a sequence (keys are locally similar; values are not).
  Pivot = highest aggregated attention. Gaussian-kernel convex merge on keys and values with the same weights
  g_pi = exp(-||k_p - k_i||^2 / 2 sigma^2), normalized. **No attention-mass correction.**
- LongBench avg Llama2-7B-chat: full 35.22; H2O 50% 33.49; CaM 50% 33.36; KVMerger 50% 35.02; 35% 34.50.
  Plain average-weighted merging scored below H2O ("merging may introduce noise"); random pivot 24.20 -> 22.12.

### D2O, Wan et al. — arXiv 2406.13035, ICLR 2025 (per arXiv) [full text, partial]
- Evicted token matched to nearest retained key by cosine; merged only if similarity > EMA threshold (beta=0.7);
  convex weighted merge of k and v with softmax-of-similarity weights. **No log-count correction.**
- Llama-3-8B throughput up to 3.04x; NIAH 91.88 vs full 94.46 (H2O 72.34); GSM8K 41.24 at 3:1. Tables inconsistent.

### MiniCache, Liu et al. — arXiv 2405.14366 (NeurIPS 2024, venue from memory) [full text, partial]
- Cross-layer merge of adjacent layers' KV per token: slerp of unit directions (t=0.6), keep both norms, restore
  x_hat_l = e * ||x_l|| / ||e||. Keep tokens with large angular distance unmerged (gamma=0.05).
- Plain averaging "can cause significant information loss" (attributed to outlier channels). Appendix rows as
  extracted (semantics partly unclear): LLaMA-3-8B GSM8K MiniCache 0.397 vs Mean 0.025; LLaMA-3-70B GSM8K 0.615 vs
  0.493. Retention gamma=0 vs 0.05 on LLaMA-2-7B GSM8K: 0.108 vs 0.152. Up to 5.02x compression (with 4-bit).

### KeepKV, Tian et al. — arXiv 2504.09936, AAAI (per ojs.aaai.org search hit) [full text, partial]
- **"Attention Sag" theorem**: any convex-combination merge gives the merged entry less attention than the sum of
  the parts (A'_r < A_e + A_c), so D2O/KVMerger-style merges perturb outputs.
- **Electoral Votes**: integer vote p_i per entry, p_r = p_e + p_c on merge; attention uses p_i * exp(q k_i / sqrt d)
  — identical to adding ln p_i to the logit, i.e. ToMe proportional attention for KV caches.
- ZIP-Merging: v_r = (w_e v_e + w_c v_c)/(w_e + w_c), key rescaled so the current-step output is unchanged.
- Throughput ~2.2x full cache (Llama-2-7B, 20%); LongBench mixed (e.g., Mistral-7B Qasper full 39.74 vs 17.95).

### How these relate to KK-LSE (inference, with a short derivation)
- m identical copies of (k, v) in the cache contribute m exp(q k) v / Z — the same as one entry with logit
  q k + log m. So **ToMe/KeepKV proportional attention on a merged entry == keeping all copies uncorrected.**
- KK-LSE gives b_j = -alpha log sum_l exp(beta (cos(k_j,k_l) - 1)) ≈ -alpha log n_j for a group of n_j near-copies.
  With alpha=1, large beta: the group's total mass becomes that of one entry. So **KK-LSE(alpha=1) on kept copies ==
  merged entry with no log-count bias == dedup.**
- The two are inverse corrections. ToMe treats multiplicity as signal (a merged token stands for s patches).
  KK-LSE and dedup treat it as an artifact. For candidates at one latent step, how many the selector picked is an
  artifact, so the KK-LSE/dedup direction is the right one, and ToMe-style +log s would bring back the dilution.
- Jensen: exp(q·(w k_a + (1-w) k_b)) <= w exp(q k_a) + (1-w) exp(q k_b). Convex key merges sag below even *one*
  copy's mass, but only at second order in q·(k_a - k_b), which is negligible for near-duplicates.

---

## 5. Learned set aggregation

- **DeepSets**, Zaheer et al. — arXiv 1703.06114, NeurIPS 2017 [abstract only]: main theorem characterizes
  permutation-invariant functions (known form rho(sum_x phi(x)); form not verified on the fetched page).
- **Set Transformer**, Lee et al. — arXiv 1810.00825, ICML 2019 [full text, partial]: PMA_k(Z) = MAB(S, rFF(Z)) with
  k learnable seed vectors (k=1 for a single output). Max-regression MAE (Table 1): rFF+mean pooling 2.133,
  sum 1.902, max 0.1355, SAB+PMA 0.2085. Mean pooling cannot express "pick one"; attention pooling nearly can.
- **Graph of Thoughts**, Besta et al. — arXiv 2308.09687 (AAAI 2024, venue from memory) [full text, partial]:
  aggregation = new thought with in-edges from several thoughts (merge sorted sublists, merge documents).
  Sorting with ChatGPT-3.5: ~62% better quality than ToT and >31% lower cost; ~61% (P=64) and ~69% (P=128) lower
  median error than ToT2; at P=32 only "negligibly" better. Aggregation itself makes errors (all 10 sample merges
  had errors), so an improve/refine step plus keep-best scoring is needed.
  Relevance: GoT aggregation is in text space with a generator that can reason about merging. Our frozen
  latent generator cannot.

---

## 6. Candidate merge operators for our setup (all inference)

Facts about our candidates that matter:
(a) At step t, all K candidates share the same input e_t and the same past cache; they differ only by dropout masks
    in the step-t forward. So their per-layer KV entries at position t are likely near-duplicates (worth measuring:
    per-layer cosine between candidates). Their next latents e_{t+1}^i, built from the final hidden state, can differ
    more.
(b) Averaging many MC-dropout samples approximates the dropout-off forward (the standard weight-scaling view of
    dropout; Srivastava 2014 / Gal & Ghahramani 2016, not fetched this pass). So **a uniform merge of all candidates
    tends to recreate candidate 0**. A merge adds information only through *which* subset is selected and how it is
    weighted.
(c) Reward-weighted averaging of perturbations around candidate 0, e = e_0 + sum_i w_i (e_i - e_0), is one
    evolution-strategies / CEM step in latent space (cf. LTPO's Gaussian perturbation policy).

Operators:
(i) Score-weighted average of next latents (keep one KV set, or average KV too).
   - KV: average is near-lossless if candidates are near-duplicates (small norm shrink, second-order sag). Keeping
     only the top-scored candidate's KV is the "keep one" ToMe ablation (81.01 vs 84.25), and with near-duplicates
     the gap should be small.
   - Latent: risk of collapse to candidate 0 (Single-threaded paper), or off-manifold blur if candidates are
     distinct (Soft Thinking Average Embedding 6.66; MoI Direct Mixture). Mitigate by anchoring on one selected
     candidate (MoI-style: w = lambda * onehot + (1 - lambda) * scores), renormalizing (iv), and keeping the
     selection stochastic for RL (Gumbel per SofT-GRPO; K-sample summed log-prob per Multiplex).
(ii) Attention pooling (PMA, 1 seed, query = current state) over candidate latents with a small learned module.
   - Strictly more expressive than mean (can approximate select-one, per Set Transformer max-regression).
   - Risk: a free output projection can leave the generator's input manifold (HRPO raw hidden state -> zero reward;
     COCONUT-TF -> 0.0). Constrain the output to a convex combination of candidates (attention weights only) plus
     renorm, or use an HRPO-style gate initialized to candidate 0. It then reduces to (i) with learned, state-dependent
     weights. Sparse RL reward makes training it plus the selector harder than training the selector alone.
(iii) Keep all selected candidates' KV entries at the same position ("superposed step").
   - **Uncorrected, this recreates the measured dilution.** m near-duplicate entries get about m times one entry's
     mass, which takes attention away from question tokens and other steps. This is exactly the near-duplicate
     latent-KV situation that dedup fixed.
   - So dedup / KK-LSE matter again. KK-LSE was inert at 6 steps because there were no near-copies within 6 steps;
     superposed candidates would create them inside 6 steps. With alpha≈1 it gives the right "count once"
     correction. ToMe proportional attention (+log s) is the wrong direction here.
   - Cleaner variant: explicit fractional votes, bias_i = log w_i with sum w_i = 1 (KeepKV votes with weights from the
     selector). The step's total mass then equals one entry, and each later query re-weights candidates by q·k_i (a
     query-dependent merge). For identical keys this reduces to the weighted value average.
   - Remaining issue: the next step still needs ONE input vector, so (iii) covers KV only and must be paired with
     (i)/(iv) or top-1 for e_{t+1}. Pairing candidate j's latent with a superposed KV is a small mismatch.
     Cache length grows by up to m-1 entries per step.
(iv) Slerp / renormalized averaging.
   - Fixes norm shrinkage (MiniCache slerp + stored norms; Zhu 1/sqrt(n); HRPO /||p|| and sqrt(1-a^2) gate;
     REPAIR rescaling). It does not fix semantic blur between distinct candidates.
   - Cheap form: rescale the mean to the mean candidate norm, or merge *before* the latent projection's final
     LayerNorm (if there is one; check the SIM-CoT/CODI projection code) so LN renormalizes. With near-duplicates
     (cos ~0.95+) lerp and slerp differ by about 1%, so this matters only when distinct candidates are merged.

Diagnostics before building any merge (inference):
- Per-layer cosine among the K candidates' KV at each step, and among the next latents.
- Decode the merged latent with SIM-CoT's step decoder: does it decode to one candidate's step, a blend, or junk?
- Compare merge-all with candidate 0 to test point (b).
