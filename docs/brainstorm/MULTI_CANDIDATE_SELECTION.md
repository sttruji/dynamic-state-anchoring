# Keeping several latent candidates per step: design study

A literature-grounded design study from the dsa-brainstorm session, 6 November 2026. **This is an extension, to be tested
after the current plan in [Step-level selection](STEP_SELECTION.md); it does not change that plan's stages or decision
rules.** The extension adds a policy that may keep more than one candidate at a latent step. It covers
two ways of using the kept candidates: merging them now, or keeping their chains alive and aggregating at the answer. No
code was written and nothing was run. The working notes, with longer extracts and formulas, are in
`research_notes/Multi candidate step selection/`: `q1_…` to `q4_…` per question area, plus `design_notes.md`.

**Conventions.**
- **Reported** marks a paper's own claim or number. **Inference** marks my design reasoning. Sections 3–6 are inference
  unless a citation is attached.
- Read levels:
  - [full]: the method and results sections were read;
  - [partial]: only part of the full text, for example because the fetch truncated it, or the numbers exist only in
    figures;
  - [abstract]: the abstract only;
  - [unverified]: existence or numbers could not be confirmed.
- ≈ marks a value read off a figure.
- "Token-forward" is one single-position forward pass of a GPT-2-small. The generator and SIM-CoT's step decoder are both
  GPT-2-small, so their costs are comparable.

## 1. Summary

**Nobody has done this in latent space.**
- I found no paper that keeps several candidates per latent step, merges latent candidates, or votes over a latent tree on
  CODI or SIM-CoT. I also found no RL-trained selector over latent candidates.
- The one per-step multi-keep method in a latent model is LatentTTS's beam search, and it runs on Coconut only. It uses
  √N beams with √N dropout expansions each, ranked by a supervised latent reward model. It ties best-of-N on GSM8K and
  trails it on GSM-Hard.
- The closest RL analogue in text is MIRAGE: a Bernoulli keep-mask over prompt "perspectives" on a frozen LLM, trained
  with REINFORCE with a cost of 0.05·k/20 for keeping k of the 20. It decides once per question, not per step.

**On GPT-2 latent models, candidates contain much more than any selector has recovered.**
- LatentTTS (Coconut): greedy 34.0, coverage@4 46.7, best-of-8 35.4, majority-of-8 33.6.
- ITS-Continuous: pass@16 42.61, best reranker 33.36.
- Cui et al. (answers sampled, not latents): SIM-CoT greedy 42.23, pass@100 69.60, majority@100 43.21.
- Conversion from coverage to accuracy has been at most about 1.5 points.

**Option (a), merging now, is weakly supported for our setting.**
- All K candidates at a step share the input and the cache, and differ only in their dropout masks.
- The mean of MC-dropout reruns approximates the dropout-off pass (Srivastava et al. 2014, §7.5). So a uniform merge
  mostly gives back candidate 0.
- Frozen models handle mixed inputs badly:
  - Soft Thinking's uniform average of the top-5 embeddings scores 6.66 on AIME, against 83.33 for its weighted mixture.
  - Deterministic weighted mixtures collapse onto their top component (the "single-threaded reasoners" paper).
  - Mixture of Inputs works only when anchored on one sampled token.
- The merges that work, Multiplex Thinking and CoT2, train the generator with RL to read them.
- If the selected candidates' KV entries are kept side by side without a correction, they recreate the near-copy dilution
  this project measured. The right correction is −log of the multiplicity: the direction of KK-LSE and dedup, the
  inverse of Token Merging's proportional attention. So KK-LSE's idea matters again here. The selector knows the merge
  weights exactly, though, so a log w bias replaces KK-LSE's estimate of the multiplicity from keys.

**Option (b), keeping chains alive, is better supported.** Voting is robust where argmax selection with a weak verifier
is not.
- DIVERSE on GSM8K: majority 80.0, verifier best-of-N 65.9, verifier-weighted vote 82.3.
- Liu et al.: voting is robust and Max is fragile.
- Dalal et al.: with a noisy scorer the useful beam width is 1.

A policy that hedges instead of committing is therefore the natural answer to a weak decoder signal. One caution
(Uesato et al.): continuations from per-step candidates can all make the same later mistake.

**Recommended design: "spine plus hedges".**
1. The current single-select head picks an anchor at each step, which continues the spine.
2. Bernoulli hedge gates may keep extra candidates. Each kept extra finishes greedily as its own chain, so it is exactly
   a Stage 0 branch.
3. The answer is a summed vote over the spine and every hedge, from every step level, with the spine winning ties.
4. The anchor head trains with the existing GRPO objective. The gates get exact counterfactual credit, because removing a
   hedge from the vote costs no model calls. A width penalty is set by dual ascent.

The always-candidate-0, no-hedge policy is the checkpoint exactly.

**How this builds on the current plan.** It runs after that plan, which stays as written.
- **Its spine is Stage 1's selector.** If Stage 1 stops, the extension can still run with greedy as the spine, the
  hedges-only version.
- **Its offline checks reuse Stage 0's records.** Stage 0's branches already are hedges off a greedy spine, so whether
  multi-select is worth building can be decided without training (Section 5).
- **Count compute in token-forwards for its comparisons.** Decoder reads dominate. Single-select with K = 4 and decoder
  features at every step costs about 247 token-forwards, close to majority-of-19 chains. Stage 0 already records 18
  chains, so majority-of-18 is available.
- **Uesato and Dalal are worth keeping in mind when reading Stage 1.** With a weak scorer, per-step argmax can do worse
  than greedy. The extension's hedge-and-vote baselines then show whether voting does better.

## 2. Prior work

### 2.1 Variable-size selection policies and adaptive-width search (question 1)

The "Fit for us" column is inference; the rest is reported.

| Paper | Venue/date | How it chooses a subset or width | Key numbers (reported) | Credit and compute cost | Fit for us | Read |
|---|---|---|---|---|---|---|
| [Lei et al., rationales](https://arxiv.org/abs/1606.04155) | EMNLP 2016 | Independent Bernoulli mask over tokens; log-probability is the sum of Bernoulli log-probabilities; size is free | Beer reviews: 96.3% precision with 14% of text kept (attention baseline 80.6%) | REINFORCE on loss + λ₁‖z‖ + λ₂·Σ\|z_t−z_{t−1}\| | Simplest valid keep-gate form | [full] |
| [BlockDrop](https://arxiv.org/abs/1711.08393) | CVPR 2018 | K Bernoulli keep/drop gates chosen at once | Frozen ResNet-110, CIFAR-10, policy only: 75.4% against 93.2% for the full net; 93.6% after joint fine-tuning | Reward 1−(used/K)² if correct, else −γ; baseline is the greedy (threshold 0.5) policy's reward; curriculum on the last gates first | Baseline and curriculum transfer. Warning: a policy over a frozen backbone was weak | [full] |
| [PonderNet](https://arxiv.org/abs/2107.05407) vs ACT | ICML 2021 workshop | Per-step halting probability gives a distribution over the number of steps | ACT fails on parity for τ ≥ ~3e-3; PonderNet works for prior λ_p in 0.1–0.8 | β·KL to a geometric prior instead of a per-step cost | Argues for a width prior or Lagrangian, not a raw per-candidate cost | [partial] |
| [Kool et al. 2019](https://arxiv.org/abs/1903.06059), [2020](https://arxiv.org/abs/2002.06043) | ICML 2019, ICLR 2020 | Gumbel-top-k (sampling without replacement); an unordered set's probability is the sum of Plackett–Luce probabilities over its k! orderings | Gradient log-variance at k = 4: 6.29 and 7.05, against 11.52 and 13.81 for plain REINFORCE | Unbiased set estimator with a built-in leave-one-out baseline | k is fixed. With K = 4 every subset's probability is exact by enumeration | [full] (2019 results are in figures) |
| [Xie & Ermon](https://arxiv.org/abs/1901.10517) | IJCAI 2019 | Relaxed top-k via successive softmaxes | IMDB explanation task 91.7 vs 90.7 | Reparameterized, not REINFORCE | Low: our 0/1 reward is not differentiable | [full] |
| [Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO) | ICLR 2022 | Gumbel-top-k picks m actions, then Sequential Halving spreads simulations over them | Learns 9×9 Go with 2 simulations, where MuZero fails at ≤16 | Policy target softmax(logits + σ(completed Q)) over all considered actions | Good soft target for the anchor head; it outputs one action, not a set | [partial] (via a text mirror) |
| [SlateQ](https://arxiv.org/abs/1905.12767) | IJCAI 2019 | Slate value = Σ P(item chosen)·item value; slate chosen by LP, top-k or greedy | Simulation 174.6 vs random 159.2 | Item-level TD | Decomposition idea only; the slate size is fixed | [partial] |
| [MIRAGE](https://arxiv.org/abs/2609.21554) | arXiv Sep 2026; labelled as an ICML 2025 workshop paper, a mismatch | Bernoulli mask over 20 perspectives, with at least one forced; frozen LLM | Qwen2.5-7B GSM8K 79.8 → 89.0 with 1.97 calls; single runs; loses to an ensemble baseline in several cells | Reward 1[correct] − 0.05·k/20; REINFORCE; mean k ≈ 8/20 | Closest analogue to our gate; it decides once per question | [full] |
| [TreeRL](https://arxiv.org/abs/2506.11902) | ACL 2025 | Branches at the highest-entropy tokens; branch counts fixed | Qwen2.5-14B, 6-benchmark average 41.6 → 44.5 | Node value = share of correct leaves below it; advantage (global + local)/√(number of leaves) | Tree advantages transfer directly | [full] |
| [ARPO](https://arxiv.org/abs/2507.19849) | arXiv Jul 2025 | Branches when entropy rises: α + β·ΔH > τ, within a global budget | Qwen2.5-7B 56.5 → 58.3, with about half the tool calls of GRPO | Shared-prefix tokens get the averaged advantage | Disagreement-triggered widening is the training-free baseline | [partial] |
| [TreePO](https://arxiv.org/abs/2508.17445), [TreeRPO](https://arxiv.org/abs/2506.05183) | arXiv 2025 | Segment trees with the budget spread by softmax(log-prob/2); fixed 8-ary trees | GRPO 46.6 → 58.2 (maj@16); 32.7 → 35.6 | Nested-subgroup and sibling-group advantages | Credit recipes | [full] |
| [Damani et al.](https://arxiv.org/abs/2410.04707) | arXiv Oct 2024 | An MLP on the hidden state predicts the marginal gain from another sample; greedy knapsack | 25–50% less compute on math at B ≥ 8; up to 10% better at a fixed budget | Supervised predictor plus a hard budget | Supervised alternative to RL for "how many to keep" | [full] |
| [When to Stop](https://arxiv.org/abs/2511.21581) | arXiv Nov 2025 | Linear stop head per latent step on CODI (Llama-1B) | 49.73% → 50.11% with 52.9% fewer latent tokens | GRPO with 0/1 reward plus a group-relative length term | Closest latent-CoT precedent; it controls depth, not width | [partial] |
| [APR](https://arxiv.org/abs/2504.15466) | COLM 2025 | The model learns spawn() and join() of child threads | Countdown 83.4% vs 60.0%. RL raised the average number of children from 6.1 to 8.2 | Correctness reward only, no width cost | Without a width cost, RL widens | [partial] |
| [AB-MCTS, "Wider or Deeper?"](https://arxiv.org/abs/2503.04412) | arXiv Mar 2025 | A "generate a new child" option at every node; Thompson sampling with Beta or Gaussian posteriors picks wider or deeper | 128 calls: CodeContest 40.6 vs repeated sampling 37.9; ARC-AGI with DeepSeek-V3 18.3 vs 18.6 | External scorer; cost counted in API calls | Fits a test-time "draw another rerun here" (adaptive K). Gains are 1–3 points | [partial] |
| [Progressive widening (Coulom)](https://www.remi-coulom.fr/Amsterdam2007/) | ICGA J. 2007 | The n-th move is added after t_n simulations, t_{n+1} = t_n + 40·1.4ⁿ | 19×19 Go win rate 0% → 37.5% | Fixed schedule | Concept only. Double progressive widening (Couëtoux 2011) is [unverified] | [partial] |
| [REBASE](https://arxiv.org/abs/2408.00724) | arXiv Aug 2024 | Children per node = Round(B·softmax(PRM score/0.1)) | Mistral-7B MATH500: 45.0% with 32 samples vs 42.8% with 256 for sampling; GSM8K 89.1 → 89.8 | Budget B shrinks as solutions finish | Template for slot allocation | [full] |
| [ETS](https://arxiv.org/abs/2502.13575) | arXiv Feb 2025 | Integer program over leaves: REBASE weight − λ_b·KV cost + λ_d·cluster coverage | KV cache 1.8× smaller at width 256; 52.0 → 52.8 | Explicit KV and diversity terms | Precedent for a KV-cost and diversity term | [full] |
| [Snell et al.](https://arxiv.org/abs/2408.03314), [DVTS blog](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) | arXiv Aug 2024; blog Dec 2024 | Beam (keep top N/M, M = 4); DVTS splits into independent subtrees | Up to 4× more efficient than best-of-N; beam hurts on easy questions at large budgets; DVTS is better at large N | PRM over-optimization | Search helps most at medium difficulty, which matches our 41% base | [full] / [partial] (the blog's numbers are in plots) |
| [Dalal et al., "More Test-Time Compute Can Hurt"](https://arxiv.org/abs/2603.15377) | arXiv Mar 2026 | Theory of selection bias: overestimation ≈ σ√(2 log(n−1)); useful width k̂ grows with exp((Δ/σ)²) | Perplexity scorer gives k̂ = 1 (search never helps); PRM gives k̂ ≥ 4 (up to +8.9 points). Self-scoring at k = 1 → 4: Llama 59.7 → 54.3 | — | A noisy selector over similar candidates makes wider argmax worse | [full] |
| [VinePPO](https://arxiv.org/abs/2410.01679), [PAV](https://arxiv.org/abs/2410.08146) | arXiv Oct 2024 | Monte Carlo values from 9 rollouts per step; advantages under a prover policy | Up to 3.0× less wall-clock; more than 8% over an outcome verifier in search | Dense per-step advantages | Cheap with a frozen GPT-2 | [partial] |

**Reported, on credit and cost.** These are the mechanisms the papers above use:
- A baseline from the greedy policy (BlockDrop) and a leave-one-out set baseline (Kool 2020).
- Node values from the share of correct leaves (TreeRL, TreeRPO).
- Penalties set as a cost per kept item (MIRAGE, Lei), a KL to a prior (PonderNet), an explicit KV cost (ETS), or a
  shrinking budget (REBASE).
- Free width widens under RL (APR). A raw per-step cost is fragile (ACT).

**Inference, on which of these fit a frozen generator, a small selector and a 0/1 reward.**
- K = 4 gives only 15 non-empty subsets, so Plackett–Luce and Gumbel machinery is unnecessary. An exact categorical, or
  an anchor plus Bernoulli gates, gives exact log-probabilities.
- Allocation methods (REBASE, ETS, AB-MCTS) assume a usable scorer, and the scorer is our weak point.
- A policy over a frozen backbone may simply not beat the default (BlockDrop's 75.4% before fine-tuning).
- The most useful transferable pieces are three:
  - the greedy-policy baseline;
  - tree or Monte Carlo node values;
  - a target-width Lagrangian.

### 2.2 Merging continuous candidates into one state (question 2)

| Paper | Venue/date | Merge operator | Key numbers (reported) | Pitfalls reported | Relevance to our merge (inference) | Read |
|---|---|---|---|---|---|---|
| [Zhu et al., Reasoning by Superposition](https://arxiv.org/abs/2505.12514) | NeurIPS 2025 | A thought is (1/√\|V_c\|)·Σ u_v, a unit-norm sum over frontier nodes, not a mean | A 2-layer Coconut is near-perfect on ProsQA-style reachability; 12-layer CoT 83% | The superposition is learned through a curriculum | The precedent for "one vector holds many" uses √n normalization and trained, near-orthogonal codes | [full] |
| [Emergence of Superposition](https://arxiv.org/abs/2509.23365) | ICLR 2026 (per arXiv) | Same setting, training dynamics | The index-matching logit rises, then stays bounded, which keeps several traces weighted | — | Superposition is a trained balance, and our frozen model never had that pressure | [abstract] |
| [Coconut](https://arxiv.org/abs/2412.06769) | Dec 2024 | Last hidden state fed back; read as an implicit BFS | ProsQA 97.0 vs CoT 77.5; GSM8K 34.1 vs CoT 42.9; frontier "values" 0.33 at step 1 and 0.87 at step 2 (§4) | Without the curriculum, GSM8K 14.4 | BFS-like behaviour is shown on branching ProsQA, not on GSM8K | [full] |
| [CoT2](https://arxiv.org/abs/2505.23648) | ICLR 2026 (per arXiv) | Sample K vocabulary tokens and average them (or a Dirichlet); GRPO ratio = (∏ per-sample ratios)^(1/K) | MNNS: SFT 43.50; K = 1 51.61; K = 3 55.66; K = 6 50.38. GSM8K, Qwen3-0.6B: 41.3 vs 37.0 (single run) | Gains saturate in K; 1/K logit scaling shifts the distribution | Closest "sample K, average, RL" recipe, but it trains the model to read the averages | [full] |
| [Soft Thinking](https://arxiv.org/abs/2505.15778) | May 2025 | Σ p_k e_k over the top-n tokens, training-free | QwQ-32B math average 83.84 → 86.32. **Uniform average of the top-5 embeddings: AIME 6.66 vs 83.33. Raw hidden state fed back: 0.0** | Off-distribution inputs; repeats to max length | Uniform averaging of distinct candidates in a frozen model is catastrophic | [full] |
| [LLMs are Single-threaded Reasoners](https://arxiv.org/abs/2508.03440) | Aug 2025 | Soft Thinking, then Dirichlet or Gumbel noise | The soft output ≈ the top-1 component's output; vanilla soft < token sampling on all 3 models (QwQ 80.06 vs 82.35); Gumbel τ = 0.5 gives 83.04 | "Greedy pitfall" | A deterministic weighted merge collapses onto the dominant member, for us candidate 0 | [full] |
| [Soft Tokens, Hard Truths](https://arxiv.org/abs/2509.19170) | Sep 2025 | Mixture at τ = 0.5 plus Gaussian input noise; RLOO with a Gaussian-density log-probability | Llama-3B GSM8K pass@1: hard 75.9, soft 77.2; Llama-8B trained hard gets MATH 20.2, soft 44.7 | Soft *inference* is worse (74.5 vs 77.2) | Mixing helps as training-time exploration, not as an inference operator | [full] |
| [Mixture of Inputs](https://arxiv.org/abs/2505.14827) | May 2025 | w = (H·p + (β+1−H)·onehot)/(β+1), anchored on the sampled token | +1.8 average over 16 model–task pairs. **Pure mixture hurts:** QwQ 60.96 vs 72.86; Nemotron LiveCodeBench 39.92 → 16.04 | β is task-dependent | Anchor any merge on one actually chosen candidate | [full] |
| [Multiplex Thinking](https://arxiv.org/abs/2601.08808) | arXiv Jan 2026 | Average of K sampled one-hots, probability-reweighted; RL log-probability = sum over the K samples; GRPO | 7B MATH500: 71.6 CoT, 74.1 discrete RL, 78.0 multiplex. Uniform ≈ weighted (AIME24 19.9 vs 20.6) | "Interference between superposed paths"; smaller gains at 1.5B | A ready RL formulation for "select a subset and merge", but it trains the generator | [full] |
| [HRPO](https://arxiv.org/abs/2505.18454) | NeurIPS 2025 | Gate a·e_tok + √(1−a²)·(i⊙h), starting at a ≈ 1 | GSM8K 1.5B 0.711 → 0.720 | Raw hidden states get zero reward; pure interpolation collapsed after a few hundred steps | Variance-preserving gate that starts at the anchor | [partial] |
| [Superposed Decoding](https://arxiv.org/abs/2405.18400) | NeurIPS 2024 | Σ γ_i z_i over k drafts' embeddings in a frozen LLM | Llama-2-7B "sufficiently linear up to 10 timesteps" (cos ≥ 0.6) | Linearity is imperfect; needs an n-gram filter | Frozen models handle superposed inputs only approximately | [partial] |
| [Continuous Mixture Collapse](https://arxiv.org/abs/2609.02049) | arXiv Sep 2026 | Theory of mixtures fed back through the softmax readout | Pretrained rollouts lie mostly on the amplifying side, so one component comes to dominate | Exact preservation needs a context-dependent correction | A fed-back merged latent may snap to one member | [abstract] |
| [Model soups](https://arxiv.org/abs/2203.05482), [REPAIR](https://arxiv.org/abs/2211.08403) | ICML 2022; arXiv 2022 | Uniform or greedy weight averaging; rescaling after interpolation | CLIP B/32: best single 80.38, uniform soup 79.97, greedy soup 81.03. REPAIR cuts the barrier by 60–100% | Averaging incompatible members; "variance collapse" of averaged activations | Admit only compatible members; renormalize after merging | [partial] / [abstract] |
| [Srivastava et al., Dropout](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) | JMLR 2014 | Monte Carlo averaging over dropout masks vs weight scaling | §7.5, MNIST: about 50 sampled nets match weight scaling (dropout off) | — | Averaging dropout reruns drifts back toward candidate 0 | [partial] (§7.5) |
| [Token Merging (ToMe)](https://arxiv.org/abs/2210.09461) | ICLR 2023 [unverified venue] | Size-weighted average; softmax(QKᵀ/√d + log s) ("proportional attention") | ViT-L MAE: weighted average 84.25, keep-one 81.01, prune 79.2. Proportional attention: AugReg 82.15 → 83.51, MAE 84.25 → 83.84 | Needed for supervised models, not MAE | Opposite sign to KK-LSE (see below) | [full] |
| [CaM](https://proceedings.mlr.press/v235/zhang24n.html) | ICML 2024 | Merges values only, additively, into a kept entry; the evicted key is dropped | 20% budget, LLaMA-7B: RTE 49.5 → 54.6 | A wrong merge ratio can be worse than eviction | An additive merge preserves attention mass | [full] |
| [KVMerger](https://arxiv.org/abs/2407.08454), [D2O](https://arxiv.org/abs/2406.13035), [MiniCache](https://arxiv.org/abs/2405.14366) | arXiv 2024; ICLR 2025 (per arXiv); arXiv 2024 | Gaussian-kernel merge around a pivot; merge only above a cosine threshold; slerp of directions with the norms restored | LongBench: full 35.22, KVMerger 50% 35.02. MiniCache appendix: GSM8K 0.397 vs 0.025 for a plain mean | Plain averaging loses information; values are less similar than keys | Gate merges on similarity; restore norms | [full] / [partial] / [partial] |
| [KeepKV](https://arxiv.org/abs/2504.09936) | AAAI [unverified venue] | Votes p_i add on merge; logit + ln p_i | About 2.2× throughput | "Attention sag": a convex merge gets less mass than its parts | Fractional votes as an attention bias, the basis for the superposed step below | [partial] |
| [Deep Sets](https://arxiv.org/abs/1703.06114), [Set Transformer](https://arxiv.org/abs/1810.00825) | NeurIPS 2017; ICML 2019 | ρ(Σφ) sum pooling; PMA, attention pooling from learned seeds | Max-regression MAE: mean pooling 2.133, max 0.1355, PMA 0.2085 | — | Mean pooling cannot express "pick one"; attention pooling nearly can | [abstract] / [partial] |
| [Graph of Thoughts](https://arxiv.org/abs/2308.09687) | AAAI 2024 [unverified venue] | An LLM writes the aggregate of several thoughts | Sorting: about 62% better than Tree of Thoughts and more than 31% cheaper | All 10 sampled merges had errors; needs a refine step | Merging in text by a model that can reason about the merge, unlike our frozen latent step | [partial] |

**Does KK-LSE become relevant again? Yes, in principle, though the selector makes its estimate unnecessary.** This is
inference, and the derivation is in the q2 notes.
- **Copies inflate attention.** m identical copies of a cache entry (k, v) act on every later query like one entry with
  logit q·k + log m. So keeping all the selected candidates' entries side by side gives a near-duplicate set m times one
  entry's attention mass.
- **That is the mechanism this project measured.** Appending exact copies of latent entry 6 reproduced SIM-CoT's whole
  past-depth drift (172 at 24 entries, against 170 plain), and dedup_soft repaired it.
- **KK-LSE was inert at 6 steps** because no near-copies exist there. A superposed step creates them by construction.
- **Token Merging's +log s and KK-LSE's −α·log n are inverses.** Token Merging treats multiplicity as evidence and KK-LSE
  treats it as an artifact. The number of candidates the selector kept is an artifact, so KK-LSE's sign is the right one.
- **With explicit selection the multiplicity is known.** A bias of log w_i on each kept candidate's entries, with
  Σw_i = 1, gives the step exactly one entry's mass when the keys agree. When they differ, each later query re-weights
  the candidates. This is KeepKV's vote bias, and it reduces to the identity when only one candidate is kept.
- **KK-LSE or dedup_soft remains useful in two places:** as a training-free fallback when the weights are unknown, such
  as a late merge over many chains, and as a check that the log w bias behaves as derived.

### 2.3 Final aggregation over kept candidates (question 3)

| Paper | Venue/date | Aggregation mechanism | Key numbers (reported) | Width/compute | Relevance to "keep alive" (inference) | Read |
|---|---|---|---|---|---|---|
| [Uesato et al.](https://arxiv.org/abs/2211.14275) | arXiv Nov 2022 | Sum of RM probability per answer | GSM8K error, K = 96: greedy 41.1, majority 22.3, RM-weighted 14.1. Picking the best of K at each step was 1–3% *worse* than reranking whole solutions, as "all K continuations would make the exact same mistake" | — | Direct warning for per-step selection when candidates lack diversity | [full] |
| [DIVERSE](https://aclanthology.org/2023.acl-long.291/) | ACL 2023 | Verifier-weighted vote; step-aware verifier | code-davinci-002 GSM8K: majority 80.0, verifier best-of-N 65.9, weighted vote 82.3 | 100 paths | A weak verifier makes argmax dangerous, while weighted voting stays safe | [full] |
| [Lightman et al.](https://arxiv.org/abs/2305.20050) | arXiv May 2023 | PRM path score = product of step P(correct) | MATH best-of-1860: PRM 78.2, ORM 72.4, majority 69.6; product 78.2 vs min 77.6. RM-weighted voting "did not noticeably improve" | — | The path reduction barely matters | [full] |
| [Math-Shepherd](https://arxiv.org/abs/2312.08935) | arXiv Dec 2023 | min over steps; SC+PRM sums per answer | LLaMA2-70B, 256 samples, GSM8K/MATH500: SC 88.0/39.4; PRM 93.2/44.5; SC+PRM 92.4/45.2 | — | Weighted vote wins on MATH, plain best-of-N on GSM8K | [full] |
| [Snell et al.](https://arxiv.org/abs/2408.03314) | arXiv Aug 2024 | Last-step PRM score | "last" beat min and product (App. E) | Beam width √N | "last" suits forward-looking (Monte Carlo) labels | [partial] |
| [Lessons of Developing PRMs](https://arxiv.org/abs/2501.07301) | arXiv Jan 2025 | last / product / min | Best-of-8: MC-trained PRM last 65.4 > min 64.9 > product 64.3; Qwen-PRM-7B product 67.6 > min 67.1 > last 66.1; maj@8 66.2 | — | Spread of at most about 1.5 points; pick one by prior and don't tune | [full] |
| [Liu et al., 1B vs 405B](https://arxiv.org/abs/2502.06703) | arXiv Feb 2025 | {Min, Last, Avg} × {Max, Vote} | MATH-500, Skywork PRM: majority 86.8; Min-Max 83.0; Min-Vote 86.6; Last-Vote 87.0 | — | **Vote is robust, Max is fragile** | [full] |
| [Self-certainty](https://arxiv.org/abs/2502.18581) | NeurIPS 2025 | Borda vote (N−r+1)^p | MATH N = 64: majority 63.40, Borda 64.10, self-certainty best-of-N 56.70. **Averaging weights per answer 32.94, summing 63.51** | — | Always sum weights per answer | [full] |
| [CISC](https://arxiv.org/abs/2502.06233) | Findings ACL 2025 | softmax(c/T)-weighted vote | 41–46% fewer samples than self-consistency; within-question ranking accuracy (52–62%) predicts the gains, calibration does not | — | T moves between majority (T→∞) and argmax; measure within-question ranking first | [full] |
| [DeepConf](https://arxiv.org/abs/2508.15260) | arXiv Aug 2025 | Confidence-weighted vote after keeping the top η% | GPT-OSS-120B AIME25: maj@512 97.0 → 99.9; hurts in 8 of 20 settings (down to −4.7) | Early stop, up to −84.7% tokens | Filtering can backfire | [partial] |
| [Universal Self-Consistency](https://arxiv.org/abs/2311.17311) | arXiv Nov 2023 | An LLM picks the most consistent response | PaLM2-L GSM8K: SC 90.4, USC 90.2 | 8 samples | Prompted selection only matches voting | [full] |
| [AggLM](https://arxiv.org/abs/2509.06870) | arXiv Sep 2025 | GRPO-trained aggregator reads 8 solutions | Qwen3-1.7B solutions, AIME25: majority 45.89, AceMath-weighted vote 39.49, AggLM 50.00; best training mix is all hard questions plus 50% of easy ones | About 1/3 of the solution tokens | Gains come where the right answer is a minority; off-the-shelf RM weights lost to majority | [full] |
| [Sample Set Aggregator](https://arxiv.org/abs/2506.09014) | arXiv Jun 2025 | 0.5–3B RL reader over K concatenated solutions | K = 5: GSM8K majority 91.66 → 93.25; MATH 68.20 → 76.80; RL beats SFT at 3B (55.2 vs 48.1) | Tournament for K ≥ 32 | Template for a learned reader; capped by pass@K; small on GSM8K | [partial] |
| [GenSelect](https://arxiv.org/abs/2507.17797) | ICML 2025 AI4Math workshop | An LLM picks one of N summaries; knockout rounds | QwQ: maj@64 68.4, GenSelect 73.4; worse than sampling at N = 4 for R1 | Tournament | Small-N caution | [full] |
| [Mixture-of-Agents](https://arxiv.org/abs/2406.04692) | arXiv Jun 2024 | Layered synthesis | AlpacaEval LC 65.1; a ranker does worse than synthesis; same-model diversity is weaker (56.7 vs 61.3) | 3 layers × 6 agents | MC dropout gives the weaker same-model kind of diversity | [full] |
| [Parallel-R1](https://arxiv.org/abs/2509.07980) | arXiv Sep 2025 | Model-decided `<Parallel>/<Path>/<Summary>` | Qwen3-4B average 45.1 → 48.9. Share of outputs that go parallel: 13.6% with an accuracy-only reward, 80.3% with a parallel-only reward, 63.0% alternating | Parallel bonus, alternating schedule | Width can be steered by the reward | [full] |
| [ParaThinker](https://arxiv.org/abs/2509.04475) | arXiv Aug 2025 | P paths with branch embeddings on K/V; one summary decode reads every path's cache | 1.5B: majority 60.4 vs 63.2. Lead over majority shrinks with P (7B: +3.9, +1.6, +0.6). Naive re-reading of all paths falls 47.1 → 35.7 as P goes 2 → 8 | P fixed; SFT | Closest precedent for one decode over all kept caches, and it needed training | [full] |
| [Multiverse](https://arxiv.org/abs/2506.09991) | arXiv Jun 2025 | Map-process-reduce; paths share a start position; merged caches | 32B AIME24 53.8 vs 54.6 autoregressive | SFT | Mask and position-id recipe | [full] |
| [Hogwild! Inference](https://arxiv.org/abs/2504.06261), [Group Think](https://arxiv.org/abs/2505.11107) | NeurIPS 2025; arXiv May 2025 | Shared KV cache, no training; token-interleaved threads | Hogwild about 1.8× / 3.4× faster with 2 / 4 workers, and **fails on Qwen3-1.7B**; Group Think reports latency only | — | An untrained shared cache is unlikely to work at 124M | [partial] / [full] |
| [Native Parallel Reasoner](https://arxiv.org/abs/2512.07461) | arXiv Dec 2025 | Plan, step, takeaway; one advantage shared by all branches | Qwen3-4B-Instruct 65.0 vs 62.0 sequential RL | No width penalty | Simplest credit baseline | [full] |
| [REBASE](https://arxiv.org/abs/2408.00724) | arXiv Aug 2024 | Weighted vote at the leaves | Mistral-7B MATH, majority / best-of-N / weighted: sampling 36.1 / 40.3 / 39.7; tree 44.1 / 45.4 / 45.0 | Shrinking budget | After tree search the voting rules converge | [full] |
| [Forest-of-Thought](https://arxiv.org/abs/2412.09078) | Dec 2024 | Majority across trees, an expert LLM when they disagree | GSM8K n = 5: 83.44 → 85.44 | n trees | A fallback for split votes | [full] |
| [Beyond the Last Answer](https://arxiv.org/abs/2504.20708) | arXiv Apr 2025 | Mode of answers completed from intermediate steps | Up to +13 points on AIME24 | — | Votes from several step levels | [abstract] |

**Reported takeaways.**
- Sum weights per answer; never average them.
- Voting is robust where argmax over a verifier is fragile.
- The choice of path reduction (min, last or product) moves results by at most about 1.5 points.
- Learned aggregators help most when the right answer is held by a minority, and are capped by pass@K.
- One decode over all kept caches needed trained branch embeddings in ParaThinker, and failed untrained in Hogwild.
- No paper found explicitly discounts leaves that share a prefix when voting.

### 2.4 Prior work in latent space (question 4)

| Paper | Venue/date | Models | Multi-candidate mechanism | Key numbers (reported) | What it leaves open (inference) | Read |
|---|---|---|---|---|---|---|
| [LatentTTS](https://aclanthology.org/2026.acl-long.2069.pdf) ([arXiv](https://arxiv.org/abs/2510.07745)) | ACL 2026 | Coconut and CODI GPT-2; CoLaR 1B | **Beam:** B = √N beams, B dropout expansions each, keep the top B by cumulative length-normalized LatentRM reward, final argmax. Also best-of-N and majority. LatentRM is trained from 128-rollout success rates with a step-wise contrastive loss; p = 0.2 | Coconut GSM-Test: greedy 34.0; one sampled chain ≈31.3; coverage@4 46.7; best-of-4 34.5; best-of-8 35.4 vs majority-of-8 33.6. Best-of-64 ≈35.5, majority-of-64 ≈34.2, beam-64 (B = 8) ≈35.5. GSM-Hard: beam ≈7.75 vs best-of-N ≈8.2, blamed on "early-step score noise". **CODI: coverage only** (greedy ≈42.5, coverage@64 ≈63.5), and its coverage collapses past diversity ≈0.15 | No beam or aggregation on CODI or SIM-CoT, and no CODI reward model. B = K, so keeping 1 of K was never tested. Supervised scorer, not RL | [full] |
| [ITS-Continuous](https://arxiv.org/abs/2510.12167) | Findings ACL 2026 | Coconut GPT-2 | Dropout chains reranked by PRM, ORM, confidence and self-consistency | Pass@1 31.08, pass@16 42.61, pass@32 44.43; best rerank (PRM-HE) 33.36 at N = 16; SC 32.15. PRM precision 41.6 | "Lack the geometric separability"; a decoder-based reader is untested | [full] |
| [Cui et al.](https://arxiv.org/abs/2602.22441) | arXiv Feb 2026 | Coconut, CODI, SIM-CoT, CoLaR GPT-2 | Latent reasoning once; the final answer sampled 100 times | GSM8K greedy / pass@100 / maj@100: CODI 43.59 / 70.43 / 42.23; **SIM-CoT 42.23 / 69.60 / 43.21** | "Increased outcome-level diversity alone is insufficient to guarantee effective aggregation" | [partial] (Table 4 checked) |
| [NoisyCoconut](https://arxiv.org/abs/2605.08221) | arXiv May 2026 | Off-the-shelf 7–20B LLMs in a hidden-state loop | K = 5 noisy paths; majority or unanimity, with abstention | Qwen GSM8K: base 71.0; ≥3/5 78.2; zero-noise control 76.4; K = 10 80.2. Unanimous: 91–95% accuracy at 31–53% coverage | Exact-match voting only; most of the gain survives zero noise | [full] |
| [SoftCoT++](https://arxiv.org/abs/2505.11484) | arXiv May 2025 | Frozen LLaMA-3.1-8B and Qwen3-8B with soft-thought prefixes | N diverse soft prefixes (contrastive loss), reasoning in tokens, majority vote | N = 10 average: LLaMA 77.57 vs SoftCoT-SC 76.88 vs token SC 75.74; contrastive ablation +0.1 to 0.4 | Diversity only in a static prefix; nothing per step | [full] |
| [LaDi-RL](https://arxiv.org/abs/2602.01705) | arXiv v3 May 2026 | Diffusion latent block; R1-Qwen-7B, Qwen3-8B | N latents × M answers per latent, **training only**; latent advantage = standardized per-latent mean reward; a repulsion term for diversity | Math pass@1 average 43.45 vs GRPO 37.73. Code: 4×4 77.1, 4×1 68.5, 1×4 63.5, GRPO 4×4 67.7 | Two-level credit, not stepwise; no test-time keep or merge | [full] |
| [LTC](https://arxiv.org/abs/2608.01593) | arXiv Aug 2026 | Qwen2.5-3B/7B, Gumbel-soft latents | K thoughts × M answers for RL credit; greedy evaluation | 3B GSM8K 84.46 vs GRPO 82.71; at budget 16, 4×4 85.15 > 2×8 84.52 > 8×2 83.10 | Within-thought variance exceeds between-thought variance | [full] |
| [LTO](https://arxiv.org/abs/2509.26314) | ICLR 2026 (per PDF header) | Huginn-3.5B | 20 trajectories from different noise seeds; latent classifier; acceptance-rejection over whole trajectories | GSM8K: base 0.326, majority 0.333, classifier-weighted majority 0.375, LTO 0.385; classifier AUC 0.884 | Latent states *can* separate in some architectures; no per-step selection | [full] |
| [GTS](https://arxiv.org/abs/2602.14077) | arXiv Mar 2026 | Coconut GPT-2, CODI Llama-1B | GRPO-trained Gaussian perturbation sampler on a **frozen backbone** | pass@N only (figures) | An RL head on a frozen latent model, but a proposer, not a selector | [full] |
| [SLPO](https://arxiv.org/abs/2607.19691) | arXiv v3 Oct 2026 | Coconut, CODI | RLOO over 8 dropout rollouts (p = 0.1); K = 4 dropout passes form a Gaussian surrogate for the likelihood | CODI GPT-2 P@8 / P@16: 52.08 / 54.36 → 54.13 / 56.71 | No selection or voting | [full] |
| [PLaT](https://arxiv.org/abs/2601.21358), [Dropout-GRPO](https://arxiv.org/abs/2606.10184) | arXiv 2026 | GPT-2/Llama; Coconut on Qwen-1.5B | Branching in the text readout, pass@k only; one dropout mask per RL rollout | CODI GPT-2 pass@1/32/128: 40.16 / 63.00 / 69.90; GSM8K 27.29 → 29.01 | Coverage "requires answer selection", which neither does | [full] |
| [C-voting](https://arxiv.org/abs/2604.13521) | ICLR 2026 (listing) | Recurrent puzzle models | Several latent initializations; keep the most confident | Sudoku-extreme 95.2 vs HRM 55.0 | Not language reasoning | [abstract] |
| [When to Stop](https://arxiv.org/abs/2511.21581) | arXiv Nov 2025 | CODI Llama-1B | GRPO stop head on a latent reasoner | See 2.1 | Precedent for a small GRPO head over CODI | [partial] |

Also checked, with no keep or merge: SIM-CoT (its decoder is dropped at inference), HRPO, CoLaR (deterministic and
sampled heads tie, 41.7 vs 41.3), LatentSeek and LTPO (optimize one latent at a time), PLRM and ATF (multi-sample, no
aggregation). Abstract only: RLTT, ATP-Latent, SWITCH, MUX, Latent-SFT.

**What is open (inference).**
- Per-step keep-1-of-K, keep-several-of-K, or merging of latent candidates on CODI or SIM-CoT.
- An RL-trained selector over latent candidates.
- A selector that reads candidates through a step decoder.
- Voting over a latent tree.

The bar every GPT-2 latent result has failed to clear is converting coverage into accuracy: +1.5 for LatentTTS
best-of-64, +2.3 for ITS's PRM, and about +1 for SIM-CoT majority@100 in Cui et al.

## 3. Design options

### 3.0 Facts that constrain every option

These come from the repo and our measurements; the consequences drawn from them are inference.

1. **Every candidate at a step shares its input.**
   - All K candidates at step s start from the same latent z_{s−1} and cache C_{s−1}. Candidate 0 runs with dropout
     off; the others draw MC-dropout masks (`stepsel.sample_step`).
   - Their step-s KV entries are therefore noisy copies of one computation. Their next latents may differ more.
   - The projection is Dropout → Linear → GELU → Linear → LayerNorm (`dsa/bench/models/codi.py:71–72`). So averaging
     pre-LayerNorm vectors and re-applying the LayerNorm removes the norm-shrinkage pitfall for free.
2. **Averaging dropout reruns approximates candidate 0** (weight scaling; Srivastava §7.5). Only a selective, weighted
   merge can move away from candidate 0.
3. **Near-copy KV entries dilute the readout.** At GPT-2 scale, copies of latent entries reproduce the past-depth drift,
   and dedup_soft repairs it. A per-entry additive attention bias already exists in `scripts/latent_gate/batched.py`
   and `scripts/inner_loop/kk.py`.
4. **Absolute positions allow siblings.** GPT-2's learned absolute position embeddings are added to the input in
   `latent_step`, so several cache entries at the same position are legal, and `latent_step` already accepts a key
   mask.
5. **Stage 0's branches are one-step deviations from greedy, each finished greedily.** That is exactly a hedge off a
   greedy spine (Section 4.1).
6. **Decoded steps are short.** They run 7–9 GPT-2 tokens plus end-of-text: 7 for `<<3*12=36>>`, 9 for
   `<<600*0.25=150>>`. So one decoder read costs T_D ≈ 9 token-forwards, and one answer readout T_A ≈ 6–7.

### 3.1 Policy parameterizations for a variable-size subset

K = 4 candidates per decision; S ⊆ {0,…,3} is non-empty.

| Form | Sampling and log π(S) | Identity start | Pros | Cons |
|---|---|---|---|---|
| **P1. Anchor + hedge gates** | Anchor a ~ Cat(softmax u); each other k kept with Bern(σ(g_k)). log π = log softmax(u)_a + Σ_{k≠a} log Bern(e_k; σ(g_k)) | Bias +5 on u_0, −6 on all g: P(a = 0) ≈ 0.98, P(no extras) ≈ 0.993; deterministic evaluation (argmax, gate > 0.5) is exactly candidate 0 | The anchor head *is* the current single-select head. A non-empty set by construction. Gate credit can be counterfactual (4.1) | Two heads to tune |
| P2. Exact categorical over the 15 subsets | Score(S) = Σ_{k∈S} u_k + size term b_{\|S\|} + redundancy term; softmax over 15 | Large bias on {0} | Exact; can price redundancy between members | No anchor, so no natural spine for (b) or for credit |
| P3. Size head + Plackett–Luce | n ~ Cat{1..4}, then a Gumbel-top-n ordered prefix; unordered probability by summing ≤ 24 orders (Kool 2020) | p(n = 1) high, u_0 high | Explicit width control | Size is decided before members; more variance |
| P4. Slot allocation (REBASE-like) | Draw W slots from Cat(softmax u); distinct draws are kept, and counts become merge weights or chains per candidate; exact multinomial log π | u_0 dominant | One action gives both membership and weights | Width ties to W; duplicate slots waste compute in (b) |
| P5. Relative threshold | Keep k if u_k ≥ max u − τ; τ ~ LogNormal(head) | τ small | Beam-pruning intuition | REINFORCE on τ only; coarse |

Forcing "at least one" outside the sampled distribution, as MIRAGE does, makes log π mismatch the action actually played.
P1 and P2 avoid this.

### 3.2 Option (a): merge now ("culmination step")

| Operator | What enters the next step | Identity when \|S\| = 1 | Literature for it | Main risk (inference) |
|---|---|---|---|---|
| **M1. Anchored latent merge** | z̃ = LN_proj(Σ_k w_k·pre-LN_k), w_anchor ≥ ½ (anchoring as in Mixture of Inputs), w over S from softmax(u/T); KV of step s from the anchor only | Exact | Mixture of Inputs (anchor); HRPO (start at the anchor); MiniCache and REPAIR (renormalize) | Collapse: the weight-scaling argument and the single-threaded finding say it stays near candidate 0. Off-manifold blur when members decode to different steps; SIM-CoT was trained for one step per latent, not superposition |
| **M2. Superposed step** (keep all members' KV) | All \|S\| candidates' entries at position q+s in every layer, with attention bias log w_k on each; next latent from M1 or the anchor | Exact (log 1 = 0) | KeepKV's votes; derivation in 2.2 | Without the log w bias, this is the measured dilution. With it, fact 1 makes the entries near-copies, so M2 ≈ the anchor's KV. The cache grows by \|S\|−1 entries per layer |
| M3. Learned pooling | A one-seed PMA over the members, zero-initialized residual on the anchor, followed by LN_proj. Trained by backprop through the frozen generator on teacher-forced gold-answer cross-entropy; RL only for S | Exact at initialization | Set Transformer; CoT2 and Multiplex show trained merges work | A free output map leaves the input manifold (raw hidden states fed back score 0.0 in Soft Thinking, zero reward in HRPO). It is no longer "selection", and it edits latents like an adapter |
| M4. KV averaging or slerp | One averaged entry per layer, norm restored | Exact | CaM, KVMerger, MiniCache | Little change for near-copies; averaged keys flatten attention |

**Verdict on (a) (inference).** Because the candidates share their input and averaging dropout reruns approximates
candidate 0, a merge of MC-dropout candidates of a frozen model is close to a soft version of single selection. The
only way it adds anything is through *which* members are merged. Evidence from frozen models is negative unless the
merge is anchored, and the positive results (Multiplex, CoT2) train the generator. M2 with the log w bias is the safe way
to keep alternatives visible to later queries, and the cheapest test of whether "superposition" helps a SIM-CoT readout
at all.

### 3.3 Option (b): keep alive, then aggregate

Tree structures:

| Structure | What branches | Cost relative to single-select | Credit | Main risk (inference) |
|---|---|---|---|---|
| **B1. Spine + hedges** | Only the spine proposes K candidates. A kept extra spawns a hedge that finishes steps s+1..6 with dropout off and reads its answer, with no further branching or decoder reads | + Σ_hedges [(6−s) + T_A], linear in the number of hedges | Exact counterfactual per gate (4.1) | Hedges share the spine's later steps, so they can repeat its later mistakes (Uesato) |
| B2. Capped beam | Every live chain proposes K; anchor and gates per node; global cap B ≤ 4 by score | ≈ B× (each live chain pays proposals and decoder reads) | TreeRL node values | Scorer noise early (LatentTTS's GSM-Hard loss; Dalal's k̂); cost |
| B3. Adaptive widening (AB-MCTS-style) | At test time draw more reruns at a node where candidates disagree; Thompson sampling decides wider or deeper | Variable | Test-time only | Needs a usable value signal; gains of 1–3 points in text |

Aggregators, "taking every kept candidate at every step level into consideration":

| Aggregator | Definition | Identity | Evidence | Main risk (inference) |
|---|---|---|---|---|
| **V1. Summed vote, spine breaks ties** | score(y) = Σ_{chains c with answer y} w_c, with w_spine = 1+ε and w_hedge = 1; later w_c = exp(v(ψ_c)/T) | One chain gives its own answer | Robust voting (Liu, DIVERSE); summing beats averaging (self-certainty) | With few hedges, one hedge can never beat the spine, so a rescue needs at least 2 agreeing hedges |
| V2. Level-balanced tree vote | The root holds mass 1; each node splits its mass over its kept children in proportion to q; leaves pass their mass to their answer, so each depth sums to 1 | Same | DVTS subtrees; TreeRL's ÷√leaves; nothing does exactly this | Mostly matters for B2 |
| V3. Learned reader | A 1–5M-parameter set transformer over leaf summaries (answer id, path features, final hidden state), outputting a logit per leaf, summed per answer; trained supervised, then GRPO | Single leaf | SSA (+1.6 on GSM8K, +8.6 on MATH), AggLM | Relearns majority, or overfits answer priors |
| V4. One decode over all kept caches | The answer decode attends to the union of kept entries under a tree mask, with siblings at the same position and an optional log w bias | Single chain | ParaThinker (trained), Multiverse | With a frozen generator: the measured copy dilution, Hogwild's failure at 1.7B, ParaThinker's 47.1 → 35.7. Not recommended |

**Votes from earlier step levels.** An answer can also come from decoding at depth s < 6, an early exit. That is
untested for SIM-CoT, which was trained for exactly 6 steps (Section 6). The cheaper way to use every level is that each
level's hedges finish to step 6 and vote.

### 3.4 Side by side

| | (a) M1/M2 merge | (b) B1 spine + hedges | (b) B2 capped beam |
|---|---|---|---|
| Generator stays in distribution | M1 no (mixed latents); M2 mostly | Yes: each chain is a normal chain with one dropout step | Yes |
| Extra inference cost over single-select | ≈ 0 | Small and linear | ≈ (B−1)× |
| Credit signal for "keep more" | Rollout-level only, or one extra no-merge rollout | Exact counterfactual per gate | Node values |
| Uses "all step levels" | Implicitly, through the merged state | Yes, one vote per kept candidate per level | Yes, through the leaves |
| Literature support for frozen models | Negative to weak | Voting is robust; MIRAGE-like gates | LatentTTS: beam ≈ best-of-N |
| Stage 0 data can test it offline | No (needs new runs) | **Yes, for a greedy spine** | Partly (one-step deviations only) |

## 4. Recommended design and alternatives

### 4.1 Recommended: spine plus hedges (P1 + B1 + V1)

**Structure.**
- At latent step s the spine proposes K = 4 candidates, as in Stage 1.
- The policy picks an anchor a_s, which continues the spine, and a set of hedges E_s from the other candidates.
- A hedge k at step s runs steps s+1..6 from (z_s^k, C_s^k) with dropout off, then reads its answer. This is exactly a
  Stage 0 branch.
- Hedges never propose candidates and never need decoder reads.
- The final answer is V1 over the spine and every hedge kept at every step.

**Policy features.** Per candidate k at step s, all cheap:
- the projected hidden state (a 64-dimensional linear projection of the final hidden state);
- the decoder features: parsed, valid, grounded, mean log-probability;
- relational features: does its decoded result equal candidate 0's, cos(z_s^k, z_s^0), and does its result appear among
  the question's numbers or earlier results;
- the step index;
- the number of hedges kept so far.

The anchor head is the planned Stage 1 selector. The gate head is a second small MLP over [φ_k, φ_a].

**Objective.**
- **Anchor head.** Unchanged GRPO on R_spine = 1[spine answer correct], so the single-select result is a sub-result of
  this design, plus a KL to the initial policy.
- **Gate head: exact counterfactual credit.** During training, run hedges for *all* extras at every spine step. That is
  3 × 6 = 18 hedges per rollout, the Stage 0 branch computation.
  - Given the other decisions E_{−k}, gate k's expected reward is
    p_k·R(E_{−k} ∪ {k}) + (1−p_k)·R(E_{−k}) − λ·c_k·p_k, with p_k = σ(g_k).
  - So its exact gradient is ∂/∂g_k = p_k(1−p_k)·[R(E_{−k} ∪ {k}) − R(E_{−k}) − λ·c_k].
  - Here R is the 0/1 correctness of the vote and c_k = (6−s) + T_A is the hedge's cost.
  - Recomputing the vote without hedge k needs no model call. This is a local-expectation gradient (Titsias 2015)
    and a COMA-style counterfactual baseline (Foerster et al.): zero variance for each gate given the others, where
    REINFORCE over 2^18 hedge patterns would be very noisy.
  - At deployment only the kept hedges run.
- **Width penalty.** λ by dual ascent to a target mean number of hedges H*: λ ← max(0, λ + η(H̄ − H*)). Sweep
  H* ∈ {0.5, 1, 2, 4} and report the accuracy-against-cost curve. A KL to a width prior centered on zero hedges
  (PonderNet) is an alternative.
- **Vote weights.** Start with plain V1 (ε = 0.01). Then fit v(ψ) as a logistic model of hedge correctness within the
  question, with T tuned on held-out data, and sum per answer. With T → ∞ it falls back to plain majority.

**Identity.** With u_0's bias at +5 and every gate bias at −6, deterministic evaluation keeps candidate 0 and no hedges
at every step. That reproduces greedy exactly, on 500/500 as Stage 0's identity check requires. A vote over one chain
returns that chain's answer.

**Compute per question.** Formulas in token-forwards, with T_D ≈ 9 and T_A ≈ 7 (fact 6); details in `design_notes.md`.

| Method | Latent steps | Decoder tokens | Readout tokens | Total | ≈ majority of N chains (13 each) |
|---|---|---|---|---|---|
| Greedy | 6 | 0 | 7 | 13 | 1 |
| Single-select K = 4, decoder at every step (current Stage 1) | 24 | 24·9 = 216 | 7 | ≈ 247 | 19 |
| Single-select, decoder at steps 1–3 only | 24 | 108 | 7 | ≈ 139 | 11 |
| Single-select, hidden-state features only | 24 | 0 | 7 | 31 | 2.4 |
| **Spine + hedges**, h hedges at mean step s̄ | 24 + h(6−s̄) | 216 | 7(1+h) | ≈ 247 + h(13−s̄); h = 2 at steps 1–2: ≈ 270 | 21 |
| Spine + all 9 hedges at steps 1–3 | 60 | 216 | 70 | ≈ 346 | 27 |
| Capped beam B = 4 | ≈ 96 | ≈ 864 | 28 | ≈ 988 | 76 |
| Merge now (M1/M2) | 24 | 216 | 7 | ≈ 247 | 19 |
| Training-time counterfactual hedges (per rollout) | +45 | 0 | +126 | +171 | — |

What the table says (inference):
- **The decoder dominates the cost of every decoder-fed selector.** Restricting reads to steps 1–3, where reads are
  reliable anyway, nearly halves it.
- **Hedges are cheap relative to that.** Spine + hedges costs about 10% more than single-select at H* = 2.
- **Wall-clock is a different count.** Decoder reads are short-context and batch well, so wall-clock differs from
  token-forwards. Both should be reported.
- **Training is cheap.** At G = 8 rollouts and 171 extra token-forwards each, a GPT-2-small run is not compute-bound.

**Matched-compute baselines.** Each is at the same token-forwards, and also at the same generator latent steps, as the
current plan counts.
1. Greedy, the identity.
2. Majority over N MC-dropout chains, at N = C/13 (about 19–21), and at N = 4 for the generator-only count.
3. Decoder-weighted vote over N chains, training-free; N is smaller because chains then pay for decoder reads.
4. Single-select RL (the current Stage 1) alone, and with a majority over its m stochastic runs at matched C.
5. Fixed hedging rules at the learned policy's mean hedge count:
   - all extras at steps 1–3;
   - disagreement-triggered, keeping extras whose decoded result differs from candidate 0's (the TreeRL/ARPO analogue);
   - random hedges drawn from the learned policy's per-step width distribution, which tests that *which* hedges matters.
6. A fixed-width beam with the decoder rule and no learning (B2), the LatentTTS analogue.

**Ways it could fail (inference).**
- **No votes to win.** Rescues where the spine is wrong usually have one lonely correct hedge, never two agreeing ones.
  Then V1 cannot rescue, and only a confident anchor can. Check O2 in Section 5 first.
- **Shared mistakes.** Hedges finish greedily and repeat the spine's later errors (Uesato), so hedge answers mostly equal
  the spine's.
- **Weak hedges.** A single dropout chain is weaker than greedy: about 31.3 vs 34.0 for Coconut in LatentTTS. Votes
  drawn from weaker samples can sit below greedy, as Coconut's majority-of-8 at 33.6 did.
- **Width collapse.** λ drives the gates fully shut, or with a low λ fully open (APR). Dual ascent and the width sweep
  guard against this.
- **The win could be compute, not selection.** The gain would then be explained by the matched-compute majority
  baseline. That is the most likely null.
- **Seed noise of about 9/500.** A pooled effect under about +15 is unreadable. Two seeds and paired sign tests, as in
  the current plan.

### 4.2 Alternative A: capped latent beam with a level-balanced vote (P1 per node + B2 + V2)

- **Structure.** Every live chain proposes K, and the anchor and gates act per node. A global cap of B ≤ 3–4 keeps the
  top chains by anchor logit.
- **Credit.** Final V2 vote. TreeRL node values, the share of correct leaves below a node, give per-node advantages. The
  width penalty is a cost per live chain per step.
- **Identity.** As in 4.1.
- **Cost.** About B× single-select, mostly decoder reads. Reading only steps 1–3 helps here too.
- **When to prefer it.** If the offline checks show that hedges fail because of *later* errors shared with the spine.
  Hedges cannot fix those, and a beam can.
- **Evidence against it.** LatentTTS's beam only tied best-of-N on Coconut, and Dalal et al. predict a width of 1 for a
  noisy scorer.

### 4.3 Alternative B: anchored merge now (P1 + M1, optionally M2)

- **Structure.** The anchor and the selected extras merge into one state:
  - z̃ = LN_proj(Σ w_k·pre-LN_k), with w_anchor ≥ ½ and the others' weights from softmax(u/T);
  - the step's KV comes from the anchor, or from all members with a log w bias (M2).
- **Training.** The subset by GRPO on 1[correct]; the cost term is negligible. Optionally, the weights by backprop
  through the frozen generator on teacher-forced gold-answer cross-entropy.
- **Identity and cost.** Exact identity when S = {anchor}; cost ≈ single-select.
- **When to prefer it.** Only if the new-run check N1 in Section 5 shows that merged latents decode to coherent steps,
  and change answers on a meaningful fraction of questions without breaking them.
- **Expected value (inference).** Low. It is the most novel option, which could still make it worth a small, cheap
  test.

### 4.4 Not recommended

- **One answer decode over all kept caches with a frozen generator (V4).** This is the measured dilution in a new form,
  and both the trained (ParaThinker) and the untrained (Hogwild) precedents warn against it.
- **Uniform averaging of candidates (M1 with uniform w).** It returns approximately candidate 0 (weight scaling), and in
  frozen LLMs it was catastrophic (Soft Thinking's 6.66).
- **Uncapped trees.** 4⁶ leaves, and APR shows RL widens.

### 4.5 Where the extension fits: after the current plan

The current plan in STEP_SELECTION.md runs first and unchanged. That means Stage 0, then Stage 1's single-select RL
selector with its own go/stop rule, then Stage 2 if Stage 1 passes. The extension follows:
1. **E0: offline checks on Stage 0's records** (Section 5), with their gate fixed before they are run. They need no
   training.
2. **E1: the gate head of 4.1**, trained on top of Stage 1's anchor head with the anchor frozen, and then optionally
   jointly. If Stage 1 stopped, greedy is the spine. Single-select stays a clean, separate result.
3. **Comparisons within the extension.** Count compute in token-forwards, decoder reads included, and compare at matched
   cost: majority over about 19 chains for K = 4 with reads at every step. Also report the training-free hedge-and-vote
   baselines of O3.
4. **A cheaper variant to test within the extension.** Read the decoder at steps 1–3 only. Reads beyond slot 2 are
   near-useless (exact steps fall from 199 at slot 0 to 21 at slot 2, then to about 0), and the reads dominate the cost.

## 5. Offline checks on Stage 0 records

All the checks below reuse Stage 0's JSONL records (greedy steps and features, 6 × 3 branches with answers and features,
18 chain answers, the rule's chain), pool both sampling seeds, and use paired sign tests against greedy, as in
`report.py`. With a greedy spine, the branches are exactly the hedges of 4.1, so O2–O4 and O8 evaluate the recommended
design itself, without training. The thresholds are suggestions, to be fixed before the checks are run.

- **O1. Diversity and the minority-correct rate.**
  - For each (question, step), take the 4-set: greedy plus 3 branches.
  - Report the number of distinct answers, and, in mixed sets, the share where the right answer is held by ≥ 2 of the
    4 against exactly 1.
  - Repeat over all 19 answers: greedy plus 18 branches.
  - This decides whether voting can rescue anything. AggLM's gains came from minority-correct cases; a vote cannot win
    those.
- **O2. Vote-reachable coverage: the oracle for the best subset to keep.**
  - With the spine always kept and winning ties, a question is vote-reachable if greedy is right, or if at least 2
    branches share the right answer, since one hedge can only tie and lose.
  - Compare it with pick-reachable coverage, greedy or any branch right, which is Stage 0's headroom.
  - Report the smallest subset that makes the vote right, and its size distribution.
  - **Suggested gate:** if vote-reachable coverage is under +20/500 over greedy, drop (b) and stay with single-select.
- **O3. Fixed hedging policies, training-free.** Spine = greedy, and the vote is V1. Hedges:
  - (i) every branch at step s only, for each s from 1 to 6;
  - (ii) every branch at steps 1–3;
  - (iii) all 18;
  - (iv) disagreement-triggered: branches whose decoded result differs from greedy's at that slot;
  - (v) branches whose rule rank is at least candidate 0's.

  Report accuracy, the paired test against greedy and the token-forward cost of each.
- **O4. Hedge against commit with the same signal.**
  - At each step s, compare committing to the fixed rule's choice among the 4-set, with the other steps greedy, against
    voting over greedy plus every candidate whose rule rank ties the best.
  - This is the DIVERSE and Dalal question in our data: with our weak signal, is the vote safer than argmax?
- **O5. Branch votes against chain votes.**
  - Majority over greedy plus 18 branches against majority over greedy plus 18 chains, the same count.
  - At matched cost: 18 branches cost about 63 latent steps plus 126 readout tokens, about 189 token-forwards, or about
    14.5 chains.
  - Also report the pass@1 of a single dropout chain against greedy. If dropout samples are much weaker, votes over them
    start from behind (LatentTTS: about 31.3 vs 34.0).
- **O6. Level-balanced against flat votes.** Give each step's 3 branches a total weight of 1, against a flat count. This
  tests whether weighting by level helps "take every level into account".
- **O7. Decoded results as a redundancy and consistency signal.**
  - Within each 4-set, compare P(same final answer | same decoded result) with P(same final answer | different decoded
    result).
  - Check whether a branch's answer appears among the decoded results of greedy's slots or its own rerun's slot, and
    whether that ranks right above wrong *within* a question (CISC's within-question ranking test).
  - This decides two things: deduplicating hedges by decoded step (DVTS/ETS-style diversity) and the first vote-weight
    feature.
- **O8. Cross-validated accuracy against width for a learned gate.**
  - Fit a within-question logistic model of "branch answer is right", using step, valid, grounded, log-probability,
    parsed, disagreement with greedy and answer-in-decoded-results.
  - Fit on seed 1 and evaluate on seed 2, then swap. Sweep the keep threshold to trace accuracy against the mean
    number of hedges.
  - This is the best offline estimate of what 4.1's gate can learn from decoder features. It compares against O3 and
    against majority-of-N at matched cost.
- **O9. "Where" against "which" oracles.**
  - Oracle A picks *the step* at which to hedge every branch, and the vote decides.
  - Oracle B picks *the one right branch*, which is coverage.
  - The gap tells whether the selector's job is mostly where to widen (AB-MCTS's question) or which candidate to trust
    (single-select's).
- **O10. The cost table from real lengths.** Use the decoded texts and answer texts in the records to replace T_D and
  T_A with measured averages, and print every method's cost in token-forwards.

**Checks that need a short new eval run** (eval-only, minutes on a G4, not in Stage 0's records):
- **N1. Merge sanity.** At each step, compare the uniform pre-LN mean of the 4 candidates, and anchored subset merges,
  with candidate 0. Measure cosine, decoded step and answer agreement after continuing. The prediction from weight
  scaling is that the mean ≈ candidate 0.
- **N2. Per-layer KV cosine between candidates at the same step.** Fact 1 predicts near-copies.
- **N3. Early-exit answers at depth s < 6.** Needed only if votes from shallower levels are wanted.
- **N4. p = 0.1 against 0.2.** SLPO uses 0.1, and LatentTTS saw CODI's coverage collapse at higher diversity.

## 6. Risks and open questions

**Risks (inference).**
- **The selector signal is the binding constraint.**
  - ITS-Continuous found correct and incorrect Coconut thoughts geometrically inseparable, while LTO's Huginn
    classifier reached AUC 0.88.
  - Multi-select lowers the dependence on the signal, because a vote hedges, but it cannot create signal.
  - Stage 0's within-set AUCs, and O4 and O7, decide this.
- **Coverage-to-accuracy conversion has stayed at about 1–2 points on every GPT-2 latent model so far** (Section 2.4).
  A design that clears +10/500 would be a first, so preregister and expect a null.
- **Same-model diversity is weak.** Uesato's shared mistakes and Mixture-of-Agents' weaker same-model ensembles apply.
  MC dropout at p = 0.2 may also hurt CODI more than SIM-CoT (LatentTTS Figure 3).
- **Per-step argmax can do worse than greedy.** Both Uesato's and Dalal's results apply to the current single-select
  plan too. The rule's "keep candidate 0 on ties" softens this. When the extension runs, its hedge-and-vote baselines
  give the comparison with Stage 1's argmax.
- **Train and deploy differ.** Training runs counterfactual hedges for every extra and samples the policy; evaluation is
  deterministic and runs only the kept hedges. The gradient is exact for the stochastic policy, but the deterministic
  policy's accuracy must be measured, not assumed.
- **The merge options (Section 4.3).** Off-manifold blur, collapse to one member (Continuous Mixture Collapse), and
  near-copy dilution if M2 is used without the log w bias.
- **Transfer to 1B.** Same-position siblings are legal under RoPE too, but at 1B the answer's position cost dominates the
  depth loss. Hedges add no positions, while M2 adds entries at an existing position, so neither should move the answer
  position.

**Open questions.**
- Does SIM-CoT give usable answers from early exits at depth < 6? That decides whether "every level" can vote without
  finishing chains (N3).
- Would a learned reader (V3) beat the summed vote at GPT-2 scale? SSA's gain on GSM8K was only +1.6.
- Should the anchor and gate heads train jointly or in sequence? Sequential is cleaner to report; joint could let the
  anchor take risks that hedges cover.
- Is adaptive K, drawing more reruns where candidates disagree (AB-MCTS), worth more than adaptive keep? This is
  testable later from per-step disagreement rates.
- Novelty, as far as these searches found:
  - per-step variable-width keep in latent space;
  - exact counterfactual gate credit through the vote;
  - a mass-preserving superposed latent step.

  Prior work comes close from three sides: LatentTTS (a fixed beam on Coconut), MIRAGE (a gate once per question) and
  Multiplex Thinking (merges with a trained generator).

## 7. Sources

Read levels are in the Section 2 tables. Sources cited only in this report's text are marked here.

**Selection policies, credit and adaptive search**
- [Lei, Barzilay, Jaakkola, Rationalizing Neural Predictions, arXiv 1606.04155](https://arxiv.org/abs/1606.04155)
- [BlockDrop, arXiv 1711.08393](https://arxiv.org/abs/1711.08393)
- [PonderNet, arXiv 2107.05407](https://arxiv.org/abs/2107.05407)
- [Kool et al., Stochastic Beams and Where to Find Them, arXiv 1903.06059](https://arxiv.org/abs/1903.06059)
- [Kool et al., Estimating Gradients for Discrete Random Variables by Sampling without Replacement, arXiv 2002.06043](https://arxiv.org/abs/2002.06043)
- [Xie & Ermon, Reparameterizable Subset Sampling, arXiv 1901.10517](https://arxiv.org/abs/1901.10517)
- [Gumbel MuZero, OpenReview](https://openreview.net/forum?id=bERaNdoegnO)
- [SlateQ, arXiv 1905.12767](https://arxiv.org/abs/1905.12767)
- [MIRAGE, arXiv 2609.21554](https://arxiv.org/abs/2609.21554)
- [TreeRL, arXiv 2506.11902](https://arxiv.org/abs/2506.11902)
- [ARPO, arXiv 2507.19849](https://arxiv.org/abs/2507.19849)
- [TreePO, arXiv 2508.17445](https://arxiv.org/abs/2508.17445)
- [TreeRPO, arXiv 2506.05183](https://arxiv.org/abs/2506.05183)
- [Damani et al., Learning How Hard to Think, arXiv 2410.04707](https://arxiv.org/abs/2410.04707)
- [When to Stop (latent stop head), arXiv 2511.21581](https://arxiv.org/abs/2511.21581)
- [Adaptive Parallel Reasoning (APR), arXiv 2504.15466](https://arxiv.org/abs/2504.15466)
- [Wider or Deeper? AB-MCTS, arXiv 2503.04412](https://arxiv.org/abs/2503.04412)
- [Coulom, Computing Elo Ratings of Move Patterns in the Game of Go (progressive widening)](https://www.remi-coulom.fr/Amsterdam2007/)
- [Inference Scaling Laws (REBASE), arXiv 2408.00724](https://arxiv.org/abs/2408.00724)
- [ETS, arXiv 2502.13575](https://arxiv.org/abs/2502.13575)
- [Snell et al., Scaling LLM Test-Time Compute Optimally, arXiv 2408.03314](https://arxiv.org/abs/2408.03314)
- [Hugging Face, Scaling test-time compute (DVTS)](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute)
- [Dalal et al., More Test-Time Compute Can Hurt, arXiv 2603.15377](https://arxiv.org/abs/2603.15377)
- [VinePPO, arXiv 2410.01679](https://arxiv.org/abs/2410.01679)
- [Setlur et al., Rewarding Progress (PAV), arXiv 2410.08146](https://arxiv.org/abs/2410.08146)
- [Titsias, Local Expectation Gradients for Doubly Stochastic Variational Inference, arXiv 1503.01494](https://arxiv.org/abs/1503.01494) (cited in the text only; abstract)
- [Foerster et al., Counterfactual Multi-Agent Policy Gradients (COMA), arXiv 1705.08926](https://arxiv.org/abs/1705.08926) (cited in the text only; abstract)

**Merging continuous states and KV entries**
- [Zhu et al., Reasoning by Superposition, arXiv 2505.12514](https://arxiv.org/abs/2505.12514)
- [Zhu et al., Emergence of Superposition, arXiv 2509.23365](https://arxiv.org/abs/2509.23365)
- [Coconut, arXiv 2412.06769](https://arxiv.org/abs/2412.06769)
- [CoT2, arXiv 2505.23648](https://arxiv.org/abs/2505.23648)
- [Soft Thinking, arXiv 2505.15778](https://arxiv.org/abs/2505.15778)
- [LLMs are Single-threaded Reasoners, arXiv 2508.03440](https://arxiv.org/abs/2508.03440)
- [Soft Tokens, Hard Truths, arXiv 2509.19170](https://arxiv.org/abs/2509.19170)
- [Mixture of Inputs (Text Generation Beyond Discrete Token Sampling), arXiv 2505.14827](https://arxiv.org/abs/2505.14827)
- [Multiplex Thinking, arXiv 2601.08808](https://arxiv.org/abs/2601.08808)
- [SofT-GRPO, arXiv 2511.06411](https://arxiv.org/abs/2511.06411) (abstract; not in the tables)
- [HRPO, arXiv 2505.18454](https://arxiv.org/abs/2505.18454)
- [Superposed Decoding, arXiv 2405.18400](https://arxiv.org/abs/2405.18400)
- [The Dynamics of Continuous Mixture Collapse, arXiv 2609.02049](https://arxiv.org/abs/2609.02049)
- [Model soups, arXiv 2203.05482](https://arxiv.org/abs/2203.05482)
- [REPAIR, arXiv 2211.08403](https://arxiv.org/abs/2211.08403)
- [Srivastava et al., Dropout, JMLR 2014](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [Gal & Ghahramani, Dropout as a Bayesian Approximation, arXiv 1506.02142](https://arxiv.org/abs/1506.02142) (background; abstract)
- [Token Merging, arXiv 2210.09461](https://arxiv.org/abs/2210.09461)
- [CaM: Cache Merging, ICML 2024](https://proceedings.mlr.press/v235/zhang24n.html)
- [KVMerger, arXiv 2407.08454](https://arxiv.org/abs/2407.08454)
- [D2O, arXiv 2406.13035](https://arxiv.org/abs/2406.13035)
- [MiniCache, arXiv 2405.14366](https://arxiv.org/abs/2405.14366)
- [KeepKV, arXiv 2504.09936](https://arxiv.org/abs/2504.09936)
- [Deep Sets, arXiv 1703.06114](https://arxiv.org/abs/1703.06114)
- [Set Transformer, arXiv 1810.00825](https://arxiv.org/abs/1810.00825)
- [Graph of Thoughts, arXiv 2308.09687](https://arxiv.org/abs/2308.09687)
- [SpecInfer (tree-based attention for branched candidates), arXiv 2305.09781](https://arxiv.org/abs/2305.09781) (background on tree masks; abstract)

**Aggregation and parallel reasoning**
- [Lightman et al., Let's Verify Step by Step, arXiv 2305.20050](https://arxiv.org/abs/2305.20050)
- [Uesato et al., Solving Math Word Problems with Process- and Outcome-Based Feedback, arXiv 2211.14275](https://arxiv.org/abs/2211.14275)
- [Math-Shepherd, arXiv 2312.08935](https://arxiv.org/abs/2312.08935)
- [DIVERSE (Making Language Models Better Reasoners with Step-Aware Verifier), ACL 2023](https://aclanthology.org/2023.acl-long.291/)
- [The Lessons of Developing Process Reward Models, arXiv 2501.07301](https://arxiv.org/abs/2501.07301)
- [Liu et al., Can 1B LLM Surpass 405B LLM?, arXiv 2502.06703](https://arxiv.org/abs/2502.06703)
- [DeepConf, arXiv 2508.15260](https://arxiv.org/abs/2508.15260)
- [Self-certainty, arXiv 2502.18581](https://arxiv.org/abs/2502.18581)
- [CISC, arXiv 2502.06233](https://arxiv.org/abs/2502.06233)
- [Universal Self-Consistency, arXiv 2311.17311](https://arxiv.org/abs/2311.17311)
- [AggLM (The Majority is not always right), arXiv 2509.06870](https://arxiv.org/abs/2509.06870)
- [Sample Set Aggregator, arXiv 2506.09014](https://arxiv.org/abs/2506.09014)
- [GenSelect, arXiv 2507.17797](https://arxiv.org/abs/2507.17797)
- [Mixture-of-Agents, arXiv 2406.04692](https://arxiv.org/abs/2406.04692)
- [Parallel-R1, arXiv 2509.07980](https://arxiv.org/abs/2509.07980)
- [ParaThinker, arXiv 2509.04475](https://arxiv.org/abs/2509.04475)
- [Multiverse, arXiv 2506.09991](https://arxiv.org/abs/2506.09991)
- [Hogwild! Inference, arXiv 2504.06261](https://arxiv.org/abs/2504.06261)
- [Group Think, arXiv 2505.11107](https://arxiv.org/abs/2505.11107)
- [Native Parallel Reasoner, arXiv 2512.07461](https://arxiv.org/abs/2512.07461)
- [Forest-of-Thought, arXiv 2412.09078](https://arxiv.org/abs/2412.09078)
- [Beyond the Last Answer, arXiv 2504.20708](https://arxiv.org/abs/2504.20708)

**Latent reasoning with several candidates**
- [Parallel Test-Time Scaling for Latent Reasoning Models (LatentTTS), ACL 2026](https://aclanthology.org/2026.acl-long.2069.pdf) and [arXiv 2510.07745](https://arxiv.org/abs/2510.07745)
- [Towards Inference-time Scaling for Continuous Space Reasoning, arXiv 2510.12167](https://arxiv.org/abs/2510.12167)
- [Cui et al., How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision?, arXiv 2602.22441](https://arxiv.org/abs/2602.22441)
- [NoisyCoconut, arXiv 2605.08221](https://arxiv.org/abs/2605.08221)
- [SoftCoT++, arXiv 2505.11484](https://arxiv.org/abs/2505.11484)
- [LaDi-RL, arXiv 2602.01705](https://arxiv.org/abs/2602.01705)
- [LTC, arXiv 2608.01593](https://arxiv.org/abs/2608.01593)
- [LTO (latent trajectory optimization for Huginn), arXiv 2509.26314](https://arxiv.org/abs/2509.26314)
- [GTS, arXiv 2602.14077](https://arxiv.org/abs/2602.14077)
- [SLPO, arXiv 2607.19691](https://arxiv.org/abs/2607.19691)
- [PLaT, arXiv 2601.21358](https://arxiv.org/abs/2601.21358)
- [Dropout-GRPO, arXiv 2606.10184](https://arxiv.org/abs/2606.10184)
- [C-voting, arXiv 2604.13521](https://arxiv.org/abs/2604.13521)
- [SIM-CoT, arXiv 2509.20317](https://arxiv.org/abs/2509.20317)
- [CoLaR, arXiv 2505.16552](https://arxiv.org/abs/2505.16552)
