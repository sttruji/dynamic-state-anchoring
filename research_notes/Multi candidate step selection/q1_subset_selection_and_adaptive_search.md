# Q1: variable-size subset selection with RL, adaptive width, wider-vs-deeper search

Working notes, 2026-11-06. Web research only. Read levels: [full text] = method and results sections read (figures sometimes unreadable); [abstract only]; [unverified] = could not open a primary source.
Everything under "Inference" is my reasoning, not a paper claim.

Setting reminder: SIM-CoT GPT-2, 6 latent steps, greedy answer decode, 204/500 on GSM8K validation (40.8%). Per step K=4 candidates (cand 0 = dropout-off; 1-3 = MC dropout). Frozen generator, small selector, GRPO, 0/1 final reward, step-decoder features. New question: keep a variable-size subset per step, then merge or run as parallel chains, with a width/compute penalty.

---

## 1. Subset-action parameterizations (how log p(S) is defined)

### Lei, Barzilay, Jaakkola 2016, "Rationalizing Neural Predictions" (EMNLP 2016, arXiv 1606.04155) [full text]
- Generator emits a binary mask z over input tokens. Independent version: p(z|x) = prod_t p(z_t|x) with sigmoid per token from a bi-RNN. Dependent version: p(z_t | x, z_<t) via an extra recurrent state.
- Cost = ||enc(z,x) - y||^2 + Omega(z), Omega(z) = lambda1 ||z|| + lambda2 sum_t |z_t - z_{t-1}| (size penalty + contiguity). lambda1 in {2e-4, 3e-4, 4e-4}, lambda2 = 2 lambda1.
- Gradient: plain REINFORCE, E_z[cost * d log p(z|x)/d theta], a few sampled masks per example. Subset size is variable and controlled only by lambda1.
- Numbers (BeerAdvocate, Table 2): precision of selected words appearance 96.3% at 14% of text selected (recurrent generator), smell 95.1% at 7%, palate 80.2% at 7%; attention baseline 80.6 / 88.4 / 65.3; SVM 38.3 / 21.6 / 24.9. Encoder MSE 0.0087 with full text (Table 3); the MSE vs % selected curve stays near full-text MSE once ~20-30% of text is kept (Fig. 2).
- log p(S) = sum_i [m_i log p_i + (1 - m_i) log(1 - p_i)]. Empty set allowed.

### BlockDrop, Wu et al. (CVPR 2018, arXiv 1711.08393) [full text]
- Policy net outputs s in [0,1]^K; action u ~ Bernoulli(s), K-dim; log pi(u|x) = sum_k log[s_k u_k + (1 - s_k)(1 - u_k)]. Single-step contextual bandit (all decisions at once).
- Reward: R(u) = 1 - (|u|_0 / K)^2 if correct, else -gamma. gamma sets the operating point. Compute cost enters only on correct outputs.
- Self-critical baseline: A = R(u) - R(u_greedy), u_greedy = threshold s at 0.5. Exploration: s' = alpha s + (1 - alpha)(1 - s), alpha = 0.8.
- Curriculum: at epoch t, keep first K - t blocks on, learn only the last t decisions. Without curriculum, "starting from a randomly initialized policy and optimizing for both accuracy and block usage is not effective".
- Numbers (Table 1): frozen ResNet-110 on CIFAR-10, policy only: 75.4% with 20.1/54 blocks (full net 93.2%); after joint finetuning 93.6% with 16.9 blocks. ResNet-32 CIFAR-10: 88.6% (9.4/15 blocks) frozen, 91.3% finetuned (full 92.3%). ImageNet ResNet-101: 20% average speedup (up to 36%), same 76.4% top-1 (abstract).
- Note: with a frozen backbone the learned drop policy lost accuracy; finetuning the backbone recovered it. Our generator is frozen, so only the "policy-only" numbers are a fair analog, and those are the weaker ones. Different task (dropping compute, not picking among candidates).

### SkipNet, Wang et al. (ECCV 2018, arXiv 1711.09485) [abstract only]
- Per-block gates, hybrid supervised (soft gates) + RL (hard gates). Abstract: 30-90% less computation, accuracy preserved, four datasets. I did not read the reward form.

### ACT, Graves 2016 (arXiv 1603.08983) [abstract only; formula from PonderNet's discussion]
- Halting units, ponder cost weighted by tau. PonderNet (below) reports ACT is very sensitive to tau.

### PonderNet, Banino, Balaguer, Blundell 2021 (ICML 2021 AutoML workshop, arXiv 2107.05407) [full text, pp. 1-5]
- p_n = lambda_n prod_{j<n} (1 - lambda_j). Loss L = sum_n p_n L(y, y_hat_n) + beta KL(p_n || p_G(lambda_p)), geometric prior with mean 1/lambda_p. No explicit step-count penalty; the KL prior acts as the compute regularizer and keeps exploration.
- Parity: PonderNet beats ACT in accuracy with fewer steps (Fig. 1). ACT solves the task only for a few tau values and fails for tau >= ~3e-3 (Fig. 2a). PonderNet works across lambda_p in 0.1-0.8 and fails only at 0.9 (prior ~1 step). With lambda_p = 0.1 (prior 10 steps), it settles at ~3 steps.
- Relevance: a per-unit-compute penalty (ACT tau, our lambda per kept candidate) is fragile. A prior/KL on the width distribution or a target constraint is more robust.

### Kool, van Hoof, Welling 2019, "Stochastic Beams and Where to Find Them" (ICML 2019, arXiv 1903.06059) [full text; results only as figures]
- Gumbel-top-k: perturb log-probs with Gumbel(0) noise, take the top k = exact sample of k items without replacement (Plackett-Luce order). Stochastic Beam Search applies this over sequences with cost linear in k and length.
- Inclusion probability with threshold kappa (priority sampling): q(s, kappa) = 1 - exp(-exp(phi_s - kappa)). Gives an importance-weighted estimator.
- Results are qualitative/figures: SBS matches Diverse Beam Search BLEU with higher diversity; sample-without-replacement estimators of BLEU and entropy have low variance.

### Kool, van Hoof, Welling 2020, "Estimating Gradients for Discrete Random Variables by Sampling without Replacement" (ICLR 2020, arXiv 2002.06043) [full text]
- Ordered sample B^k (Plackett-Luce): p(B^k) = prod_i p(b_i) / (1 - sum_{j<i} p(b_j)).
- Unordered set: p(S^k) = sum over all k! orderings of p(B^k) (eq. 6). Efficient computation in App. B. For k <= 4 that is at most 24 terms, so it can be computed exactly by brute force.
- Unordered set estimator: e^US(S) = sum_{s in S} p(s) R(S,s) f(s), with leave-one-out ratio R(S,s) = p^{D\{s}}(S\{s}) / p(S).
- Policy gradient with built-in baseline (Thm 2): sum_s grad p(s) R(S,s) (f(s) - sum_{s'} p(s') R^{D\{s}}(S,s') f(s')). Unbiased.
- Also introduces "REINFORCE with replacement + built-in baseline" (leave-one-out mean of other samples, eq. 23). This is the RLOO estimator, close to GRPO's group baseline.
- Numbers: categorical VAE gradient log-variance with k=4 (Table 1): small domain (10^2) unordered 6.29 = best (sum&sample 6.29, REINFORCE no-bl 11.52, RELAX 11.67, ARSM 13.45); large domain (10^20) unordered 7.05, REINFORCE-w.r. built-in 7.06, REINFORCE no-bl 13.81, RELAX 15.86, ARSM 15.55. TSP20: matches the biased risk estimator, beats REINFORCE w.r. and greedy-rollout baseline (figures only).
- Paper's own takeaway: REINFORCE with replacement + built-in baseline "is a simple yet strong estimator" with performance similar to theirs in the high-entropy setting.
- Subset size k is fixed. A variable size needs a separate size distribution (my note).

### Xie & Ermon 2019, "Reparameterizable Subset Sampling via Continuous Relaxations" (IJCAI 2019, arXiv 1901.10517) [full text]
- p(S|w) = sum over orderings of weighted-reservoir (Plackett-Luce) probabilities. Gumbel keys r_i = log w_i + g_i; relaxed top-k by k successive softmaxes, alpha^{j+1} = alpha^j + log(1 - softmax(alpha^j / t)). Relaxed k-hot output; converges to exact samples as t -> 0.
- Fixed k. Numbers: L2X IMDB post-hoc accuracy 91.7 vs 90.7 (word, k=10), 83.2 vs 82.9 (sentence, k=1). Stochastic kNN CIFAR-10 90.1 vs NeuralSort 89.5.
- Relevance: low. Our candidates are discrete states fed to a frozen model. A relaxed k-hot "merge" is possible in principle, but the reward is a 0/1 decode, not differentiable.

### Gumbel MuZero / Gumbel AlphaZero, Danihelka, Guez, Schrittwieser, Silver (ICLR 2022, OpenReview bERaNdoegnO) [full text via a third-party text mirror; OpenReview blocked]
- Root: sample m actions without replacement by Gumbel-top-k on g + logits. Then Sequential Halving: split n simulations across log2(m) phases, visit the remaining actions equally, drop half each phase by g + logits + sigma(q). Final action = argmax of the same score. Using the same g in top-k and argmax gives a policy-improvement guarantee E[q(A)] >= E_pi[q].
- sigma(q) = (c_visit + max_b N(b)) c_scale q; c_visit = 50, c_scale = 1.0 (Go/chess), 0.1 (Atari). m = min(n, 16) for board games, min(n, 18) for Atari.
- Policy target pi' = softmax(logits + sigma(completedQ)), with unvisited actions completed by v_mix. Trained with KL(pi', pi) over all actions, so every candidate gets a learning signal, not just the chosen one.
- Numbers (text): 9x9 Go, MuZero fails to learn with <= 16 simulations, Gumbel MuZero learns with 2. Atari ms_pacman, MuZero fails with <= 4, Gumbel learns with n = 2. Table 1 training speedup on 9x9 vs MuZero n=200: n=32 5.9x, n=16 11.3x, n=8 16.2x, n=4 24.3x. Elo numbers only in figures.
- Fixed m. A subset is chosen only to decide where to spend simulations; the final output is one action.

### SlateQ, Ie et al. (IJCAI 2019; long version arXiv 1905.12767) [full text, ~78% of the arXiv version]
- Q(s, A) = sum_{i in A} P(i | s, A) Qbar(s, i) under single-choice + reward/transition-depends-on-selected-item assumptions. Conditional-logit choice with a null item.
- Slate optimization: exact via a fractional MIP -> LP (Charnes-Cooper), polynomial time. Heuristics: top-k by v(s,i) Qbar(s,i), greedy. The top-k approximation ratio is unbounded.
- Formal model uses fixed slate size k (k=3 of m=10 in simulation). The live YouTube system used variable slate sizes with ~1000 candidates.
- Simulation: QL-OT-OS 174.6 vs Random 159.2 (+9.67%), vs myopic 166.3. In a second experiment, full-slate Q-learning got 164.2 vs SARSA-TS 170.7 (Random 160.6). The live results were in the truncated part.
- Relevance: the decomposition idea (a subset's value from item-level values) helps generalization when the subset space is large. With K=4 we can also enumerate.

### CUCB, Chen, Wang, Yuan (ICML 2013) [abstract only]
- Super arms built from base arms. An (alpha, beta)-approximation oracle picks the super arm from estimated means. O(log n) regret. Value-based (UCB) rather than policy-gradient. Relevant only as a bandit view of "which subset to keep" with semi-bandit feedback.

### DPP (background, not re-read)
- L-ensemble DPP: P(S) = det(L_S) / det(L + I). Variable size natively, favors diverse subsets, log-prob exact. The k-DPP fixes size. The only RL use I found is DNS (ICML 2022, arXiv 2201.13357) [abstract only], with a fixed-size k-DPP over ensemble critics. I found no variable-size DPP policy-gradient paper.

### MIRAGE, Lagzian, Anumasa, Liu (arXiv 2609.21554, Sept 2026; page says ICML 2025 MAS workshop) [full text]
- The closest analog to our plan: a frozen reasoner (Qwen2.5-14B-Instruct) and a trainable selector (Qwen2.5-7B + LoRA) that scores 20 "perspectives". Training samples m_i ~ Bernoulli(p_i), "ensure at least one perspective is selected". REINFORCE loss -(1/B) sum r log Pr(m|q), with no baseline stated.
- Reward r = 1[majority-vote answer correct] - lambda k / |S|, lambda = 0.05, |S| = 20, k = number selected. Average k drifts down to ~8/20 during training.
- Inference: try perspectives in rank order, stop when confidence >= tau, else aggregate.
- Numbers (single runs, no error bars): Qwen2.5-7B GSM8K CoT 79.80% -> MIRAGE 89.01% (1.97 reasoner calls). MMLU-Pro (100 q x 4 subjects) DeepSeek-v3 57.25 -> 71.00 (4.07 calls). It loses to DIPPER n=5 in several cells (e.g., DeepSeek-v3 GSM8K 89.99 vs 93.48). Workshop quality, and the claims are inconsistent with the tables.
- Note the "at least one" fix. If it is implemented by forcing a selection outside the sampled mask, log Pr(m) no longer matches the behavior distribution (my note).

---

## 2. Learned width / adaptive branching

### TreeRL / EPTree, Hou et al. (arXiv 2506.11902; ACL 2025 per my memory, not checked) [full text]
- Fork at the top-N highest-entropy tokens per tree (excluding the tail), T continuations per fork, L iterations; leaves = M (T N L + 1). Default (M,N,L,T) = (6,2,1,2) -> 30 responses vs 16 for chain sampling at a similar token cost.
- Advantages: V(s) = fraction of correct leaves under s. G_A = V(s) - V(root), L_A = V(s) - V(parent). R(s) = (G_A + L_A) / sqrt(|L(s)|).
- Numbers: Qwen-2.5-14B six-benchmark avg SFT 36.0, ChainRL 41.6, TreeRL 44.5. AIME24 22.2 -> 28.0. Omni-MATH-500 PassRate at matched tokens: entropy-forking 56.9% (22.3k tokens) vs random forking 54.8% vs 16-chain 52.4%; (8,4,2,2) 71.0% vs 64-chain 67.4%. Ablation: dropping L_A -1.9 avg; 16 vs 30 responses -3.2.
- Branch count is fixed by hyperparameters. Only the location is adaptive (entropy).

### ARPO, Dong et al. (arXiv 2507.19849) [full text, first ~58%]
- After each tool call, compute the normalized entropy change dH. Branch probability P_t = alpha + beta dH. If P_t > tau, fork Z partial paths, until the global budget M - N is spent. Global budget M (e.g., 16) with N initial samples (best 8).
- Advantage: shared prefix tokens get the average advantage of the branches that share them ("soft" via GRPO ratios is the default).
- Numbers: 10-task avg Qwen2.5-7B GRPO 56.5 -> ARPO 58.3; Llama3.1-8B 51.1 -> 55.3; Qwen2.5-3B 50.4 -> 52.8. Deep search Qwen3-14B 27.0 -> 32.0. Text says about half the tool calls of GRPO (figure only).

### TreePO, Li et al. (arXiv 2508.17445) [full text]
- Segment-wise tree sampling. Budget 2^d at depth d, spread over active paths ("budget transfer"). Branching by softmax(segment log-prob / 2.0) with at least 1 branch per active path. Early stop on repetition. Fallback restarts. Max width 16.
- Advantage: average over depths of (reward - subgroup mean) for nested ancestor subgroups, normalized by global std. Simple averaging beats size weighting. Dropping all-same subgroups hurts.
- Numbers (Qwen2.5-7B, maj@16 avg of 5 benchmarks): GRPO 46.63 -> TreePO sampling 54.61 -> full TreePO 58.21. GPU hours -12% / -43% depending on config. The abstract says 22-43%.
- Static "encourage low/high prob" branching rules both underperformed.

### TreeRPO, Yang et al. (arXiv 2506.05183) [full text]
- N-ary tree (N=8, depth D=3, 384 tokens/step). Node reward = mean of children. Advantage within sibling groups A = (R - mu)/sigma with sigma = mu(1 - mu) (sic). Keep groups only if max - min > 0.1.
- Qwen2.5-Math-1.5B avg Pass@1 GRPO 32.7 -> 35.6. Response length -18.1%.

### Damani et al., "Learning How Hard to Think: Input-Adaptive Allocation of LM Computation" (arXiv 2410.04707; ICLR 2025 per my memory, not checked) [full text]
- Predict the marginal reward of the j-th extra sample per query. Binary case: q(x,b) = 1 - (1 - lambda)^b, predict lambda by cross-entropy, via a 2-layer MLP on the LM's last hidden state or LoRA.
- Allocation: integer program max sum c_ij Delta_ij s.t. sum c_ij <= B n. Greedy is optimal (matroid). Online (batch) or offline (bins by predicted difficulty).
- Numbers: Mathstral-7B on Numina 25-50% less compute at B >= 8 for the same success. Chat 25-40% on a "tranches" subset (0-10% on the full set). Routing calls the strong decoder only 50-75% of the time. Predictor discriminates above/below-median difficulty at 74% (code), 84% (math). Headline: up to 50% compute saving, or up to 10% quality gain at fixed budget.
- Relevance: a cheap probe on hidden states predicts how much extra width helps, and the width budget is solved as a knapsack. This is a supervised alternative to RL for "how many to keep".

### Learning When to Stop: Adaptive Latent Reasoning via RL, Ning, Kuo, Gomes (arXiv 2511.21581) [full text, partial]
- CODI-based latent reasoner on Llama 3.2 1B. A linear stop head decides continue/stop per latent step. GRPO with 0/1 correctness, a -1 format penalty, and a group-relative length term (lambda_penalty 1e-4, lambda_reward 0.1, p_cutoff 1).
- GSM8K-Aug test: Latent-6 49.73% (8 tokens) -> +RL 50.11% (3.76 tokens), i.e., -52.9% tokens with no accuracy loss. The Latent-6-by-1 variant lost accuracy (51.40 -> 48.14).
- The closest latent-CoT precedent for an RL-trained compute knob on a CODI model (depth, not width).

### Learning Adaptive Parallel Reasoning (APR), Pan et al. (arXiv 2504.15466; COLM 2025 per my memory, not checked) [full text, partial]
- spawn()/join(). SFT on solver traces conditioned on 0-10 children, then GRPO on correctness only. No compute penalty. RL raised the average child count from 6.1 to 8.2.
- Countdown at a 4k context: APR 83.4% vs serialized SoS+ 60.0%. At ~5,000 ms latency: 75.2% vs 57.3%.
- Without a width penalty, RL learned to go wider. The authors attribute most of the RL gain to scaling test-time compute rather than better decisions at a fixed budget.

### Learning Adaptive LLM Decoding, Su et al. (arXiv 2603.09065) [full text]
- Frozen LLM plus a small RL-trained adapter that picks a temperature per token (greedy, 0.5, 1.0, 1.25) from the hidden state and remaining budget. REINFORCE with an entropy bonus and baseline. Masks tokens with max-prob > 0.95. Filters sparse/noisy-reward prompts.
- Qwen3-4B MATH Pass@1: greedy 71.33 -> adapter with budget 80.82 (no CoT). The abstract says "up to 10.2%".
- A precedent for a frozen generator + small RL controller at step granularity. Its action is sampling temperature, not candidate selection.

### Adaptive-Consistency (Aggarwal et al., EMNLP 2023, arXiv 2305.11860) [abstract only]
- Stop sampling when answer agreement is stable. Up to 7.9x fewer samples, <0.1% average accuracy drop, 17 datasets.

### DeepConf (Fu et al., arXiv 2508.15260) [abstract only]
- Training-free online pruning of traces by model-internal confidence. AIME 2025 up to 99.9% (DeepConf@512), up to 84.7% fewer tokens.

### Adaptive beam search via RL for seq2seq (Lin & Lin 2018, course-project PDF) [unverified, not read]
- The only "learn beam size with RL" result the search surfaced. Not peer reviewed.

### Knapsack RL (Li et al., arXiv 2509.25849) [abstract only]
- Allocates GRPO rollouts across prompts as a knapsack. 20-40% more non-zero gradients, +2-4 avg math points, ~2x compute equivalence.

---

## 3. Wider or deeper tree search

### AB-MCTS, Inoue et al. (Sakana), "Wider or Deeper?" (arXiv 2503.04412) [full text, first ~73%]
- Every node has a GEN child. Choosing GEN = generate a new child (wider). Choosing an existing child = descend (deeper). Unbounded branching, and non-leaf nodes can be re-expanded.
- Selection by Thompson sampling over posterior predictive scores, including GEN. The authors note UCT does not apply since GEN creates new arms.
  - AB-MCTS-M: mixed model r = alpha_j + sigma_y eps, alpha_j = mu_alpha + sigma_alpha eps_j; priors mu_alpha ~ N(0.5, 0.2^2), sigma_alpha ~ HalfN(0.2^2), sigma_y ~ HalfN(0.3^2); MCMC.
  - AB-MCTS-A: CONT node aggregates children. Conjugate Normal-inverse-chi^2 or Beta (alpha + sum r, beta + sum(1 - r)) with non-informative priors (mean 0.5). Stable across prior ranges (App. A.6).
- Scores come from an external evaluator (public tests, ARC train pairs). Budget 128 calls.
- Table 1 (Pass@1; ARC Pass@2): LiveCodeBench GPT-4o RS 37.8, SeqRef 37.8, MCTS 36.7, AB-MCTS-M 38.9, -A(Gauss) 39.1. CodeContest GPT-4o RS 37.9, SeqRef 30.1, MCTS 37.5, M 40.6. ARC-AGI DS-V3 RS 18.6, SeqRef 10.0, MCTS 14.0, M 16.0, A(Gauss) 18.3. Average rank M 2.3 vs RS 3.5, SeqRef 5.5, MCTS 4.2.
- ARC-AGI DS-V3 up to 512 calls: AB-MCTS keeps improving from 200 to 500 while repeated sampling plateaus (figure).
- Gains over repeated sampling are small (1-3 points) and within ~1-2 std in several cells.

### Progressive widening, Coulom 2007 (ICGA Journal 30(4)) [full text, section 4.2 + Table 2]
- The n-th move (pattern-ranked) is added after t_{n-1} simulations, t_0 = 0, t_{n+1} = t_n + 40 x 1.4^n. Width grows like the log of the visit count.
- Crazy Stone vs GNU Go: 9x9 win rate 68.2% -> 90.6% with PW; 19x19 0.0% -> 37.5% (32 min/game), 57.1% at 128 min.

### Double progressive widening, Couëtoux et al. 2011 (LION 5, "Continuous Upper Confidence Trees") [unverified, HAL blocked; secondary sources]
- Children limited to k N^alpha (N = visits), applied to both actions and stochastic outcomes. They show a deceptive problem where UCT + PW fails.

### REBASE, Wu et al., "Inference Scaling Laws" (arXiv 2408.00724; ICLR 2025 per my memory, not checked) [full text]
- At depth i with remaining budget B_i, child width W_j = Round(B_i exp(R(n_j)/T_b) / sum_k exp(R(n_k)/T_b)), T_b = 0.1. PRM = Llemma-34B fine-tuned on Math-Shepherd. B_i decrements by finished solutions.
- MATH500 weighted voting: Mistral-7B sampling 256 samples 42.8% (8.70e14 FLOPs) vs REBASE 32 samples 45.0% (1.36e14). Llemma-7B 45.5 (256) vs 46.8 (32). Llemma-34B 46.7 (64) vs 49.2 (32).
- Fixed budget (Table 4), GSM8K weighted vote: Llemma-7B (32) 85.4 -> 87.3; Mistral-7B (32) 89.1 -> 89.8; Llemma-34B (8) 86.0 -> 86.9. MATH: Mistral-7B (32) 39.7 -> 45.0.
- MCTS was worse than sampling at every budget (no numbers given). REBASE's advantage is on MATH levels 3-5 and about equal on levels 1-2.
- Width is variable per node (0..B) through a softmax of scores with a total budget. The temperature controls how concentrated it is.

### ETS, Hooper et al. (arXiv 2502.13575) [full text]
- At each step an ILP picks which leaves to keep: maximize retained REBASE weight - lambda_b (fraction of tree nodes kept, i.e., KV cost) + lambda_d (fraction of semantic clusters covered). lambda_d = 1, lambda_b in [1, 2].
- Llemma-34B MATH500 width 256: REBASE 52.0 -> ETS 52.8, KV cache 1.8x smaller, throughput 1.4x. Llama-3.2-1B width 256: 53.8 -> 53.8, 1.6x KV reduction.
- A direct precedent for "keep a variable subset with an explicit KV-cache cost + diversity term". Solved as an ILP, not learned.

### DVTS, HuggingFace "Scaling test-time compute with open models" blog (Dec 2024) [full text of blog; numbers mostly in plots]
- Split N into N/M independent subtrees (M = 4). In each subtree, greedy PRM-guided expansion of M children, keeping the best. PRM RLHFlow/Llama3.1-8B-PRM-Deepseek-Data. "Last" step-score reduction.
- Llama 3.2 1B MATH-500: beam search ~55% max. Beam N=4 ~ best-of-N 16. Beam N=16 ~ best-of-N 256. DVTS better at large N and on easy/medium; beam better at small N and on hard. Compute-optimal 3B beats 70B Instruct (plots only).

### Snell et al., "Scaling LLM Test-Time Compute Optimally..." (arXiv 2408.03314; ICLR 2025 per my memory, not checked) [full text, ~90%]
- Beam search: keep the top N/M by PRM, expand each into M children (M = 4 for the difficulty study). Lookahead k. Difficulty = 5 quantile bins of pass@1 (oracle from 2048 samples, or predicted by verifier score).
- At small budgets beam beats best-of-N; at large budgets it often underperforms. Easy questions (levels 1-2): beam degrades with budget. Medium (3-4): beam consistently better. Level 5: nothing helps.
- Compute-optimal per-bin strategy beats best-of-N with up to 4x less compute (16 vs 64). PRM over-optimization explains it: short 1-2 step solutions and repetitive steps. Lookahead (strongest search) underperforms most.

### Dalal et al., "More Test-Time Compute Can Hurt: Overestimation Bias in LLM Beam Search" (arXiv 2603.15377) [full text]
- Max over n noisy scores gives a bias of about sigma sqrt(2 log(n - 1)). Search helps only if the quality gap Delta exceeds it. Max useful pool n_hat = 1 + exp(Delta^2 / 2 sigma^2), beam width k_hat = floor(sqrt(n_hat)).
- MR-BEN, k = 1..4: with self-perplexity scoring, Llama 59.7 -> 54.3 and Mistral 42.4 -> 38.5 (wider hurts). With the Math-Shepherd PRM, Llama 59.2 -> 65.4 and Mistral 42.3 -> 51.2. Diagnostic: 44.3% of selections had a top-2 margin < 0.1 when wider search hurt.

---

## 4. Credit assignment and compute cost in the objective

- Tree MC values: TreeRL (G_A + L_A, 1/sqrt(leaves)), TreeRPO (sibling-group normalized), TreePO (averaged nested-subgroup advantages), ARPO (shared-prefix averaged advantage). All with 0/1 outcome reward and no PRM.
- VinePPO (Kazemnejad et al., ICML 2025 per my memory, arXiv 2410.01679) [full text, partial]: per-step value = mean of K = 9 fresh rollouts from that prefix. Reaches PPO's peak in up to 3.0x less wall clock (RhoMath 1.1B), 1.51x (DeepSeekMath 7B), with 9x / 2.8x fewer gradient steps. Each iteration is up to 5x / 2x slower. The PPO value net was near chance at ranking actions. Exact accuracies only in figures.
- PAV, Setlur et al. (ICLR 2025 per my memory, arXiv 2410.08146) [full text, partial]: step reward = Q^pi + alpha A^mu, with A^mu the advantage under a "prover" policy (Bo4 a good default), alpha = 0.5 for 2B/9B, 0.2 for 27B. Search vs ORM: >8% more accurate, 1.5-5x compute-efficient (abstract; body says ~10x for 2B/9B, 5x for 27B). Online RL: 5-6x sample efficiency, >6% accuracy.
- Kool 2020: unordered-set PG with built-in baseline. RLOO-style leave-one-out also works well.
- Gumbel MuZero: the policy target covers all considered actions via completed Q, a lower-variance alternative to a single-action PG.
- Compute cost terms seen:
  - Additive per-unit: Lei lambda1 |z|; MIRAGE lambda k/|S| (lambda = 0.05); ACT tau x ponder (fragile).
  - Cost only when correct: BlockDrop 1 - (|u|/K)^2, else -gamma.
  - Prior/KL: PonderNet beta KL(p_n || Geometric(lambda_p)).
  - Group-relative length term: Learning-When-to-Stop (Kimi-k1.5 style).
  - Hard budget: REBASE/DVTS fixed N; Damani knapsack with sum c <= B n; ETS ILP with lambda_b KV cost; Knapsack RL.
  - Lagrangian: RCPO (Tessler et al., ICLR 2019, arXiv 1805.11074) [abstract only] uses a multi-timescale penalty signal. The Lagrange-multiplier details were not on the abstract page.
  - Value-of-computation reward: Rational Metareasoning (De Sabbata et al., arXiv 2410.05563) [abstract only], Expert Iteration, 20-37% fewer tokens at the same performance.

---

## 5. Inference: what this means for us (my reasoning, not paper claims)

1. The action space is tiny: 2^4 - 1 = 15 non-empty subsets per step. An exact categorical over 15 subsets (or a factorized Bernoulli renormalized by 1 - prod(1 - p_i)) gives exact log-probs. Plackett-Luce / Gumbel-top-k / unordered-set machinery is only needed if K grows. If using size-then-members, k! <= 24 orderings make the unordered PL probability exact by brute force.
2. A slot-allocation view unifies "how many" and "merge weights". Sample W slots (e.g., W = 4) from a categorical over the 4 candidates (multinomial, exact log-prob). The distinct candidates drawn are the kept set (size 1-4, variable), and the counts give merge weights or how many chains each gets. This is a stochastic, learned version of REBASE's softmax allocation W_j = round(B softmax(R/T)).
3. With parallel chains, width compounds over 6 steps (up to 4^6). A global cap is needed (REBASE/DVTS-style total width W <= 4, or ETS-style per-step KV cost). At the final step, an aggregation rule is needed (majority vote over decoded answers, or selector-weighted vote). Without a cap, the APR result suggests RL will widen (6.1 -> 8.2 children with no penalty).
4. Penalty form: avoid a raw per-candidate cost as the only control (ACT's tau failure mode; PonderNet Fig. 2a). Prefer (a) a target-width constraint with a Lagrange multiplier, or (b) a KL to a width prior concentrated on 1, or (c) BlockDrop-style cost charged only on correct answers. Sweep it and report the accuracy-vs-mean-width Pareto curve. MIRAGE's lambda = 0.05 per full set is a usable starting scale for a 0/1 reward.
5. Credit assignment is cheap because the generator is frozen and GPT-2 steps are cheap. For each step t and each candidate, roll out the default policy (cand 0) to the end and decode: 4 x 6 = 24 extra decodes per question gives per-candidate MC Q-values (VinePPO / TreeRL style). These can serve as (i) dense per-step advantages, (ii) a SlateQ-style decomposition for subset value, or (iii) a Gumbel-MuZero-style soft policy target, lower variance than GRPO on one 0/1 outcome.
6. Use the default trajectory as a baseline: A = R(selected) - R(cand-0-only). This is BlockDrop's self-critical baseline, and it directly measures "does selection beat the default". Gate the RL claim on that difference exceeding our seed noise (~9/500).
7. Width can hurt when the scorer is noisy (Dalal et al.; Snell on easy questions; HF on level-2). MC-dropout candidates probably differ little (small Delta), and the selector's features are noisy (large sigma). That favors (a) keeping width 1 = cand 0 as the default and widening only on high step-decoder disagreement (TreeRL/ARPO entropy-triggered branching as the training-free baseline the RL policy must beat), and (b) aggregation over several chains rather than argmax selection when uncertain.
8. Diversity of the kept set matters (ETS clusters, DVTS subtrees). Near-duplicate MC-dropout candidates waste width. Dedup by decoded step (the step-decoder already gives the arithmetic step) before the policy sees them, or use a DPP-style log det term.
9. Curriculum: learn the decision at step 6 first with steps 1-5 default, then extend backward (BlockDrop curriculum). BlockDrop's frozen-backbone numbers are a warning that a policy on a frozen generator can lose accuracy; its gain came after finetuning.
10. AB-MCTS's GEN node maps to "draw another dropout candidate" (adaptive K). Thompson sampling with Beta priors fits a 0/1 reward at test time. This is a separate axis from the subset choice and worth deferring.

## 6. Could not verify / gaps
- Couëtoux et al. 2011 DPW exact formula (HAL access denied; secondary sources say k N^alpha).
- DVTS exact accuracy numbers (blog gives plots only).
- VinePPO and Gumbel MuZero headline accuracies/Elo (figures only).
- SkipNet, ACT, RCPO, Rational Metareasoning, DeepConf, Adaptive-Consistency, CUCB, Knapsack RL: abstract only.
- The "learn beam size with RL" seq2seq paper (Lin & Lin 2018) is a course-project PDF. Not read.
- No paper found that learns a variable-size per-step keep-set with RL on top of a frozen latent reasoner. MIRAGE (frozen LLM, Bernoulli perspective mask, REINFORCE, size penalty) is the closest, but it acts once per question, not per step.
- Venue tags "per my memory" (Damani ICLR 2025, APR COLM 2025, VinePPO ICML 2025, PAV ICLR 2025) were not checked against official pages.
