# Q3 working notes: final aggregation over a tree/beam of kept latent candidates

Date: 2026-11-06. Web research only. Read levels: [full text] = arXiv HTML or the PDF text was read for the parts cited; [abstract only] = only the abs page; [unverified] = could not confirm.
Everything under "Inference" headings is my own reasoning, not a paper's claim.

Setting reminder: SIM-CoT GPT-2 (124M), 6 latent steps, greedy answer decode 204/500 on GSM8K val. Plan: K=4 candidates per step (own step + MC-dropout reruns), GRPO-trained selector, frozen generator. Option (b) "keep alive": the selector keeps several candidates per step, chains branch, and the answer step aggregates over all kept chains. Per-step features from a frozen step decoder (validity, groundedness, decoder confidence). No trained PRM. Seed noise is about 9/500.

---

## 1. Verifier-weighted voting and path-score aggregation

### Lightman et al. 2023, "Let's Verify Step by Step" (arXiv 2305.20050) [full text, PDF]
- Solution score = **product** of per-step P(correct) (neutral counted as positive). App. F.2, Table 4, best-of-1860 on the MATH subset: product/neutral=pos **78.2%**, product/neutral=neg 77.4%, min/neutral=pos 77.6%, min/neutral=neg 77.8%. They say the differences are minor, and that product "does create a slight bias against solutions with a larger number of steps."
- Best-of-1860: PRM **78.2%**, ORM 72.4%, majority voting 69.6%.
- "We experimented with using RM-weighted voting (Li et al., 2022; Uesato et al., 2022) ... but this did not noticeably improve performance." (Sec. 3). So with a strong PRM at huge N, weighted voting did not beat PRM best-of-N.

### Uesato et al. 2022, "Solving math word problems with process- and outcome-based feedback" (arXiv 2211.14275) [full text, PDF]
- K=96 samples, T=1.0. **RM-weighted decoding**: f* = argmax_f Σ_{i: ans(y_i)=f} rm_prob(y_i), then return the top-RM sample within f*. They say it beats plain top-RM selection by "about 1% final-answer error with the SFT model, slightly more with RL".
- Table 3, GSM8K final-answer **error** (greedy / majority / RM-weighted): SFT 41.1 / 22.3 / **14.1**; SFT+ORM-RL 31.2 / 17.8 / **12.7**; Few-shot+ORM-RL 31.5 / 18.8 / 13.8.
- **Directly relevant to per-step selection**: "step-level reranking, by applying an argmax over K options at each step ... increased final-answer error by about 1% with the PRM, and 3% with the ORM." Their manual check put this down to *insufficient policy entropy*: "often, all K continuations would make the exact same mistake." With solution-level reranking, small early differences gave more varied completions. Sec. 2.5 also says per-step RM reranking was worse by 1-2%.
- Selective prediction: abstaining on 30% of inputs by RM score cuts error 14.1% -> 2.7%.

### Math-Shepherd, Wang et al. (arXiv 2312.08935) [full text]
- PRM solution score = **min** over steps ("we use the minimum score across all steps").
- "SC + RM": a_sc+rm = argmax_a Σ_i 1(a_i=a)·RM(p,S_i), an unnormalized sum per answer group.
- 256 candidates (Table 1), GSM8K / MATH500:
  - LLaMA2-70B MetaMATH: SC 88.0/39.4; ORM 91.8/40.4; SC+ORM 92.0/42.0; Math-Shepherd 93.2/44.5; SC+Math-Shepherd 92.4/45.2.
  - LLemma-34B: SC 82.6/44.2; ORM 90.0/43.7; SC+ORM 89.6/45.4; MS 90.9/46.0; SC+MS 89.7/47.3.
  - DeepSeek-67B: SC 88.2/45.4; ORM 92.6/45.3; SC+ORM 92.4/47.0; MS 93.3/47.0; SC+MS 92.5/48.1.
  - Mistral-7B (Table 3, before PPO): SC 83.9/35.1; ORM 86.2/36.4; SC+ORM 86.6/38.0; MS 87.1/37.3; SC+MS 86.3/38.3.
- Pattern: on GSM8K, PRM best-of-N slightly beats PRM-weighted voting. On MATH, weighted voting is best.

### DIVERSE / step-aware verifier, Li et al. (arXiv 2206.02336; ACL 2023 long 291) [full text, ACL PDF]
- Voting verifier: ŷ = argmax_y Σ_i 1(y_i=y)·f(x,z_i,y_i), where f is the verifier probability. M = 100 paths (5 prompts × 20 samples).
- Table 5, GSM8K (Voting / Verifier-only / Voting Verifier): davinci 21.3/27.0/30.6; text-davinci-002 61.3/62.7/68.9; **code-davinci-002 80.0/65.9/82.3**. With the strongest generator, verifier-only best-of-N lost **14 points** to plain voting, while weighted voting gained 2.3. A weak verifier relative to the generator makes BoN dangerous and weighted voting safe.
- Step-aware (Table 6, GSM8K, without/with step-level loss): 30.6->30.9; 68.9->70.2; code-davinci-002 **82.3->81.5** (slightly worse). Gains from step-level labels are small.
- Step labels: a step is negative if it never appears in any positive path (Fig. 3). This is a cheap "consensus" step label, similar to what our step decoder could give.

### Snell et al. 2024, "Scaling LLM Test-Time Compute Optimally..." (arXiv 2408.03314) [full text, first 100k chars]
- PRM solution score = **last-step prediction** ("Rather than aggregating the per-step scores by taking the product or minimum, we instead use the PRM's prediction at the last step"). App. E: "last performs the best across all aggregation strategies". They attribute this to soft MC-return labels, because the last step then acts like an ORM.
- Across samples: "best-of-N weighted" (sum verifier scores per final answer, pick the max). For revisions, hierarchical weighting misbehaved with few trajectories, so they pooled answers with majority voting.
- Beam search vs BoN (PaLM 2-S*, MATH, budgets 4-256, beam width √N or 4): beam search clearly wins at small budgets and loses at large ones. It degrades on easy questions (PRM exploitation) and wins on medium/hard questions. Exact numbers are only in figures.
- Compute-optimal policy: more than 4× efficiency over BoN (16 vs 64 generations).

### Zhang et al. 2025, "The Lessons of Developing PRMs" (arXiv 2501.07301) [full text]
- Table 13, Best-of-8 average over 7 tasks, policy Qwen2.5-Math-7B-Instruct (last / product / min):
  - PRM800K-trained 64.7 / 64.9 / **65.6**; Math-Shepherd-MC-trained **65.4** / 64.3 / 64.9; Qwen2.5-Math-PRM-7B 66.1 / **67.6** / 67.1; PRM-72B 68.5 / **69.3** / 69.2.
  - Lesson: **MC-trained PRMs favor "last"** (MC labels look forward). **Human- or judge-labelled PRMs favor product or min.** The spread is at most about 1.5 points.
- maj@8 vs PRM BoN@8 (Table 6, product): maj@8 GSM8K 96.7, MATH 87.1, avg 66.2; PRM-7B 97.1 / 88.0 / 67.6; PRM-72B 97.6 / 88.7 / 69.3. With a 72B policy (Table 9), PRM-7B averages 69.9, below maj@8 at 70.2.
- No PRM-weighted voting is evaluated.
- "Process to outcome" drift: in several open PRMs (Math-Shepherd, Skywork, EurusPRM), over 40% of minimum step scores fall on the final answer step. These PRMs act like ORMs.

### Liu et al. 2025, "Can 1B LLM Surpass 405B LLM?" (arXiv 2502.06703) [full text]
- A systematic grid of {PRM-Min, PRM-Last, PRM-Avg} × {Max (pick the top solution), Vote (sum scores per answer)} plus majority. MATH-500, Table 2 (Skywork-PRM-7B / Qwen2.5-Math-PRM-7B):
  - Majority 86.8 / 87.6; Min-Max 83.0 / 87.4; Min-Vote 86.6 / 87.6; Last-Max 84.4 / 87.6; **Last-Vote 87.0** / 87.6; Avg-Max 85.8 / **87.8**; Avg-Vote 86.8 / 87.6.
  - Lesson: **Vote is more robust than Max** (Skywork: Max variants lose up to 3.8 points, Vote variants are within 0.2 of majority). The best path aggregator depends on the PRM, and with a good PRM nothing matters much.
- The best TTS method depends on policy size: search beats BoN for small policies, BoN wins for large ones. Under 7B: BoN on easy, beam on hard.
- PRM failure modes: over-criticism, error neglect, error-localization bias, and a length-linked scoring bias.

### HF "Scaling test-time compute with open models" blog (Beeching, Tunstall, Rush; Dec 2024) [full text, blog HTML]
- Llama-3.2-1B-Instruct, RLHFlow Llama3.1-8B PRM, MATH-500. Step reduction min/prod/last: "last" performed best (no numbers).
- Weighted BoN (sum RM per answer) "consistently outperforms vanilla Best-of-N, especially with larger generation budgets". Majority plateaus around N=64.
- Beam search N∈{4,16,64,256}, M=4: at N=4 it matches BoN@16, and at N=16 it matches BoN@256.
- **DVTS** (diverse verifier tree search): split the N beams into N/M **independent subtrees** to stop the whole beam collapsing onto one high-reward prefix. DVTS > beam search at large N on easy and medium problems; beam search is better at small N.

### DeepConf, Fu et al. 2025 (arXiv 2508.15260) [full text, first 100k chars]
- Confidence = token confidence aggregated over windows: average trace (self-certainty), group (sliding window), bottom-10% group, lowest group, tail (last 2048 tokens). The **local/min or tail** measures beat the global mean.
- Weighted vote V(a) = Σ_t C_t·1(ans_t=a), after keeping the top η% of traces (η = 10 or 90).
- K=512 offline (Pass@1 / Cons@512 / Tail-10% / Bottom10-10%): DeepSeek-8B AIME25 76.9 / 82.3 / 87.4 / 87.5; GPT-OSS-120B AIME25 91.8 / 97.0 / **99.9** / 98.1; Qwen3-32B HMMT25 51.9 / 63.3 / 62.9 / 63.3.
- Filtering hurts in 8 of 20 settings (by -0.31 to -4.69). Examples: GPT-OSS-120B HMMT25 88.9 vs 92.9 majority; Qwen3-8B AIME25 online 78.1 vs 82.5. The failure mode is "confidently wrong".
- Online early stopping saves up to 84.7% of tokens.

### Self-certainty, Kang, Zhao, Song (arXiv 2502.18581; NeurIPS 2025) [full text]
- Self-certainty = mean KL(Uniform || p) over tokens. Borda vote: rank r gets (N-r+1)^p votes. p=0 is majority; large p is pure BoN.
- Llama-3.1-8B-Instruct, MATH N=64: majority 63.40; self-certainty BoN 56.70; Borda p=1.2 **64.10**; GSM8K N=64: SC 90.99, Borda p=0.3 91.07. The gains over majority are +0.1 to +0.7.
- Table 2, MATH: weighting by **average** self-certainty per answer group is catastrophic (N=64: 32.94 vs 63.40 majority). **Sum** is ≈ majority (63.51). Lesson: always **sum** weights per answer group and never average, because averaging throws away the count.

### CISC, Taubenfeld et al. (arXiv 2502.06233; Findings ACL 2025) [full text]
- c̃_i = softmax(c_i / T) within a question; â = argmax_a Σ_i 1(a_i=a)·c̃_i. T→∞ gives majority and T→0 gives BoN, so **T is the knob between voting and selection**.
- P(True) works best: 41% (budget 5) and 46% (budget 10) sample reduction vs SC; accuracy gain +1.6 / +1.1. Verbal confidence is worst (10-30%).
- **Within-question discrimination (WQD)**: the fraction of (correct, incorrect) pairs from the same question that the confidence ranks correctly. P(True) 62.3%, response prob 59.0%, verbal 52-56%. WQD ranks the methods the same way CISC gains do, while calibration (ECE) does not. **This is the right pre-check for our step-decoder features.**

### Universal Self-Consistency, Chen et al. (arXiv 2311.17311) [full text]
- The LLM itself picks the most consistent of the concatenated responses. Default 8 samples.
- GSM8K / MATH: PaLM 2-L SC 90.4/37.9 vs USC 90.2/37.4; gpt-3.5 SC 78.5/38.0 vs USC 77.8/38.1. USC **matches but does not beat SC** when answers are parseable, and it gets *worse* on GSM8K at 16 samples (long context, counting errors).

---

## 2. Learned aggregators (RL/SFT) over multiple solutions

### AggLM, Zhao et al. 2025, "The Majority is not always right" (arXiv 2509.06870) [full text]
- Aggregator Qwen3-1.7B, GRPO (group 8, KL 0.001), binary reward (math_verify). Reads **full solutions** (post-think text), k=8. Data: DeepScaleR ~40k problems, 128 samples/problem, 446k training sets.
- Easy/hard mixing: hard = majority answer wrong. Keep all hard plus p% easy. The p ablation (AIME24/AIME25/HMMT24/HMMT25): 0% 64.22/46.06/27.80/28.73; **50% 70.69/50.00/33.34/32.07**; 270% 66.20/46.70/30.01/28.94; untrained 63.57/44.85/29.52/27.91.
- Table 1 (Qwen3-1.7B thinking solutions, k=8): pass@1 50.91/35.68/22.45/22.84; pass@8 76.48/61.38/36.67/44.27; **majority 67.92/45.89/29.01/26.72**; BoN AceMath-72B 56.64/40.35/29.58/21.99; **weighted majority AceMath-7B 64.09/39.49/25.04/17.71** (below majority!); prompted 1.7B 63.57/44.85/29.52/27.91; **AggLM 70.69/50.00/33.34/32.07**.
- On Qwen3-8B solutions: AggLM 82.38/79.70/53.01/60.66 vs majority 81.61/78.70/44.58/56.35.
- Gains are concentrated where the majority answer set is small (diverse, minority-correct). It is on par with majority when the majority answer is frequent. With 8 solutions AggLM beats majority@16 on 3 of 4 benchmarks.
- Lesson for us: an off-the-shelf RM used for weighting **can lose to plain majority**, while a learned aggregator trained on a hard/easy mix beats it.

### Sample Set Aggregator (SSA), Qi, Ye, Tang, Zhu, Choi (arXiv 2506.09014) [full text, first 100k]
- SSA 0.5B/1.5B/3B (Qwen2.5 base) reads the question plus K full candidate solutions concatenated in random order. GRPO, reward 1 / 0.05 (format) / 0. Trained with K=5 candidates from Qwen2.5-7B-Instruct. Training K=8 did not beat K=5.
- K=5, Qwen2.5-7B candidates (GSM8K / MATH / AIME24 / AMC23 / Olympiad / avg): pass@5 95.45/78.00/16.67/67.50/41.10/59.74; **majority 91.66/68.20**/10.00/47.50/31.01/49.67; Qwen PRM-72B 92.87/69.6/13.33/57.50/33.68/53.40; USC 7B MATH 61.20; **SSA-3B 93.25/76.80**/13.33/57.50/39.76/**56.13**.
- On GSM8K the gain over majority is small (+1.6). On MATH it is large (+8.6). Pass@5 caps it.
- Across K (avg): K=15 maj 54.73 / PRM-7B 56.02 / SSA 59.12; K=64 maj 54.43 / PRM 58.49 / SSA 59.78 (two-stage tournament for K≥32).
- SFT vs RL: comparable at 0.5B/1.5B, and **RL clearly better at 3B** (55.22 vs 48.12) and at K=10.
- SSA **mostly selects**. It fails when no candidate is correct, and attempts to synthesize new answers did not help. RL shrank its thinking, and a no-think variant lost little.
- Overhead: 0.55 s/question vs 5.86 s for the 72B PRM.

### GenSelect, Toshniwal et al. (arXiv 2507.17797; ICML 2025 AI4Math workshop) [full text]
- The LLM reads N candidate **summaries** at once and picks one. Knockout tournament for large N (64 candidates = 5 comparisons in 16-way rounds).
- QwQ-32B solutions on Comp-Math-24-25: pass@64 85.2; maj@64 68.4; Qwen2.5-Math-RM-72B 66.8; pointwise GenRM 69.1; **GenSelect@8 73.4**.
- Gen(N)+GenSelect(N) vs Gen(2N): QwQ N=16 71.9 vs 68.0. DeepSeek-R1 N=4 82.4 vs 82.8 (worse at small N).

### Mixture-of-Agents, Wang et al. (arXiv 2406.04692) [full text]
- 3 layers × 6 proposers, with a synthesizing aggregator (not a ranker). AlpacaEval 2.0 LC 65.1% vs GPT-4o 57.5%.
- An LLM *ranker* that selects one proposer answer does "significantly worse" than synthesis (figure only).
- MATH aggregator accuracy over layers 1/2/3: Qwen1.5-110B 0.500/0.570/0.576; LLaMA-3-70B 0.456/0.584/0.578.
- Single-proposer diversity (same model, temp 0.7) is weaker than multi-model: n=6 gives 56.7 vs 61.3.

---

## 3. Parallel thinking: spawn, join, width reward

| Paper | Spawn | Join | Width handling | Numbers |
|---|---|---|---|---|
| Parallel-R1 (2509.07980) [full] | model emits `<Parallel>`; N paths free, ~2 in cases | `<Summary>` after paths, main chain continues | Seen: R = R_par × R_acc; Unseen S2: 10-step windows, 20% tiered reward (+1.2 parallel&correct / +1.0 correct / -1) | Qwen3-4B-Base; GRPO-DAPO avg 45.1 vs Seen 48.9 (AIME25 14.8 -> 19.2). Parallel ratio: acc-only 13.6%, par-only 80.3%, alternating 63.0%. AIME25 peak 25.6 in 2-stage "scaffold" |
| ParaThinker (2509.04475) [full] | fixed P paths with `<think i>` tokens + **thought-specific embeddings added to K/V** | **one summary decode attends to all paths' KV caches** (no re-prefill) | P fixed (2/4/8); First-Finish stop | 1.5B avg: seq-32K 50.9, maj 8×16K 60.4, ParaThinker 8×16K **63.2**; 7B: 69.5 / 76.4 / **77.0**. vs maj at same P: +4.3 (1.5B), +2.0 (7B), shrinking with P (7B: +3.9, +1.6, +0.6). Re-prefill baseline degrades with P (1.5B 47.1 -> 35.7). No thought embedding: 48.1 -> 46.7 at P=8. Flattened positions: "severe drop". +maj@16 on top: AIME24 P=4 43.3 -> 66.7 |
| Multiverse (2506.09991) [full] | model writes `<Goal>`/`<Outline>`; engine counts outlines | `<Conclusion>` attends to all paths; paths share start position; Reduce at max path position; KV merged by index concatenation (radix) | SFT only; RL listed as future work | 32B: AIME24 53.8 vs AR 54.6; AIME25 45.8 vs 45.0; actual parallelism 1.15-1.18; up to 2× speedup |
| APR (2504.15466; COLM 2025) [full] | `spawn(msgs)`: parent writes child contexts | `join(msg)`: child returns only the result; parent never sees child traces | **no width penalty; RL increased children 6.1 -> 8.2** and most gain came from more compute (forced 10 children: 83.2 pre-RL, 83.3 post-RL) | Countdown, 228M model: 83.4% vs 60.0% (4k ctx); 80.1 vs 66.6 cons@7 at 20k tokens |
| Hogwild! (2504.06261; NeurIPS 2025) [full, partial] | N workers prompted, training-free | shared KV via RoPE query rotation; final answer prompted after concatenation | none | works for large models; **Qwen3-1.7B fails/distracted**; 2 workers 1.8×, 4 workers 3.4× tokens/s |
| Group Think (2505.11107) [full] | N thinkers, token-interleaved shared KV | answer from prompt + all thinkers' CoTs | none, training-free | latency halved at N=4 on Floyd-Warshall; no table numbers |
| NPR (2512.07461) [full, ~all] | model plans `<guideline>`/`<plan>` | `<takeaway>` after `<step>`s; masks block cross-step attention; positions reset | PAPO: **one advantage per sample shared by all branches**; ±1 reward; no width penalty | Qwen3-4B avg 64.4 (SFT 58.9); Instruct: NPR 65.0 vs sequential RL 62.0; 4.6× throughput; 100% parallel trigger rate |
| Multiplex Thinking (2601.08808) [full] | sample K tokens per step | **merge into one embedding**: c = Eᵀ(mean one-hot ⊙ w); w uniform or LM-head prob | K=3 default; RL log-prob = sum of K sample log-probs | R1-Distill-7B: AIME24 20.6 vs discrete RL 17.2; averaged and weighted merges "nearly identical"; K=1 -> 2: AMC 44.7 -> 49.6, diminishing after |
| Parason (2608.24658) [abstract only] | subtask vs trial parallelism | trial branches verified and aggregated | PA-GRPO balances accuracy, latency, parallel ratios | ~1.7× speedup on AIME24/25 |

Takeaway for width: with a pure correctness reward and no cost, RL **widens** (APR). Papers that want a target width shape it with a reward (Parallel-R1) or fix it (ParaThinker). No paper found uses an explicit per-branch compute penalty in a reasoning RL objective, and Parason's latency term is the closest.

---

## 4. Tree-level aggregation

### REBASE (Wu et al., arXiv 2408.00724) [full text]
- Expansion: the children per node are a softmax share of the depth budget, W_j = round(B·exp(R_j/T_b)/Σexp(R_k/T_b)), T_b=0.1. The budget shrinks as solutions complete. The node reward is the PRM score of the prefix.
- Final answer: weighted majority vote (RM-weighted). They prove weighted voting has a higher accuracy limit than majority when the RM is better than random.
- Table 4 (MV / BoN / WV): Mistral-7B sampling-32 MATH 36.1/40.3/39.7, GSM8K 85.7/89.4/89.1; **REBASE-32 MATH 44.1/45.4/45.0, GSM8K 88.8/89.4/89.8**; Llemma-7B REBASE-32 MATH 46.1/44.1/46.8, GSM8K 86.1/86.9/87.3.
- **With tree search the gaps between voting methods shrink**, because search already concentrated on high-reward leaves. Majority over tree leaves jumps the most (36.1 -> 44.1) since the leaves are pre-filtered.
- MCTS underperforms sampling at each budget (figures).

### Forest-of-Thought (arXiv 2412.09078, preprint) [full text]
- n trees (MCTSr on math). Final: majority vote across trees, then an "expert" LLM when there is no consensus (CGED).
- GSM8K Table 4 (MV / expert / CGED): n=3 81.31/81.88/84.31; n=5 83.44/83.70/85.44.
- Qwen2.5-7B: GSM 88.48 -> FoT n=8 96.89 (with self-correction). There are internal inconsistencies in its rStar comparisons.

### TS-LLM (Feng et al., arXiv 2309.17179) [full text, partial]
- Aggregation options: Majority, ORM-Max, ORM-Vote. Intermediate values drive search through backup (MCTS-α backs up the value to all ancestors). Final ranking uses ORM on complete answers.
- At equal tokens, CoT-SC_MAJ GSM8K 46.8 vs CoT-SC_ORM 52.3. CoT-SC beats tree search on GSM8K at equal tokens, and tree-search aggregation gains less on small problems.

### TreeRL (Hou et al., arXiv 2506.11902; ACL 2025) [full text]
- Node value V(s) = fraction of correct leaves in the subtree. Step reward = (V(s)-V(root)) + (V(s)-V(parent)), **divided by √|L(s)|** for non-leaf nodes. This downweights nodes that have many descendant leaves, a shared-prefix correction for credit.
- Qwen-2.5-14B avg 41.6 (ChainRL) -> 44.5. GLM4-9B 27.2 -> 29.3.

### Correlated votes / shared prefixes
- HF DVTS: split into independent subtrees so that one high-reward prefix cannot dominate.
- AgentAuditor (arXiv 2602.09341) [abstract only]: argues the effective number of independent votes is much smaller than N when traces share prefixes or biases ("confabulation consensus"). It builds a reasoning tree of agreements and divergences and adjudicates at branch points. Up to +5% over majority.
- Shu 2026, "Signed Pivotal Value in Correlated Self-Consistency" (arXiv 2609.26144) [abstract only, single-author preprint]: replacing a ballot with a verifier only changes ties or one-vote margins. A different-model verifier gave +24.2 pp on the pivotal MATH-500 stratum, while **same-source signals lost accuracy on the pivotal stratum (65% -> 44%)**. This is weak evidence, but our step-decoder features are same-source.
- I found no paper that proposes an explicit shared-prefix discount for leaf voting. TreeRL's √|L| and DVTS's independent subtrees are the closest.

### Answers from intermediate levels
- "Beyond the Last Answer" (Hammoud et al., arXiv 2504.20708) [abstract only]: complete an answer from each intermediate "subthought" and take the mode. Up to +13% AIME24 and +10% AIME25 over the final answer.

---

## 5. Closest latent-space precedent

### Parallel Test-Time Scaling for Latent Reasoning Models (You et al., arXiv 2510.07745; ACL 2026 long) [full text, v3]
- COCONUT and CODI on GPT-2 with T=6 latent steps; CoLaR on Llama-3.2-1B. Sampling by **MC-dropout** (after FFN; p tuned, p=0.2 for RM data) or additive Gaussian noise. MC-dropout gives higher coverage.
- LatentRM: GPT-2 124M (from a COCONUT checkpoint) with a scalar head on the prompt plus the latent prefix, one score per step. Labels come from 128 stochastic rollouts per thought. Loss: softmax over the N candidates at each step (contrastive), not BCE. Trajectory score is the **sum/mean** of step scores.
- COCONUT GSM8K (Test / Hard): deterministic 34.0 / 7.0; Coverage@4 46.7 / 9.5; **majority 33.6 / 6.1; best-of-8 LatentRM 35.4 / 7.8**; BCE-trained 33.5 / 7.4; random head 28.9 / 5.8.
- Beam search with LatentRM is **not better than best-of-N** (comparable on GSM-Test, worse on GSM-Hard).
- So for a GPT-2 latent reasoner with 6 steps and MC-dropout, coverage@4 is about 12-13 points above greedy, yet a *trained* latent PRM recovers only about 1.4 points, and majority is *below* greedy. CODI-specific numbers are only in figures.

### Superposition view (Zhu et al., arXiv 2505.12514; NeurIPS 2025) [abstract only]
- Continuous thoughts can encode multiple search frontiers at once (parallel BFS) for graph reachability. This motivates merge-style aggregation (one latent holding several branches) as an alternative to keeping separate chains.

---

## 6. Inference: aggregation designs for our "keep alive" tree

Notation: depth t = 1..6. Node n at depth t is a kept candidate latent with features φ(n) = (validity, groundedness, decoder confidence, ...). Leaves ℓ at depth 6 give decoded answer a(ℓ). A path is P(ℓ) = (n_1, ..., n_6). The selector keeps S_t(n) ⊆ the 4 children of each alive node.

### Pre-checks (cheap, do first)
1. **Coverage ceiling**: on val, run the tree with the selector replaced by keep-all-up-to-budget B. Measure the leaf oracle (any leaf correct). The LatentRM paper suggests about +12 points of coverage@4 over greedy for COCONUT, while a trained selector recovered about 1.4. If the oracle over B leaves is under about 215/500, no aggregator can clear seed noise.
2. **WQD of step features** (CISC metric): over within-question pairs of leaves with different correctness, how often does a path score (min / product / last / mean of φ) rank the correct leaf higher? About 50% means the features carry no aggregation signal, so use plain majority over leaves. CISC saw 52-62% for LLM confidences.
3. **Leaf diversity**: count distinct answers and distinct decoded step strings per question. Uesato found K step continuations often "make the exact same mistake". If MC-dropout reruns mostly duplicate the main step (likely with a frozen generator), the tree is effectively narrow.

### (i) Tree-weighted voting over leaf answers
- Score: s(a) = Σ_{ℓ: a(ℓ)=a} w(ℓ). **Sum, never average** (self-certainty Table 2).
- Path weight options:
  - w = Π_t q(n_t) (product, as in Lightman), min_t q(n_t) (Math-Shepherd), or q(n_6) (last, as in Snell/HF).
  - q = σ(linear(φ)) fitted by logistic regression on train-set leaves (label = leaf answer correct). This is a 3-feature ORM, cheap.
  - Lessons-paper prior: labels that are local and step-correctness-like (our decoder validity) suit min or product. Forward-looking labels (rollout success) suit last. Expect at most about 1.5 points between them (Lightman, Lessons, Liu), which is below our noise, so pick one by prior and do not tune.
- Temperature: use softmax(log w / T) within a question as in CISC. Tune T on a held-out split. T→∞ is majority, which is the safe fallback because RM-weighted voting can lose to majority (AggLM AceMath rows, Lightman).
- **Shared-prefix correction** (my proposal; no paper does exactly this):
  - Mass-splitting tree: the root has mass 1. Each node splits its mass over its kept children in proportion to their q (or uniformly). The leaf mass is the product of normalized splits, and s(a) sums leaf mass. Each depth then sums to 1, so a branchy subtree gets no extra votes just for branching, and agreement *across different early branches* counts more than agreement inside one subtree. This turns "product along path" into a proper mixture weight.
  - Alternatives: weight by 1/√|siblings| in the spirit of TreeRL, or DVTS-style forced independent subtrees from step 1.
- **Intermediate levels**: optionally decode an answer at every alive node at depths 4-5 (anytime), giving "Beyond the Last Answer"-style votes with a depth discount. Risk: SIM-CoT is trained for exactly 6 steps, so early decodes may be off-distribution. Check per-depth answer accuracy first using the existing depth sweeps.
- Cost: one answer decode per leaf (a few tokens each), plus step-decoder passes (already needed by the selector). Compute is dominated by latent steps: Σ_t (#alive at t) × K forward passes, so with a beam cap B it is at most 6·B·4 one-token steps. Trivial at 124M.
- Risks:
  - Features are same-source and probably low-WQD, so weighted ≈ majority.
  - REBASE shows that once the tree pre-filters, voting variants converge.
  - **Compare against B independent MC-dropout chains + majority at matched latent forward passes.** LatentRM found beam ≤ BoN in latent space, and Uesato found step-level argmax worse than solution-level.

### (ii) Learned reader over kept chains (SSA/AggLM analogue)
- Input: a set of leaves, each as a vector [final latent h_6 or the answer-decode hidden state, answer-token embedding, path-stat features (min/mean/last/product of φ, depth of first branching, subtree size), count of leaves sharing its answer]. The architecture is a tiny set transformer or DeepSets that outputs a logit per leaf. Aggregate as a softmax over leaves, summed per answer, then argmax.
- Training:
  - Gold answers are known, and candidates are a finite list. A supervised loss (log Σ_{ℓ correct} p(ℓ)) on train-set trees is the cheap baseline.
  - GRPO with 0/1 reward is the SSA/AggLM recipe. SSA saw SFT ≈ RL at small sizes and RL better at 3B.
  - Use AggLM's mix: all minority-correct questions plus about 50% majority-correct, and drop all-wrong and all-right questions.
  - Train on trees from the *frozen* selector, or alternate, to avoid co-adaptation.
- Expected gain: SSA/AggLM gains come where the correct answer is a minority. On GSM8K, SSA added only +1.6 over majority (vs +8.6 on MATH). It cannot exceed leaf coverage.
- Cost: a train-set tree dump (7.4k questions × B leaves × small vectors) and a model of about 1-5M parameters. Inference is negligible.
- Risks:
  - Overfitting answer priors such as small numbers.
  - The reader can learn "count" and become majority with extra steps.
  - The selector and reader must be evaluated jointly at fixed B.
  - Another learned component makes the attribution story harder for the report.

### (iii) One answer decode attending to all kept caches (ParaThinker/Multiverse analogue)
- Layout: shared question KV, then each kept node's per-layer KV entry with a **tree attention mask**. During latent steps a node attends only to its ancestors (SpecInfer/Multiverse-style masks). The answer decode attends to the question plus *all* kept entries.
- Positions: give same-depth entries the same position id (Multiverse). GPT-2 uses learned absolute positions added at the input, so siblings become indistinguishable except by content. ParaThinker needed **trainable branch embeddings on K/V**, and dropping them cost 1.4 points at P=8, while flattened positions cost "severely".
- Weighted variant: add log w(n) as an attention-logit bias on node n's keys. This performs soft weighted voting inside attention.
- Merge variant (Multiplex Thinking analogue): at each depth, replace the kept siblings by one weighted-average latent (or average their K/V). This collapses the tree back to a single chain of 6 entries. Multiplex found plain and weighted averages "nearly identical", but needed RL.
- Cost: cheapest at inference (one decode). With a custom 4D mask it is easy in HF GPT-2.
- Risks (high, especially with a **frozen** generator):
  - The decoder was trained on exactly 6 latent entries and would see 6·B.
  - Project memory: "copied cache entries alone reproduce the drift" and dedup restores the 6-step score. Side-by-side near-duplicate siblings are exactly the duplication that hurt us.
  - Hogwild-style training-free shared cache **failed on Qwen3-1.7B**, so a 124M model is unlikely to cope zero-shot.
  - ParaThinker shows that naive concatenation (re-prefill) degrades as width grows (1.5B: 47.1 -> 35.7 from P=2 to 8).
  - Making this work would likely need fine-tuning the answer decoder (LoRA) on multi-branch caches, which breaks the frozen-generator premise.
  - The merge variant risks producing off-manifold latents. It does avoid the duplication problem, since it is one entry per depth, and lines up with the project's dedup finding.

### Width and credit in the selector's RL (affects every option)
- With a 0/1 reward and free keeps, GRPO will learn to keep everything (APR: RL raised children 6.1 -> 8.2, and the gain was mostly compute).
- Add a per-kept-node cost λ·|S_t|, or a hard beam cap B (top-B by selector logit), and report accuracy against latent-forward-pass count.
- Credit: NPR shares one advantage over the whole tree. TreeRL instead gives per-node advantages from subtree success rates (GA+LA, /√|L|). In our tree, node value = fraction of correct leaves below each keep decision, which is free once the tree is rolled out.

---

## 7. Could not verify / caveats
- Snell App. E exact numbers for min/prod/last were not extracted (figure only). The direction ("last" best) is quoted.
- HF blog DVTS numbers are figure-only.
- DIVERSE venue: confirmed through ACL Anthology 2023.acl-long.291 (title match). The arXiv page lists no venue.
- MUX (arXiv 2607.18264): the abs page says submitted 19 May 2026, which is odd for a 2607 ID. Not used.
- Group Think and Hogwild have no tabulated accuracy numbers in their text.
- Parallel-R1's "42.9% on AIME25" baseline is ambiguous: 25.6 vs 17.9 from an ablation, not the 14.8 GRPO row.
- AgentAuditor, Shu 2026, Parason, SoftCoT++, Superposition, Beyond-the-Last-Answer: abstract only.
- LatentRM (2510.07745): CODI-specific BoN and majority numbers are only in figures, and the Table 1 caption (N=32) conflicts with the text (N=8).
