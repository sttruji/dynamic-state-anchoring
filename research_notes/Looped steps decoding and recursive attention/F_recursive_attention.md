# F. Recursive attention: attention mechanisms where one attention computation shapes or selects another, assessed as a learned, query-dependent generalization of dedup_soft for CODI / SIM-CoT latent KV entries

Notes compiled 2026-11-05. Scope: prior art where a second attention-like computation (key-key, head-head, query-query, or a coarse selector) shapes the logits or selection of the main attention; whether any of it fixes dilution among near-duplicate latent entries; how to hook it into GPT-2 CODI (`GPT2Attention._attn` bias patch); and whether it can raise accuracy above the 6-step score.

Evidence levels used below:
- **(abs)**: arXiv abstract and metadata, pulled from the arXiv API on 2026-11-05. The id, title, date and authors are verified.
- **(FT-summary)**: the full text, read through a WebFetch summary this session. The details are as the summarizer reported them.
- **(prior notes)**: checked in full text in this project's earlier notes (`research_notes/Latent thought attention gating/`). I re-cite the primary URL.
- **(derivation)**: my own algebra. It has no source and lives in the Inferences sections.
- **2026 preprint**: first posted in 2026.

Project facts come from local files: the memory notes and `scripts/latent_gate/`.

---

## Q1. Prior art: which attention mechanisms let one attention computation shape or select another? What does each do, was it tested on reasoning or length extrapolation, and does it touch the duplicate-dilution problem?

### Takeaway
"Attention on attention" is a crowded family, but almost all of it targets something else: expressivity (talking heads, DCMHA, MTA, 2-simplicial, Nexus), noise or sinks (Diff, Cog, Lazy, gated, softpick), recency or forgetting (FoX, stick-breaking, Selective attention), global sharpness (SSMax, SSA), or efficiency (NSA, MoBA, Quest, R-KV).

Only a handful of mechanisms compute a **key-to-key** quantity and feed it back into the main attention:
- ToMe's proportional attention (+log s). It is the algebraic inverse of dedup_soft.
- R-KV and KeyDiff, which use key-cosine redundancy for eviction.
- LUCID (2026), an exponentiated key-key kernel used as a preconditioner.
- CoPE's gated counts. They are a mechanism for counting.

None of these four is a soft −log(multiplicity) bias inside softmax, and none was tested on latent CoT. Almost nothing in the family was tested on multi-step math reasoning. Dreamer (GSM8K/MATH) is the exception, and it has no redundancy treatment.

### Cited Findings

**A. Head-to-head and score-to-score composition ("attention applied to attention maps")**
- **Talking-Heads Attention.** Shazeer, Lan, Cheng, Ding, Hou (2020). It adds learned linear projections across the heads dimension "immediately before and after the softmax". It improves MLM perplexity and transfer to comprehension and QA. It was not tested on reasoning or extrapolation. (abs) — [arXiv:2003.02436](https://arxiv.org/abs/2003.02436)
- **DCMHA (Dynamically Composable Multi-Head Attention).** Xiao, Meng, Li, Yuan. ICML 2024 oral. A "Compose function transforms the attention score and weight matrices in an input-dependent way". It matches models with about 1.7–2.0× the compute on LM. This is a query-dependent version of talking heads. (abs) — [arXiv:2405.08553](https://arxiv.org/abs/2405.08553)
- **Multi-Token Attention (MTA).** Golovneva, Wang, Weston, Sukhbaatar (2025). (abs + FT-summary) — [arXiv:2504.00927](https://arxiv.org/abs/2504.00927)
  - **Mechanism.** It applies a 2-D convolution over the query and key axes of the pre-softmax logits, A = Softmax(Conv2d_θ(Â)), with c_q = 6 and c_k = 11 in the LMs. Causal masking is applied twice (Mask₀ before the conv, Mask₋∞ after), which "masks out a little more than necessary". There is also a post-softmax variant, a head-mixing convolution over groups of c_h = 16 heads, and group norm with a scalar gate.
  - **Size and init.** About 29.6K extra parameters on an 880M model. The intro says 0.001%, which does not match the table. Kernels are identity-initialized, which "corresponds to initialization with a regular Transformer". Zero init cost +0.02 PPL.
  - **Training.** The 880M model is trained from scratch on 105B SlimPajama tokens. A preliminary continued-training experiment on Llama 3.2 1B/3B and 3.1 8B used 5.3B tokens.
  - **Results.** Average 44.9 vs 43.7 on standard zero-shot benchmarks, and gains on Needle-in-a-Haystack (2–6 needles) and BabiLong QA1–5. There is no math-reasoning benchmark, and no length extrapolation beyond the 2K→4K context extension.
  - **Cost.** It is "not currently compatible" with optimized attention kernels: throughput 5.7K vs 54.3K tok/s, and memory 73.8 vs 17.5 GB (unoptimized).
- **Re-attention (DeepViT).** Zhou et al. (2021). Deeper ViTs suffer "attention collapse": "attention maps gradually become similar and even much the same after certain layers". Re-attention regenerates the maps by mixing heads with a learnable matrix. It was tested on vision only. (abs) — [arXiv:2103.11886](https://arxiv.org/abs/2103.11886)
- **2-simplicial attention.** Roy et al. (2025). It "generalizes standard dot-product attention to trilinear functions" with a Triton kernel, motivated by token efficiency. Its logits involve the query and *two* keys, so key-key interactions are expressible in principle. Pretraining-scale only. (abs) — [arXiv:2507.02754](https://arxiv.org/abs/2507.02754)
- **Nexus.** Chen et al. (Dec 2025). "The Query and Key vectors are themselves outputs of inner attention loops." It uses weight sharing across the recursive steps, so it adds "O(1) additional parameters". This is the literal "recursive attention" in name, but its aim is expressivity, not redundancy. (abs) — [arXiv:2512.03377](https://arxiv.org/abs/2512.03377)
- **Higher-order Linear Attention.** Zhang, Qin, Wang, Gu (Oct 2025, revised 2026). Second-order interactions computed through streaming prefix statistics. Efficiency-oriented. (abs) — [arXiv:2510.27258](https://arxiv.org/abs/2510.27258)

**B. Attention over attention outputs (two-pass and gating)**
- **Attention-over-Attention reader.** Cui et al. ACL 2017. It "place[s] another attention mechanism over the document-level attention, and induces 'attended attention'". Cloze-style reading comprehension. (abs; the exact column/row-softmax averaging is from my memory and was not re-read) — [arXiv:1607.04423](https://arxiv.org/abs/1607.04423)
- **Attention on Attention (AoA).** Huang et al. ICCV 2019 oral. It builds an "information vector and an attention gate" from the attention result and the query, then multiplies them. This is a gate on the attention *output*, used for image captioning. It cannot re-weight keys against each other. (abs) — [arXiv:1908.06954](https://arxiv.org/abs/1908.06954)
- **Gated attention.** Qiu et al. (2025), NeurIPS 2025 best paper. A head-specific sigmoid gate after SDPA. It removes sinks and helps long-context extension. It is an output gate, so it gives no per-key re-weighting. (abs; prior notes) — [arXiv:2505.06708](https://arxiv.org/abs/2505.06708)

**C. Two maps or signed maps that cancel noise**
- **Differential Transformer.** Ye et al. ICLR 2025 oral. It computes attention as "the difference between two separate softmax attention maps", which "cancels noise". Gains on long context, key-information retrieval and hallucination. Requires pretraining. I did not check its math-reasoning evals. (abs) — [arXiv:2410.05258](https://arxiv.org/abs/2410.05258)
- **Cog Attention.** Lv et al. (2024), "More Expressive Attention with Negative Weights". Attention weights can be negative, and the paper claims robustness against representational collapse "by preventing the over-squashing of earlier tokens into later positions". Tested on decoder-only LM and U-ViT diffusion; no reasoning or extrapolation tests in the abstract. (abs) — [arXiv:2411.07176](https://arxiv.org/abs/2411.07176)

**D. One token masking or decaying others (cumulative additive biases)**
- **Selective Attention.** Leviathan, Kalman, Matias. ICLR 2025. (abs; prior notes, full text) — [arXiv:2410.02703](https://arxiv.org/abs/2410.02703)
  - **Mechanism.** It is parameter-free. Head 0's logits become S. S goes through ReLU so masking can only reduce attention, the BOS column and diagonal are zeroed, and the mask is shifted to future queries and cumulatively summed into F. Attention is softmax(QKᵀ/√d − F). The intuition is that a token masks another "after absorbing its contents".
  - **Results.** It matches about 2× more heads on C4, and allows 16×/25×/47× smaller attention memory at 512/1K/2K context. It trains at each context size, so length extrapolation is not tested.
- **Forgetting Transformer (FoX).** Lin, Nikishin, He, Courville. ICLR 2025. It adds a data-dependent cumulative log forget gate D_ij = Σ log f_l to the logits; ALiBi is the special case. It beats the Transformer on long-context LM and **length extrapolation**. (abs; prior notes) — [arXiv:2503.02130](https://arxiv.org/abs/2503.02130)
- **Stick-breaking attention.** Tan, Yang, Courville, Panda, Shen. ICLR 2025 per the PDF header. Each earlier token gets a break point β = σ(qᵀk), and weights are allocated nearest-first on the remaining stick. This "naturally incorporates recency bias" and needs no positional embedding. A model trained at 2¹¹ context "perform[s] well at 2¹⁴". (abs; prior notes) — [arXiv:2410.17980](https://arxiv.org/abs/2410.17980)
- **Dynamic Context Pruning.** Anagnostidis et al. (2023). Per-layer interaction queries and keys (r = 64) give pairwise drop scores via an annealed α-sigmoid. The scores enter attention as log I, and drops are irreversible through a cumulative product. Fine-tuned **GPT-2**, with up to about 80% of context pruned. This is a learned, pairwise "token drops token" precedent at our scale. (prior notes) — [arXiv:2305.15805](https://arxiv.org/abs/2305.15805)
- **Contextual Position Encoding (CoPE).** Golovneva, Wang, Weston, Sukhbaatar (2024). Positions are computed "by incrementing position only on certain tokens determined by the model", using gated cumulative counts. It solves counting and selective-copy tasks where standard position encodings fail. This is the clearest published instance of an attention-derived *soft count*. (abs) — [arXiv:2405.18719](https://arxiv.org/abs/2405.18719)

**E. Global or query-level sharpening (temperature on the main logits)**
- **SSMax.** Nakanishi (2025), a single-author preprint. It multiplies logits by s·log n with a learnable s per head. Better length generalization, and it can be swapped in after pretraining. (abs; prior notes) — [arXiv:2501.19399](https://arxiv.org/abs/2501.19399)
- **Selective Self-Attention (SSA).** This is a *different* "selective attention", from Zhang et al. (Nov 2024). It uses a query- and position-dependent temperature, which they say "alleviates attention dilution", plus value temperature scaling to suppress noisy tokens. It adds under 0.5% parameters and "can be fine-tuned on existing LLMs". Tested on LM benchmarks only. (abs) — [arXiv:2411.12892](https://arxiv.org/abs/2411.12892)

**F. Sink and "lazy" fixes (letting attention go nowhere)**
- **Lazy Attention.** Fu et al. (Jan 2026), **2026 preprint**. It names "attention overload", where tokens receive comparable high weights and representations collapse, and "attention underload", where no token is relevant and a sink appears. Its fixes are positional discrimination across heads and dimensions, and **Elastic-Softmax**, which relaxes the sum-to-one constraint. Trained on FineWeb-Edu and tested on nine benchmarks, with up to 59.58% attention sparsity; no reasoning or extrapolation tests are named. (abs) — [arXiv:2601.00919](https://arxiv.org/abs/2601.00919)
- **A Unifying View of Attention Sinks.** Fesser, Jacobs, Fel, Keller, Kakade (Jun 2026), **2026 preprint**. It separates "adaptive nop" sinks (negligible value norms) from "broadcast" sinks (low-rank outputs). "Gating eliminates detected nop-like sinks but increases broadcast-like sinks". The study is on ViTs. (abs) — [arXiv:2606.08105](https://arxiv.org/abs/2606.08105)
- **Attention-sink survey.** Su et al. (Apr 2026), **2026 preprint**. A consolidated survey of sink utilization, interpretation and mitigation. (abs) — [arXiv:2604.10098](https://arxiv.org/abs/2604.10098)
- **Softpick.** Zuhri, Fuadi, Aji (2025). A rectified, not-sum-to-one softmax replacement with a 0% sink rate at 340M and 1.8B. (abs) — [arXiv:2504.20966](https://arxiv.org/abs/2504.20966)
- **Active-dormant heads.** Guo et al. (2024). Explains attention sinks, value-state drains and residual peaks as "extreme-token phenomena". (abs) — [arXiv:2410.13835](https://arxiv.org/abs/2410.13835)

**G. A coarse attention pass that selects what the fine attention sees (learned or query-aware block selection)**
- **Native Sparse Attention (NSA).** Yuan et al. (2025). Three branches:
  - compressed blocks;
  - fine blocks selected top-n by the *compressed branch's attention scores*, which makes it literally attention selecting attention;
  - a sliding window.
  
  They are mixed by MLP-sigmoid gates and pretrained end-to-end. (abs; prior notes) — [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)
- **MoBA.** Lu et al. (2025). Mixture-of-block attention under a "less structure" principle, deployed for Kimi long context. Its router (parameter-free top-k by query · mean-pooled block key) is as recorded in prior notes; I did not re-verify it this session. (abs) — [arXiv:2502.13189](https://arxiv.org/abs/2502.13189)
- **Quest.** Tang et al. ICML 2024. "The criticality of a token highly depends on the query." It keeps per-page min and max keys, upper-bounds each page's score for the current query, and loads the top-K pages. (abs) — [arXiv:2406.10774](https://arxiv.org/abs/2406.10774)

**H. Key-key redundancy, merging, and density (the family that actually addresses duplicates)**
- **Token Merging (ToMe), proportional attention.** Bolya et al. ICLR 2023 oral. (FT-summary) — [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)
  - The formula is A = softmax(QKᵀ/√d + log s), where s is "the size of each token (number of patches the token represents)". The motivation: "if we merge two tokens with the same key, that key has less effect in the softmax"; adding log s "performs the same operation as if you'd have s copies of the key".
  - Ablation: off-the-shelf AugReg ViT-L scores 82.15% without proportional attention and 83.51% with it. On MAE it is unnecessary and slightly hurts (84.25% vs 83.84%).
- **SemantiCache.** Wu et al. (Mar 2026), **2026 preprint**. LLM KV compression that clusters tokens within delimiter-bounded chunks, merges them into "semantic cores", and adds a "Proportional Attention mechanism that rebalances the reduced attention contributions of the merged tokens". Up to 2.61× decode speedup. (abs) — [arXiv:2603.14303](https://arxiv.org/abs/2603.14303)
- **R-KV.** Cai et al. (2025). Built for reasoning models. (abs; prior notes) — [arXiv:2505.24133](https://arxiv.org/abs/2505.24133)
  - **Redundancy.** Cosine similarity of L2-normalized keys, with similar pairs above a threshold linked and only the β most recent kept. Then R_i = softmax(mean similarity), combined with attention importance as Z = λI − (1−λ)R, λ = 0.1.
  - **Results.** Lossless at 10% of the cache and 105% of full at 16% on R1-Llama-8B AIME24. "Repetitive content often receives disproportionately high attention scores." Pure attention importance and pure redundancy both perform worst.
- **KeyDiff.** Park et al. NeurIPS 2025. Training-free eviction "based solely on key similarity". "Geometrically distinctive keys … tend to have high attention scores." (abs) — [arXiv:2504.15364](https://arxiv.org/abs/2504.15364)
- **KVMerger** (Wang et al. 2024, [arXiv:2407.08454](https://arxiv.org/abs/2407.08454)) and **D2O** (Wan et al., ICLR 2025, [arXiv:2406.13035](https://arxiv.org/abs/2406.13035)) merge similar or evicted KV instead of dropping it. Both are training-free compression methods. (abs)
- **LUCID Attention.** Duvvuri, Patel, Gupta, Dhillon (Feb 2026), **2026 preprint**. (abs + FT-summary of the first 100K of 120K characters) — [arXiv:2602.10410](https://arxiv.org/abs/2602.10410)
  - **Formula.** LUCID(Q,K,V) = softmax(QKᵀ/√d + M̂) · (M ∘ exp(K_RN K_RNᵀ/√d − √d))⁻¹ · V. Keys are RMS-normalized, so the causal key-key kernel is unit-diagonal. A triangular solve preconditions the *values*; the softmax itself is unchanged.
  - **Purpose.** It "minimizes overlap between the keys in a Reproducing Kernel Hilbert Space".
  - **Cost.** O(N²d) with a TRSM solve, adding 0–5.5% training time and about 1.3% inference time at 32K.
  - **Results.** About 1B models with long-context fine-tuning. +18% on BABILong and +14% on RULER multi-needle; MNIAH-6 at 2K scores 43.6 vs 38.8.
  - **Gaps in the paper.** It does not analyze identical keys explicitly.
- **Robust KDE attention.** Han et al. NeurIPS 2023. This line views self-attention "as a non-parametric estimator based on kernel density estimation" and robustifies it against contaminated samples. It is the theoretical frame in which "duplicates are density". (abs) — [arXiv:2210.05794](https://arxiv.org/abs/2210.05794). Related: **Elliptical attention**, which uses a Mahalanobis metric against "representation collapse" (NeurIPS 2024, [arXiv:2406.13770](https://arxiv.org/abs/2406.13770)). (abs)
- **Modern Hopfield networks.** Ramsauer et al. (2020). The update rule "is equivalent to the attention mechanism". Fixed points include "metastable states averaging over a subset of patterns", which is how near-duplicate stored patterns blur together under softmax retrieval. (abs) — [arXiv:2008.02217](https://arxiv.org/abs/2008.02217)
- **ReAttn.** Tian et al. EACL 2026, **2026 preprint**. A post-hoc re-weighting of attention for LLM re-ranking that applies "cross-document IDF weighting to down-weight attention on query-overlapping tokens". It is an IDF (−log frequency) correction on attention mass, in a retrieval setting. (abs) — [arXiv:2602.19969](https://arxiv.org/abs/2602.19969)
- **DPP attention.** DPP-based attention weights have been used in multi-document summarization to balance relevance and diversity (JAIR 2021, per a search snippet; I did not open the primary). DppNet uses an "inhibitive" attention to approximate DPP sampling (NeurIPS 2019). I found no DPP attention used as a softmax replacement in LLMs. — [arXiv:1901.02051](https://arxiv.org/abs/1901.02051)

**I. Normalizing over the other axis (doubly-stochastic or slot competition)**
- **Sinkformers** (Sander et al., AISTATS 2022) make attention doubly stochastic via Sinkhorn — [arXiv:2110.11773](https://arxiv.org/abs/2110.11773). **Doubly-normalized attention** (Ding et al. 2020) avoids "explaining away" input neurons — [arXiv:2009.14308](https://arxiv.org/abs/2009.14308). **Slot Attention** (NeurIPS 2020) normalizes over slots, so queries compete for keys — [arXiv:2006.15055](https://arxiv.org/abs/2006.15055). All (abs).

**J. Mechanisms from the notes series that recur here**
- **Disagreement regularization** (EMNLP 2018) encourages heads to attend to different positions — [arXiv:1810.10183](https://arxiv.org/abs/1810.10183). (abs)
- **TRIM-KV** (ICLR 2026) learns a per-token, per-head retention gate as a log-bias. On Qwen3-4B AIME24 it scores 75.8 at a 4096 budget vs 65.5 with the full cache. (prior notes) — [arXiv:2512.03324](https://arxiv.org/abs/2512.03324)
- **Think Clearly** (EMNLP 2025 Findings) prunes redundant reasoning tokens by attention to an end-of-thinking token, and accuracy rises. (abs) — [arXiv:2507.08806](https://arxiv.org/abs/2507.08806)

### Inferences
- **The family splits by what the second computation compares.**
  - Query-query and head-head mixing (talking heads, DCMHA, MTA head-mix, Re-attention) and query-level temperature (SSMax, SSA) never compare *keys with keys*. So they cannot tell a duplicated entry from a single one (formal argument in Q3).
  - Mechanisms that *do* compare keys are the only candidates for generalizing dedup_soft: ToMe/SemantiCache, R-KV, KeyDiff, LUCID, Selective attention (by token-to-token logits), Dynamic Context Pruning (pairwise), and CoPE-style counting.
- **ToMe is the cleanest citation for our algebra.** Its proportional attention, "+log s ≡ s copies of the key", is the published statement of our "k copies add log(k+1)" identity, used in reverse. dedup_soft is exactly ToMe *without* proportional attention, applied to unmerged copies.
- **ToMe's ablation is a useful caution.** Removing the multiplicity term costs 1.4 points on supervised ViTs, where multiplicity means patch area, a real signal. Whether multiplicity is signal or artifact is a property of the model. For CODI past step 6 it is an artifact, because training never saw more than 6 distinct latents.
- **LUCID is the closest 2026 mechanism to "key-key attention fed into the main attention".** Its correction acts on values, not on the softmax normalizer, so it is not equivalent to dedup (see Q3).
- **Literal "recursive attention" papers (Nexus, AoA reader, AoA) target expressivity or relevance gating.** A paper on "recursive attention" for latent CoT would need to cite them for the name. The mechanism lineage, though, is ToMe, R-KV and LUCID.

### Gaps
- I did not re-read Diff Transformer, stick-breaking or Cog attention for math-reasoning evaluations. The statements above about which tasks were tested come from abstracts and prior notes only.
- MoBA's router details are from prior notes ([PK] there), not verified this session.
- I could not verify the JAIR 2021 DPP-attention summarization paper (snippet only), and I found no 2025–2026 DPP attention for LLMs.
- I read LUCID only through a summarizer, and its last 20K characters were not read. Its treatment of exact duplicates is my derivation (Q3), not the paper's.
- I did not search OpenReview directly. All coverage comes from arXiv API keyword queries and web search.

---

## Q2. Has anyone applied a learned, redundancy-aware or second-order attention to latent CoT, looped transformers or recurrent-depth models? Who are the closest competitors?

### Takeaway
I found **no paper that applies a key-key (redundancy-aware) or second-order attention over latent-thought KV entries, or over loop-iteration KV, to keep accuracy, let alone raise it, as the number of latent steps grows.** The closest competitors split into four groups:
1. **Attention or decay over loop iterations.** InfiLoop's learned decay (no decay collapses accuracy), RecurTrace's gated, recency-biased window, and Dreamer's depth attention. Dreamer is a 2026 GSM8K/MATH result, with no redundancy handling and no depth extrapolation.
2. **Bounded caches across loops.** MELT, PLT, Huginn and MoR.
3. **Redundancy-aware KV selection on explicit reasoning.** R-KV, TRIM-KV and Think Clearly. KaVa brings R-KV into latent CoT, but only on the teacher.
4. **Generic key-key attention (LUCID) and token merging (ToMe/SemantiCache).** These come with no latent application.

### Cited Findings
- **InfiLoop.** Li et al. (Oct 2026), **2026 preprint**, "Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals". (abs; prior notes, main text) — [arXiv:2610.11570](https://arxiv.org/abs/2610.11570)
  - **Mechanism.** Weights over past loop states are α_{t,i} ∝ exp(qᵀRMSNorm(z_i) − λ(t−i)).
  - **Ablation.** With λ = 0, which is full attention over all past states with no decay, Sudoku-Extreme accuracy is 38.4% / 40.2%, against 82.1% / 89.7% with learned decay.
  - **Depth.** It trains at 12 segments and tests up to 160.
  - **Relevance.** This is the strongest published evidence that attention over an ever-growing set of iteration states fails without a learned correction. The correction is *recency*, not redundancy.
- **RecurTrace.** Wang et al. (Sep 2026), **2026 preprint**. (abs; prior notes, partial main text) — [arXiv:2609.03379](https://arxiv.org/abs/2609.03379)
  - **Mechanism.** Looped layers attend over "a sliding window of the layer's own outputs from up to W earlier loops". The window has a per-head ALiBi-initialized recency bias and scalar and token gates.
  - **Results.** W = 3 is best. Accuracy at 16 loops is stable at 47.5%, below the 54.7% peak.
- **Dreamer (Depth-Recurrent Attention Mixtures).** Knupp et al. (Jan 2026), **2026 preprint**. (abs + FT-summary) — [arXiv:2601.21582](https://arxiv.org/abs/2601.21582)
  - **Mechanism.** "Depth attention" is "a separate instance of attention over the previous depths of the same token", a single head with RoPE over depth. The paper does not address redundant or similar states. Generalization beyond trained depth is "still underexplored" and is not tested.
  - **Results.** At depth 32, DR+DA averages 51.4 vs 39.5 for the layered baseline on GSM8K, MATH, MMLU-math and MathQA. GSM8K is 56.3 vs 49.7. Models are about 1.2–2.1B, trained on 100B tokens.
- **MELT.** Conchello Vendrell et al. (May 2026), **2026 preprint**. A single KV cache per layer shared across loops of Ouro, updated through a learnable gate. 3.98× less KV than Ouro-1.4B at equal accuracy. The loop count is fixed. (abs; prior notes) — [arXiv:2605.07721](https://arxiv.org/abs/2605.07721)
- **Parallel Loop Transformer (PLT).** Wu et al. (Oct 2025). It shares the first loop's KV with later loops and adds "Gated Sliding-Window Attention (G-SWA)". (abs) — [arXiv:2510.24824](https://arxiv.org/abs/2510.24824)
- **Ouro.** Zhu et al. (Oct 2025). The baseline Ouro LoopLM; MELT cites its per-loop KV as "grow[ing] linearly with reasoning depth". (abs; prior notes) — [arXiv:2510.25741](https://arxiv.org/abs/2510.25741)
- **Huginn recurrent depth.** Geiping et al. (2025). Zero-shot cache sharing across recurrences (a modulo-k rule, per prior notes). (abs; prior notes) — [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)
- **Mixture-of-Recursions.** Bae et al. (2025). "Lightweight routers" assign recursion depth per token over a shared layer stack. (abs) — [arXiv:2507.10524](https://arxiv.org/abs/2507.10524)
- **LoopMTP.** Shomali et al. (Aug 2026), **2026 preprint**. Looped models "suffer from latent overthinking and undifferentiated computation". It aligns loop t with the token t steps ahead, and "a lightweight gate preserves useful information across iterations". Stable to 15 loops. (abs) — [arXiv:2608.03624](https://arxiv.org/abs/2608.03624)
- **Adaptive Loops and Memory.** Frey et al. (ICLR 2026 Latent & Implicit Thinking workshop), **2026 preprint**. Per-layer looping with learned halting plus gated memory banks. "Looping primarily benefits mathematical reasoning." (abs) — [arXiv:2603.08391](https://arxiv.org/abs/2603.08391)
- **Training-Free Looped Transformers.** Chen et al. (May 2026), **2026 preprint**. "Naive block reapplication usually degrades performance." Damped sub-steps on a frozen checkpoint give +2.64 pp MMLU-Pro on Qwen3-4B. (abs) — [arXiv:2605.23872](https://arxiv.org/abs/2605.23872)
- **Rivals that avoid appended latents altogether.**
  - **LLoCoT** (Oct 2026, **2026 preprint**) iteratively refines "a compact latent workspace" of slots instead of appending latents — [arXiv:2610.11472](https://arxiv.org/abs/2610.11472). (abs)
  - **PCCoT** (EMNLP 2025) uses Jacobi iteration on a fixed set of latent tokens — [arXiv:2506.18582](https://arxiv.org/abs/2506.18582). (abs) It found 12 latents "much better than using 6" for GPT-2, with 48.2% for its 12-latent baseline. (project memory, [codex-review-takeaways](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/codex-review-takeaways.md))
- **KaVa.** Kuzina et al. ICLR 2026. It compresses a teacher's explicit-CoT KV with an R-KV-style redundancy and importance score, then distills it into a latent student's KV. Eviction is applied to the *teacher* only. (abs; prior notes) — [arXiv:2510.02312](https://arxiv.org/abs/2510.02312)
- **SIM-CoT.** Wei et al. (2025). Training with more implicit tokens "often becomes unstable and collapses", because latents become "homogeneous and los[e] semantic diversity". The fix is step-level supervision, not an attention change. (abs) — [arXiv:2509.20317](https://arxiv.org/abs/2509.20317)
- **Targeted searches for latent CoT plus redundancy or key-key attention returned nothing.** These were arXiv API queries such as "continuous thought" AND redundant, "latent reasoning" AND attention AND redundancy, "implicit chain-of-thought" AND attention, and "latent tokens" AND attention AND dilut*, plus web searches. — searches run 2026-11-05; no source to cite.

### Inferences
- **Ranked closest competitors for "learned redundancy-aware attention over latent entries"**:
  1. **LUCID** (2026): the mechanism, a key-key kernel inside attention, in general LMs.
  2. **InfiLoop** (2026): the problem, attention over a growing set of iteration states collapses, fixed by a learned correction.
  3. **R-KV / KaVa**: key-cosine redundancy, and its import into latent CoT.
  4. **Selective attention**: learned token-to-token down-weighting.
  5. **ToMe / SemantiCache**: the log-multiplicity algebra.
  6. **Dreamer and RecurTrace**: attention over depth, with no redundancy handling.
- **What still looks open.** A query-conditioned, key-key, log-multiplicity bias over *appended latent-thought entries*, evaluated past trained depth, together with the duplicate-copy causal test. The mechanism parts are each published, so the paper would have to win on the application and on a *gain*.
- **InfiLoop's λ = 0 collapse is the published analogue of our 24-step drop.** Its fix (recency decay) is the opposite of what CODI needs: recency scored 141 at 24 steps vs 173 plain on CODI in Stage 0 ([project memory](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/dsa-pivot-latent-eviction.md)). A redundancy-based correction is therefore genuinely different from the loop-model literature, and InfiLoop is a natural baseline to beat.
- **PCCoT, LLoCoT and LOTUS sidestep dilution by construction**, because they do not append latents. If they show depth gains where appended-latent CODI cannot, a reviewer can argue that the right fix is "don't append", not "dedup the appended entries".

### Gaps
- I did not search OpenReview for ICLR 2027 submissions, which would be posted around this date, or check citations of SIM-CoT, LUCID or InfiLoop. A same-month competitor could exist.
- I did not verify Dreamer's KV memory growth numbers or its exact depth-attention parameterization beyond the summary.

---

## Q3. Which design best generalizes dedup_soft? Theory (does it cancel the log(k+1) boost?), cost and stability for each option

### Takeaway
**A per-entry score cannot cancel duplicates; only a set-level (key-key) term can.**
- **Why per-entry fails.** Any bias or temperature computed from one entry alone (MLP gate, SSMax, FoX-by-content, query-aware top-k, head mixing) gives identical copies identical treatment. Cancelling log(k+1) needs to know k.
- **The minimal correct generalization** is the log-partition of a key-key attention used as a bias: b_j = −α·log Σ_l κ(x_j, x_l). This is −α·LSE of key-key logits, i.e. "attention applied to attention". It cancels exact copies for every k, costs O(n²d) on at most 24 latents, and is bounded (b ≤ 0).
- **A query-conditioned version** can use a kernel on the query's own logit differences and on value similarity. That matches the Codex near-copy bound, whose error terms are logit error ε and value error η.
- **The rest only bound the boost.** Stick-breaking, DPP and finite-strength Selective-attention masking keep cluster mass bounded but growing with k. LUCID zeroes copies' values but leaves the dilution of other entries.

### Cited Findings
- The duplication identity in its published form: "adding log s performs the same operation as if you'd have s copies of the key" (ToMe). — [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)
- dedup_soft as implemented. The bias is −log(cluster size), where the cluster is the set of *fed latent input embeddings* (projection outputs) with cosine ≥ 0.95. It is one bias vector broadcast to every layer, head and query (`bias[:, None, None, :]`), applied both during the latent loop and during readout, and only when more than `budget` latents exist. — [stage1_eval.py](/Users/stevotrujillo/Desktop/DSALT/dynamic-state-anchoring/scripts/latent_gate/stage1_eval.py) (lines 81–101)
- The learned gate is `Gate`: an MLP on the final hidden state at an entry's own position, giving bias log σ(score). Its last layer is zero-initialized with bias 3.0, so g ≈ 0.95. — [gate.py](/Users/stevotrujillo/Desktop/DSALT/dynamic-state-anchoring/scripts/latent_gate/gate.py)
- On CODI, the gate "learned phase-parity kept sets" and fails with the tie bug removed (652 vs 854 on the held-out pool). On SIM-CoT it holds depth. — [project memory: heldout-scale-results](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/heldout-scale-results.md); [dsa-pivot-latent-eviction](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/dsa-pivot-latent-eviction.md)
- On CODI at 24 steps with a 6-entry budget (Stage 0): eviction by attention score gives **77**, random 157, recency 141, keep_first 205, dedup 201, plain 173. — [project memory](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/dsa-pivot-latent-eviction.md)
- The Codex near-copy bound (checked in project memory). Replace one entry by m near-copies, each corrected by −log m, with logit error ≤ ε and value error ≤ η. The output error is then ≤ tanh(ε/4)·‖v_ref − mean of the other values‖ + η, independent of m. "Dedup should bound key/value error rather than hidden-state cosine." — [project memory: codex-review-takeaways](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/codex-review-takeaways.md)
- dedup_soft *hurts* ft12: 182 vs 210 at 24 steps. — [project memory](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/dsa-pivot-latent-eviction.md)
- Mechanism facts used below:
  - Selective attention subtracts a ReLU'd cumulative mask, so it never boosts ([arXiv:2410.02703](https://arxiv.org/abs/2410.02703)).
  - Stick-breaking allocates nearest-first with "recency bias" ([arXiv:2410.17980](https://arxiv.org/abs/2410.17980)).
  - LUCID preconditions values with the inverse of a unit-diagonal causal exp-kernel Gram matrix ([arXiv:2602.10410](https://arxiv.org/abs/2602.10410)).
  - MTA convolves logits over c_k = 11 neighboring keys ([arXiv:2504.00927](https://arxiv.org/abs/2504.00927)).
  - Talking heads mixes across heads linearly ([arXiv:2003.02436](https://arxiv.org/abs/2003.02436)).
  - SSMax scales all logits by s·log n ([arXiv:2501.19399](https://arxiv.org/abs/2501.19399)).

### Inferences
**Setup (derivation).** A cluster C of k+1 exact copies has query logit z. The other entries have total exp-mass Z_o. Under plain attention the cluster mass is P(k) = (k+1)e^z / (Z_o + (k+1)e^z), so the copies act as one entry with logit z + log(k+1). "Cancels" means P(k) = P(0) for all k.

1. **Per-entry bias or temperature: cannot cancel (derivation; impossibility).**
   - **Why.** Any b_j = f(q, k_j, v_j, h_j), and any per-query temperature, gives identical copies identical values, so P(k) = (k+1)e^{z+b} / (Z_o + (k+1)e^{z+b}). Cancelling needs b = −log(k+1). That depends on how many *other* entries match, which no function of entry j alone can know.
   - **What it covers.** The MLP gate, SSMax and log-n, SSA, talking heads, DCMHA (copies identical in every head stay identical under any head mix), and query-aware top-k (Quest, NSA, MoBA select or drop copies together).
   - **What it predicts.** (a) Attention-score eviction on CODI should keep the attractor copies, and it scored 77, the worst policy. (b) A per-entry gate can only mimic dedup by exploiting non-content signals that *do* differ between copies, such as position or cycle phase. That is a plausible account of CODI's learned "phase-parity kept sets", and of why the gate works on SIM-CoT (no cycle) but not on CODI. This is an inference; the gate's features were not probed.
   - **The one exception is log-n block scaling (−log(n/6)).** It cancels only when every entry is duplicated uniformly. That explains why it is "weak".

2. **Key-key multiplicity bias, "KK-LSE": cancels exactly (derivation). This is the recommended core.**
   - **The bias.** b_j = −α·log m_j, where m_j = Σ_{l ∈ latents} κ(x_j, x_l), κ(x, x) = 1 and κ ∈ [0, 1]. For exact copies isolated from the rest, m = k+1, so with α = 1, P(k) = P(0) for every k.
   - **Distinct entries.** b_j = −log(1 + Σ_{l≠j} κ_jl) ≤ 0, a small leak. With κ = exp(β(cos − 1)), β = 14 and cos = 0.5, each pair leaks e^{−7} ≈ 10⁻³.
   - **Why "attention applied to attention".** With κ = exp(β(k̂_j·k̂_l − 1)), we get −log m_j = β − LSE_l(β·k̂_j·k̂_l). The bias is minus the log-partition of a key-to-key attention row with temperature 1/β. A second attention pass (key queries over keys) produces the bias for the first pass.
   - **Relation to dedup_soft.** dedup_soft is the special case: hard κ = 1[cos ≥ 0.95], computed on the fed input embeddings, shared across all layers and heads.
   - **Cost.** O(n²·d_head) per head per decode step, with n ≤ 24 and d_head = 64. That is about 37K multiply-adds per head, negligible against the attention itself.
   - **Stability.** The bias is bounded in [−log n, 0], so it can never boost, as in Selective attention. It is smooth in the keys when κ is soft. It should be computed in FP32 (the bf16 tie lesson) and never applied to BOS or question keys.
2b. **Joint key-value kernel (derivation from the Codex bound).** Use κ = exp(β_k(cos_k − 1) + β_v(cos_v − 1)). The bound's error terms are logit error and value error, so redundancy should be judged in (k, v) space per head. Hidden-state or input-embedding cosine is a proxy one step removed.
   - **Testable prediction.** dedup_soft's input-embedding clusters may merge ft12 latents whose deeper-layer keys or values differ. That would explain 182 vs 210. A key/value-space kernel should not hurt ft12.

3. **Query-conditioned KK, the "recursive attention" proper (derivation).**
   - **The bias.** Let s_j = q·k_j/√d be the main logits over latents. Use κ^q_jl = exp(−(s_j − s_l)²/(2σ_h²)) · exp(β_v(cos_v,jl − 1)) and b_j(q) = −α·log Σ_l κ^q_jl.
   - **What it does.** Two entries count as duplicates *for this query* only if the query cannot tell them apart (equal logits) **and** they return the same value. That is exactly the condition under which merging them is harmless per the near-copy bound. Copies cancel exactly (s_j = s_l, cos_v = 1).
   - **Extra freedom.** Entries that differ only in directions the query ignores also merge, which a purely key-space kernel would not do.
   - **Cost.** One extra [n × n] kernel per head per query, still negligible.
   - **Risk.** The bias now depends on the query's own logits, so gradients flow through s twice. Detaching s inside κ is a reasonable default; this is untested.

4. **Selective-attention-style "later similar entries mask earlier ones" (derivation).**
   - **The mask.** F_{t,j} = Σ_{i>j} a·1[similar(i, j)].
   - **What it does to copies.** With masking strength a per later copy, the cluster mass becomes ∝ e^z·Σ_{m=0}^{k} e^{−am} = e^z(1 − e^{−a(k+1)})/(1 − e^{−a}). That is bounded by e^z/(1 − e^{−a}) but is not e^z. For example, at a = log 2 it tends to 2e^z. Exact cancellation only comes as a → ∞, which keeps the newest copy (recency within the cluster).
   - **Masking by "absorption" instead.** The original uses head-0 logits, not similarity. In CODI each latent has "absorbed" its predecessor through the fed hidden state, so absorption-masking ≈ recency, and recency scored 141 on CODI. If used at all, it must be gated by similarity.
   - **Cost and stability.** Cheap and stable (ReLU, never boosts). It has no normalization guarantee.

5. **Stick-breaking or "first-match" (derivation).**
   - **Nearest-first (as published).** With k+1 copies nearest the query, the cluster takes 1 − (1−β)^{k+1}, which grows with k. Every earlier entry is multiplied by (1−β)^{k+1}. So appended copies *geometrically crush* entry 1, which is the most causally important entry in our label check (0.42–0.54 nats; [project memory](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/dsa-pivot-latent-eviction.md)).
   - **Oldest-first ordering.** This is "first match". Later copies are suppressed only as β → 1, the hard limit.
   - **Verdict.** Duplicates are not naturally suppressed in the useful direction. Converting a pretrained softmax GPT-2 also amounts to retraining attention, so it is not a cheap patch.

6. **DPP marginals as attention (derivation).** Use L = diag(e^{z/2})·S·diag(e^{z/2}). A cluster of k+1 identical items is a rank-1 block with eigenvalue λ = (k+1)e^z, so its expected inclusion is λ/(1+λ). That is bounded by 1 but still grows with k. The O(n³) cost is fine for n ≤ 24, but it replaces softmax and departs far from the pretrained model.

7. **LUCID applied to latents (derivation from its Eq. 5).** With RMS-normalized keys the kernel is unit-diagonal, and the entry between exact copies is exp(0) = 1. If the copies are far from the other entries, forward substitution gives y_copy = v − v = 0.
   - So each copy's value becomes zero, and the duplicated content is counted once in the numerator.
   - The softmax denominator still contains (k+1)e^z, so the copies' mass goes to a zero vector, every other entry is still diluted, and the output norm shrinks.
   - **Verdict.** This handles value double-counting, not dilution. It could be *combined* with KK-LSE, but it is not a substitute.

8. **MTA key-axis convolution (derivation).** A learned kernel on neighboring *logits* can penalize runs of equal logits within c_k = 11 positions. CODI's past-depth repetition is an odd-even cycle, so copies are not contiguous, and the kernel would have to learn the period. MTA also needs billions of tokens to train (5.3B tokens even for continued training) and is about 10× slower unoptimized. It is a poor fit for the budget.

9. **Ranking for this project.**
   - (1) **KK-LSE in key or key+value space**, per head, with learnable α_h and β_h. This is the direct, provable generalization of dedup_soft.
   - (2) **The query-conditioned variant (3)** as the "recursive attention" headline, if (1) holds.
   - (3) **Selective-attention-style similarity masking** as a published-mechanism baseline.
   - Stick-breaking, DPP, MTA and LUCID are related work, not candidates.

### Gaps
- None of the derivations above has been run. In particular, how much near-copies leak at CODI's real cosine levels (are the 24-step entries within cos 0.95–0.999 of earlier ones in key space, per layer?) is unknown, and it sets the β needed.
- I found no published analysis of attention mechanisms under exact key duplication, apart from ToMe's one-line argument. The cancellation claims are my own algebra.

---

## Q4. Implementation in a pretrained GPT-2 CODI checkpoint: where it hooks in, parameter count, identity init, training signal, and cheap training at 12–24 steps

### Takeaway
KK-LSE fits inside the existing `_attn` patch, because `_attn` receives the full cached `key` and `value` tensors; it needs only the latent slice indices. Everything else is cheap:
- **Size.** About 288–576 scalar parameters (α, β per head; optionally σ and β_v), or about 150K with a rank-16 key projection.
- **Identity init.** Initialize α = 0. Alternatively, warm-start at α = 1 with β ≈ 14, which mimics dedup_soft's 0.95 threshold.
- **Training.** Calibrating these scalars on a frozen model at T ∈ {12, 24} is minutes of compute, but by construction it can at best "hold" (Q6). A gain test needs the latent generator trained *with* the bias on: the Stage 1 ft12 recipe plus KK-LSE, at roughly Stage 1 cost.
- **The LLaMA port** must also fix the RoPE answer-distance cost, which dedup does not touch.

### Cited Findings
- The repo pins `transformers==4.46.2` ([requirements.txt](/Users/stevotrujillo/Desktop/DSALT/dynamic-state-anchoring/requirements.txt)). The patch wraps `GPT2Attention._attn(self, query, key, value, attention_mask, head_mask)`, adds `extra_bias` to `attention_mask`, and asserts eager attention. — [gate.py](/Users/stevotrujillo/Desktop/DSALT/dynamic-state-anchoring/scripts/latent_gate/gate.py)
- Latent entry t sits at cache index `question + t`, where `<bot>` is at `question`. The bias is applied in both the latent loop (`run`) and the readout (`readout`). — [stage1_eval.py](/Users/stevotrujillo/Desktop/DSALT/dynamic-state-anchoring/scripts/latent_gate/stage1_eval.py)
- The hard-gate tie bug: zero-initialized scores sat at exactly 3.0, and bf16 spacing near 3.0 is about 0.016. The fix is a stable argsort plus FP32 gate computation outside autocast. — [project memory: codex-review-takeaways](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/codex-review-takeaways.md)
- MTA reports that identity initialization beats zero or constant init (+0.02 / +0.08 PPL), and Selective attention's constraints (never boost, never mask BOS or self) are the published templates for safe logit biases. — [arXiv:2504.00927](https://arxiv.org/abs/2504.00927); [arXiv:2410.02703](https://arxiv.org/abs/2410.02703)
- Gate-only training on a frozen LLM is a validated recipe: TRIM-KV trains "only gate fine-tuning" via distillation from the frozen LLM plus a capacity loss. — [arXiv:2512.03324](https://arxiv.org/abs/2512.03324). Dynamic Context Pruning fine-tuned pairwise drop scores on GPT-2. — [arXiv:2305.15805](https://arxiv.org/abs/2305.15805)
- At LLaMA-1B, most of the depth loss is RoPE answer distance, not dilution. dedup alone fails at 1B, while `dedup_anchor` holds (CODI-1B 1055 vs plain@6 1077 and plain@24 1027). A full fine-tune at 1B *hurt* (ft6@6 246 vs base 277), because training CE was already flat at 0.04–0.11. — [project memory: heldout-scale-results](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/heldout-scale-results.md)
- On GPT-2 CODI, ft12 holds depth (210 at 12, 203 at 24) but does not beat ft6 (209 at 6). On SIM-CoT, no 12-step arm beats ft6's 215. Single-seed noise is about 10 per 500. — [project memory](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/dsa-pivot-latent-eviction.md)
- PCCoT found 12 latents "much better than using 6" when training GPT-2 continuous CoT from scratch. — [project memory: codex-review-takeaways](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/codex-review-takeaways.md); [arXiv:2506.18582](https://arxiv.org/abs/2506.18582)

### Inferences
**Hook.** This is a sketch to adapt in a new script; it is not an edit to the repo files. It goes inside the existing `_attn` wrapper, before calling `original`:

```python
sl = getattr(self, "latent_slice", None)          # (start, stop) cache indices of latent entries, set per forward
if sl is not None and sl[1] - sl[0] > 1:
    k = key[..., sl[0]:sl[1], :].float()           # [B, H, n, 64], FP32 outside autocast
    kh = torch.nn.functional.normalize(k, dim=-1)
    cos = kh @ kh.transpose(-1, -2)                # [B, H, n, n] key-key "attention" logits
    kern = torch.exp(self.kk_beta.view(1, -1, 1, 1) * (cos - 1))   # 1 on the diagonal
    # optional value term: * torch.exp(self.kk_beta_v.view(1,-1,1,1) * (cos_v - 1))
    m = kern.sum(-1)                               # effective multiplicity >= 1
    b = -self.kk_alpha.view(1, -1, 1) * m.log()    # alpha = 0 -> exact identity with the base model
    extra = query.new_zeros(query.shape[0], query.shape[1], 1, key.shape[-2])
    extra[..., sl[0]:sl[1]] = b[:, :, None, :].to(query.dtype)
    attention_mask = extra if attention_mask is None else attention_mask + extra
```

- **Causality is automatic.** At each decode step `key` holds exactly the entries visible to the current query, so the multiplicities are recomputed as copies arrive. Nothing is baked into the cache.
- **Where it applies, layers.** Default to all 12 layers. dedup_soft, which holds on both GPT-2 models, applies to all layers. The per-head α can learn to switch layers off.
- **Where it applies, queries.** Run three training-free conditions:
  - readout-only (answer-token queries);
  - loop-only (latent-step queries);
  - both, which is what dedup_soft does now.
  
  This separates "fixing the readout" from "changing the latent trajectory".
- **Parameter count.**
  - (α, β) per head: 2 × 12 × 12 = 288.
  - Adding β_v and σ for the query-conditioned variant: 576.
  - A rank-16 learned key projection for the kernel: 12 × 12 × 64 × 16 ≈ 147K.
  - All are negligible next to 124M.
- **Initialization.** Two choices:
  - α = 0: exact identity with the base checkpoint, the "zero bias at start" requirement.
  - α = 1 with β ≈ 14: a warm start near dedup_soft, since exp(14 × (0.95 − 1)) ≈ 0.5, about the 0.95 threshold. This starts training from the known "hold" solution instead of from the drift. My recommendation for any gain-seeking run.
  - Keep α ≥ 0 via softplus so the bias can never boost.
- **Training signal.**
  - **(a) Scalar calibration on a frozen model.** Answer-only CE at T sampled from {12, 16, 24}, optionally with KL to the model's own 6-step answer distribution. This is a "distill from the 6-step self" objective, the analogue of TRIM-KV's distill-from-frozen. It can only recover the 6-step behaviour.
  - **(b) Gain-seeking fine-tune.** Use the CODI loss (teacher CE + student CE + hidden-state distillation), or SIM-CoT's per-step auxiliary decoder, with KK-LSE active in training. Use a randomized latent count T ∈ {6..16} so the model learns that extra steps are free to use.
  - Answer-only CE alone is probably too weak, given that Stage 1 at GPT-2 and Part C at 1B showed no gain from it.
- **Cheap 12–24-step training.**
  - Cache the question prefix once per batch.
  - Backpropagate through at most 12 latent steps, using a stop-gradient on the cache entries from earlier steps when T > 12 (truncated BPTT).
  - Evaluate at 24 only.
  - Compute the kernel in FP32 and keep the rest in bf16.
  - The FLOP overhead of KK-LSE is negligible, so cost equals the matching Stage 1 arm (ft12). I do not have the Stage 1 GPU-hour figure, so I cannot quote it.
- **Order of spending (cheapest first).**
  - (i) Training-free KK-LSE vs dedup_soft at T ∈ {6, 12, 24} on the 500-question validation split for CODI, SIM-CoT and **ft12**. ft12 is the diagnostic case where dedup_soft hurt. This is eval only.
  - (ii) Scalar calibration: minutes.
  - (iii) A single ft12 + KK-LSE fine-tune, run only if (i) shows KK-LSE ≥ dedup_soft and does not hurt ft12.
- **The LLaMA-3.2-1B port.** It needs a new hook (no `_attn`). One option is to patch `LlamaAttention.forward` (eager) or pass a per-head float mask to SDPA. With GQA, compute the kernel on the KV heads and broadcast to the query groups. KK-LSE does not address RoPE answer distance, so it must ship with the existing `anchor` (position re-anchoring), i.e. "KK-LSE + anchor" vs `dedup_anchor`.

### Gaps
- I did not verify LLaMA-3.2-1B's head and KV-head counts or the exact LLaMA eager-attention code path in transformers 4.46.2; those details above come from memory.
- I do not know the Stage 1 Colab GPU-hours per arm, so I cannot cost step (iii) in credits.
- The leakage threshold (β) cannot be set from the literature. It needs the per-layer key-cosine distribution of 24-step CODI latents, which is unmeasured.

---

## Q5. Can it be done training-free, as a parameter-free key-key attention, and how would that differ from dedup_soft?

### Takeaway
Yes. Training-free KK-LSE is the same "−log(cluster size)" idea, but it differs in five ways:
1. It is computed **per layer and per head**.
2. It uses **keys, optionally keys plus values**, which is what the attention actually reads, instead of the fed input embeddings.
3. It uses a **soft kernel** instead of a hard 0.95 threshold, so near-copies get graded correction and it is differentiable.
4. It **can be query-conditioned** at no parameter cost.
5. It pairs naturally with LUCID-style value de-duplication.

Its prior art is R-KV's key-cosine redundancy (used for eviction), ToMe's proportional attention (the inverse operation) and LUCID (a key-key kernel used differently). The main measurable prediction is that it should not hurt ft12, where dedup_soft does (182 vs 210).

### Cited Findings
- dedup_soft computes cosine on `fed` latent input embeddings, applies one bias to all layers and heads, and uses a hard ≥ 0.95 threshold. — [stage1_eval.py](/Users/stevotrujillo/Desktop/DSALT/dynamic-state-anchoring/scripts/latent_gate/stage1_eval.py)
- dedup_soft holds the 6-step score at 24 steps: CODI 201 vs 173 plain, SIM-CoT 207 vs 170. It holds on both GPT-2 models on the held-out pool, fails alone on both 1B models, and holds at 1B only with `anchor`. — [project memory: dsa-pivot-latent-eviction](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/dsa-pivot-latent-eviction.md); [heldout-scale-results](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/heldout-scale-results.md)
- R-KV, a training-free published analogue, scores redundancy by key cosine and evicts. — [arXiv:2505.24133](https://arxiv.org/abs/2505.24133). KeyDiff, also training-free, evicts by key similarity. — [arXiv:2504.15364](https://arxiv.org/abs/2504.15364)
- ToMe's proportional attention is used off-the-shelf, training-free, on supervised ViTs, where it is "necessary". — [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)
- LUCID is parameter-free in its preconditioner apart from the fixed √d scaling. It was trained into models, not applied post hoc. — [arXiv:2602.10410](https://arxiv.org/abs/2602.10410)
- SSMax can be swapped in after pretraining with partial benefit, but it rescales all logits uniformly. — [arXiv:2501.19399](https://arxiv.org/abs/2501.19399)

### Inferences
- **Parameter-free recipe.** Use KK-LSE (Q3 item 2) with α = 1 and a single global β, which is the one hyperparameter. Use per-head key cosine at every layer, optionally multiplied by a value-cosine kernel. Fix β on 100 training questions at T = 24, not on validation.
- **Variant ladder** for a clean ablation table, each step changing one thing:
  1. dedup_soft (embedding-space, hard, shared).
  2. Hard threshold on per-head key cosine.
  3. Soft kernel on key cosine.
  4. Soft kernel on key + value cosine.
  5. The query-conditioned logit-difference kernel.
  
  This isolates which feature matters. It is all eval-only, run on the existing Stage 1 eval loop.
- **How it differs in behaviour, as predictions to check.**
  - (a) On base CODI and SIM-CoT at 24 steps it should match dedup_soft (about 201–207), since exact or near-exact copies dominate there.
  - (b) On ft12 it should *not* drop to 182, if ft12's later latents are similar in input embedding but distinct in deeper-layer keys and values.
  - (c) At T ≤ 6 it should be inert. With a soft kernel it is not exactly inert, so measure the 6-step leakage cost explicitly; it must be ≤ 1–2 answers.
- **Training-free is the right first experiment but the wrong headline.** Even a perfect training-free dedup reproduces the 6-step multiset of information. It is a stronger version of the existing "hold" result, not an accuracy gain (Q6).

### Gaps
- Whether per-head key-space clusters at 24 steps match the embedding-space clusters is unmeasured; it is the cheapest diagnostic to run first.
- I found no published post-hoc (training-free) use of a soft key-key −log multiplicity bias in LLM attention. The search was limited to arXiv API keywords and the web.

---

## Q6. What result would be most convincing for a paper, what are the main ways it could fail, and could it only ever "hold" accuracy?

### Takeaway
**Any readout-side selection method, recursive attention included, is capped at "hold" when the extra latents carry no new information.** That is exactly what the project's evidence shows:
- keep_first@24 ≈ plain@6;
- no budgeted policy, gate or depth exceeds 6 steps;
- an oracle "any of 6/12/24 correct" adds only 9–15 of 500.

A gain must come from making steps 7–24 *compute something new*: training the generator with the bias on, more compute per step, or step-level supervision beyond 6. Recursive attention only removes the penalty for redundancy.

**The most convincing paper result**, all at paired significance on held-out sets:
- a CODI or SIM-CoT GPT-2 model trained with KK or recursive attention at T > 6 that beats the best of {CODI@6, ft6@6, ft12@12, dedup_soft@24};
- accuracy non-decreasing from 6 to 24 steps;
- the gain vanishing when the bias is removed at test time;
- the gain transferring to LLaMA-1B with `anchor`.

### Cited Findings
- **The project's ceiling evidence.**
  - Stage 0 (CODI): keep_first@24 205 vs plain@6 206. "The budget restores the 6-step score and nothing beats it."
  - Depth voting: "taking a question as correct if any of the three depths gets it right adds only 9–15 answers".
  - Stage 1: "the gates mostly keep entries 1–6"; ft12 210@12 vs ft6 209@6 (CODI); no 12-step arm beats ft6's 215 (SIM-CoT).
  - Held-out: "G, a free gain, fails everywhere", and at 1B "no depth beats 6 steps".
  
  — [project memory: dsa-pivot-latent-eviction](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/dsa-pivot-latent-eviction.md); [heldout-scale-results](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/heldout-scale-results.md)
- "Every published past-depth stabilization only preserved peak accuracy" (the project's literature verdict). — [project memory](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/dsa-pivot-latent-eviction.md)
- **External evidence that a gain from bounded or selected attention is possible on reasoning, but on *explicit* tokens:**
  - R-KV reaches 105% of full-cache accuracy at 16% of the cache — [arXiv:2505.24133](https://arxiv.org/abs/2505.24133).
  - TRIM-KV scores 75.8 vs a full-cache 65.5 on AIME24 — [arXiv:2512.03324](https://arxiv.org/abs/2512.03324).
  - Think Clearly "significantly improves" accuracy by pruning redundant reasoning tokens — [arXiv:2507.08806](https://arxiv.org/abs/2507.08806).
  
  In those settings the pruned tokens are model-generated *distractors*, with "repetitive content often receiv[ing] disproportionately high attention" per R-KV. They are not information-free copies of useful entries.
- **External evidence that more latent depth can raise accuracy when trained for it:**
  - PCCoT: 12 > 6 latents, from scratch — [arXiv:2506.18582](https://arxiv.org/abs/2506.18582).
  - Dreamer: depth attention gives +3.8 average at depth 32 (DR+DA 51.4 vs DR 47.6). At depth 16 it is −2.3 (43.4 vs 45.7), so the gain is mixed — [arXiv:2601.21582](https://arxiv.org/abs/2601.21582).
  - InfiLoop: learned decay over loop states doubles Sudoku accuracy vs no decay — [arXiv:2610.11570](https://arxiv.org/abs/2610.11570).
- SIM-CoT attributes collapse at larger latent counts to "latent representations becoming homogeneous", caused by "insufficient step-level supervision". — [arXiv:2509.20317](https://arxiv.org/abs/2509.20317)
- **Training-free looping of layers** degrades naively, but damped sub-steps give small gains (+1–2.6 pp) on Qwen3 and Moonlight. — [arXiv:2605.23872](https://arxiv.org/abs/2605.23872). **Adaptive per-layer looping** "primarily benefits mathematical reasoning". — [arXiv:2603.08391](https://arxiv.org/abs/2603.08391)

### Inferences
- **Why "hold" is the ceiling for readout-side methods (argument).** Recursive attention changes only *how* the readout weighs existing entries. If entries 7–24 are near-copies of 1–6 (CODI's attractor and cycle; "later latents compute nothing new"), then the de-duplicated 24-step multiset ≈ the 6-step set. The readout then sees the same information it saw at 6 steps, and the expected score is the 6-step score.
  - **One loophole, testable now.** Trained recursive attention might improve the readout *at T = 6* by re-weighting entries 1–6 or question tokens. Train the scalars at T = 6 and check. If even that gives nothing, the readout is not the bottleneck.
- **Three routes by which the combination could gain, in order of plausibility:**
  1. **Dedup during training as a collapse stabilizer.** Train ft12 or ft16 with KK-LSE active and SIM-CoT-style step supervision. The hypothesis: if part of SIM-CoT's large-T collapse is homogeneous latents diluting each other, the bias lets you train at T = 12–16 without collapse, and you could reproduce PCCoT's 12 > 6 inside the CODI recipe.
  2. **More compute per latent step** (loop a mid-block r times inside each step, damped as in 2605.23872), with KK-LSE handling any extra repetition. The gain would come from the inner loop, and recursive attention only keeps the extra steps harmless. The paper must then credit the loop, not the attention.
  3. **Query-conditioned selection** helping problems with more than 6 rationale steps. Score the 4+-step subset separately, since that is where extra latents could matter.
- **Most convincing design** (pre-registered, matching the project's style):
  - **Primary.** ft12 + KK-LSE (training and test) vs ft12 plain, ft6, and base + dedup_soft@24, on the held-out pool (n ≈ 2,818) with paired McNemar or bootstrap CIs.
  - **Bar.** Beat max(ft6@6, ft12@12) by at least 2 points with a CI excluding 0. The 500-question validation split is too small: noise is about 10 answers.
  - **Mechanism checks.**
    - (a) The gain disappears when KK-LSE is removed at test time, while dedup_soft at test does not recover it.
    - (b) Latent diversity (effective rank or logit-lens distinct decodes over steps 1–12) is higher than in ft12 plain.
    - (c) The copy test: k exact copies of entry 6 cost 0 answers, by construction.
  - **Generality.** SIM-CoT GPT-2, then LLaMA-1B with `anchor`.
- **Main failure modes.**
  1. **Information ceiling.** Hold only. This is the most likely outcome, given every selection result so far.
  2. **Dedup makes collapse harmless, so it gets entrenched.** With repetition free, the generator has no pressure to make later steps distinct, and the fine-tune converges to "repeat after 6". Watch latent diversity during training. A diversity or progressive-prediction auxiliary may be needed, and then the gain belongs to the auxiliary.
  3. **Over-merging** of distinct-but-similar entries. This happened with dedup_soft on ft12 (−28 answers). A loose β or an embedding-space kernel causes it.
  4. **Position at scale.** At 1B, RoPE answer distance dominates, and no key-key bias fixes it. Without `anchor` the LLaMA port will fail as dedup alone did.
  5. **Leakage at T ≤ 6.** A soft kernel slightly changes the trained regime, and the 6-step score could drop by a few answers.
  6. **Novelty.** Mechanism-level prior art is dense: ToMe proportional attention, R-KV, LUCID, Selective attention, InfiLoop. Without a gain, the contribution reduces to the diagnosis plus a cleaner hold, which the project already has with dedup_soft.
  7. **Statistics.** Single-seed noise of about 10/500 swamps a 1–2 point effect. Use held-out pools and 2+ seeds for the final claim.
- **Where it sits in the hierarchy.** A gain from dedup in training would *test* the user's thesis: dilution is not just a test-time artifact but a training-time obstacle to using more latents. If ft12 + KK-LSE ≈ ft12, the thesis is answered negatively, cheaply, and the brainstorm should pivot to generator-side methods (inner looping, non-appending refinement as in PCCoT/LLoCoT, step supervision past 6).

### Gaps
- I found no published case where a selection or re-weighting method over *latent* (not explicit) reasoning entries raised accuracy above the model's best trained depth, so there is no external precedent for the gain claim.
- Whether SIM-CoT's training-time collapse is mediated by attention dilution (which KK-LSE would fix) or by information loss (which it would not) is unknown. SIM-CoT attributes it to supervision, not attention.
- I have no compute estimate for the ft12 + KK-LSE run in Colab credits. Stage 1 per-arm GPU-hours are not in the memory notes I read.
