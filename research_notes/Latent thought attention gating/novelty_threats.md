# Novelty threats: learned keep/drop gating of past latent-thought KV entries ("Dynamic State Anchoring")

Idea under test: in CODI / Coconut / SIM-CoT style latent CoT (continuous thoughts fed back as input embeddings and appended to the KV cache), a learned gate decides which past latent-thought KV entries to keep or drop as the loop runs past its trained depth (6 -> 12/24). The goal is to stop attention dilution so that extra steps can add reasoning. The gate is trained with verifiable intermediate-result labels: each latent is decoded with the logit lens and matched to gold GSM8K-Aug results such as "<<4-2=2>>".

Search date: 2026-11-02. About 35 search/fetch calls across arXiv, alphaxiv, OpenReview/ICLR proceedings, Hugging Face, LessWrong and aggregators. Many 2026 papers were checked only through their abstract pages. Where that is the case, the finding says so.

## Q0 (objective): Closest competitors overall, ranked

### Takeaway
I found no direct competitor. No paper I found combines all four parts: (a) Coconut/CODI-style appended continuous thoughts, (b) a learned keep/drop gate over past latent KV entries, (c) running past the trained latent depth, and (d) gate labels from decoded intermediate results. The nearest threats each cover one or two parts:
- AGCLR (2606.07720): learned gates inside a Coconut loop.
- TRIM-KV (ICLR 2026): learned per-head keep/drop gate on KV entries, for explicit reasoning tokens.
- Selective Attention (ICLR 2025): learned masking and pruning of past tokens.
- Huginn's zero-shot KV sharing: a bounded cache across recurrences, with a fixed policy.
- PENCIL: learned erasure of explicit thoughts.
- Think Shallow, Solve Deep (2608.18222) and Kuo et al. (2606.29983): rival explanations of, and fixes for, depth degradation.

### Cited Findings
Ranked by closeness. "Partial overlap" means a reviewer would cite it as the closest prior. "Adjacent" means related but different in mechanism or setting.

1. **AGCLR**: Farhan and Chaudhary, "Why Limit the Residual Stream to Layers and Not Tokens? Persistent Memory for Continuous Latent Reasoning", arXiv 2606.07720v1, submitted 5 Jun 2026 (**2026 preprint**). **Partial overlap. This is the closest paper inside the latent-CoT literature.**
   - It adds read, forget and write gates to CoCoNuT on GPT-2. The gates act on a separate persistent memory vector c_t, not on the KV cache: h'_t = (1−f_t)⊙h_t + r_t⊙c_{t−1}, then c_t = LayerNorm(c_{t−1} + w_t⊙h'_t). The gated h'_t becomes the next input embedding. — [arXiv html](https://arxiv.org/html/2606.07720)
   - Each gate is a sigmoid of a linear projection of LayerNorm(h_t). The gates get no supervision of their own ("learned end-to-end", zero-initialized). The paper does not discuss masking or dropping past latent tokens from attention. — [arXiv html](https://arxiv.org/html/2606.07720)
   - It uses 0 to 6 latent tokens across curriculum stages and runs nothing past the trained count. Attention dilution is not discussed. GSM8K: AGCLR 34.0 vs CoCoNuT 31.4 vs CoT 40.6. — [arXiv html](https://arxiv.org/html/2606.07720)
   - Overlap: learned gating in the latent loop, GPT-2, GSM8K. Not overlapping: no KV-entry selection, no intermediate-result labels, no past-depth regime, no CODI/SIM-CoT.

2. **TRIM-KV**: "Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs", arXiv 2512.03324, ICLR 2026. **Partial overlap. This is the closest paper on mechanism.**
   - Each attention head of a frozen pretrained LLM gets a learned retention gate. The gate predicts how long a token should stay cached, and its score also modulates attention. When the cache is over budget, the lowest-scoring token is dropped. Only the gates are fine-tuned, with distillation and capacity losses. — [ICLR 2026 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/c3887741ee4e655494770be95303383f-Paper-Conference.pdf); [alphaxiv overview](https://www.alphaxiv.org/overview/2512.03324v1)
   - Reported: up to 198% relative improvement over heuristic eviction baselines on math benchmarks, often beating the full cache. Gates trained on math data transfer to non-math procedural tasks. — [ICLR 2026 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/c3887741ee4e655494770be95303383f-Paper-Conference.pdf) (via a search summary; numbers not checked in the full text)
   - Overlap: a learned keep/drop gate on KV entries under a budget, which can beat the full cache on reasoning. Not overlapping: explicit text tokens, not continuous thoughts. Labels come from distillation and capacity losses, not verifiable intermediate results. No latent-depth extrapolation.

3. **Selective Attention**: Leviathan, Kalman and Matias, "Selective Attention Improves Transformer", arXiv 2410.02703, ICLR 2025. **Partial overlap, on mechanism only.**
   - Tokens build a learned soft mask that is subtracted from attention logits (softmax(QKᵀ/√d − F)V), so a token can reduce later attention to earlier tokens. Tokens that are masked enough can be evicted from the context buffer under a fixed per-layer budget. An auxiliary memory loss can encourage pruning. — [arXiv PDF](https://arxiv.org/pdf/2410.02703); [arXiv html v2](https://arxiv.org/html/2410.02703v2)
   - Reported: 16x, 25x and 47x less attention memory at context sizes 512, 1,024 and 2,048 (v2 abstract). A third-party summary gives other figures for another setup. — [arXiv html v2](https://arxiv.org/html/2410.02703v2); [liner summary](https://liner.com/review/selective-attention-improves-transformer)
   - Not overlapping: language modeling on C4, no latent CoT, no intermediate-result supervision.

4. **Huginn recurrent-depth model**: Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach", arXiv 2502.05171. **Adjacent to partial.**
   - In "Zero-Shot KV-cache Sharing" (Sec. 6.2), each token gets a fixed KV budget k for its recurrence. Iteration i reads and writes cache entry i mod k, so step 17 overwrites step 1 when k=16. The authors report "minimal impact to performance". On MTBench, k=4 "does not reduce performance". — [arXiv html](https://arxiv.org/html/2502.05171)
   - Training samples the recurrence count at random (mean 32) so the model works when recurrence is scaled up at test time. Evaluating at r=64 further improves GSM8K. — [arXiv html](https://arxiv.org/html/2502.05171)
   - Overlap: a bounded KV memory across recurrent steps, and more steps at test time than the training mean. Not overlapping: depth recurrence inside one token rather than appended latent tokens, a fixed circular overwrite rather than a learned gate, no labels.

5. **PENCIL**: Yang, Srebro, McAllester and Li, "PENCIL: Long Thoughts with Short Memory", arXiv 2503.14337, ICML 2025. **Partial overlap, on the idea only.**
   - Thought generation alternates with a learned reduction step that erases intermediate thoughts that are no longer needed, so the model reasons longer with a bounded context. Tested on SAT, QBF and Einstein's puzzle. — [arXiv html](https://arxiv.org/html/2503.14337v2); [PMLR](https://proceedings.mlr.press/v267/yang25ac.html)
   - Not overlapping: explicit tokens with reduction-rule traces, not continuous thoughts, and no attention-dilution framing.

6. **Think Shallow, Solve Deep**: Viakhirev et al., arXiv 2608.18222, submitted 18 Aug 2026 (**2026 preprint**). **Partial overlap. It competes on the explanation, not the mechanism.**
   - It sorts trained recurrent-depth operators into settling, marginal and drifting regimes. Extra iterations "can improve, preserve, or degrade an answer". "Settling operators do not degrade with added depth" and sometimes improve past the training horizon (Sudoku 0.19 to 0.34). A terminal fixed-point objective moves an operator into the settling regime. The mechanism is dynamical stability (a contraction-style margin condition), not attention. It uses Huginn-3.5B as a check and does not mention Coconut, CODI or latent CoT. — [arXiv abs](https://arxiv.org/abs/2608.18222)

7. **Kuo et al.**: arXiv 2606.29983, NeurIPS 2026 (**2026**). **Partial overlap, as a baseline and rival fix.** Randomizing loop counts during training stabilizes predictions across inference depths, on algorithmic tasks only. This comes from the project's earlier reading, not re-fetched this session. — [arXiv PDF](https://arxiv.org/pdf/2606.29983)

8. **KaVa**: Kuzina et al., "KaVa: Latent Reasoning via Compressed KV-Cache Distillation", arXiv 2510.02312, ICLR 2026. **Adjacent to partial.** A latent-reasoning student (continuous thoughts, as in Coconut) is trained to match a teacher's KV cache from explicit CoT. A redundancy- and importance-aware eviction module first compresses that cache to the latent budget. Eviction is only a way to build the supervision target. The student's own latent cache is never gated at run time. — [arXiv html](https://arxiv.org/html/2510.02312v2); [OpenReview PDF](https://openreview.net/pdf/95777660f1bc7a3933327d8196c548e2164e809d.pdf)

9. **Other adjacent work (one line each)**
   - **MoR**, Mixture-of-Recursions, arXiv 2507.10524, NeurIPS 2025: a learned token router sets each token's recursion depth. Recursion-wise caching stores KV only for tokens active at each depth, and a variant shares the first recursion's KV across later ones. — [arXiv html](https://arxiv.org/html/2507.10524v3)
   - **T-LoopFormer**, arXiv 2609.15160, 14 Sep 2026 (**2026**): token-choice routing over loop counts. Each loop keeps its own KV cache of active tokens, and exited tokens' slots are masked. — [arXiv abs](https://arxiv.org/abs/2609.15160)
   - **MarCos**, arXiv 2509.25020: a Markov chain of continuous thoughts, where each thought depends on the previous one. Reported +4.7% over token CoT on GSM8K. Relevant as the extreme "keep only the last latent" structural baseline. — [arXiv html](https://arxiv.org/html/2509.25020v1)
   - **Deliberation in Latent Space**, arXiv 2412.17747: latent cache-augmentation embeddings are masked from attending to other latent embeddings. — [arXiv PDF](https://arxiv.org/pdf/2412.17747)

### Inferences
- The novel combination appears to be: a gate over past continuous-thought KV entries + verifiable intermediate-result labels from logit-lens decoding + the past-trained-depth regime + the attention-dilution motivation. Each part has neighbours, but I found no paper with two or more of them in the latent-CoT setting.
- The strongest reviewer objection will likely be "TRIM-KV or Selective Attention applied to latent tokens". The defence has to rest on the latent-specific parts: verifiable decoded labels, and a gain past trained depth.
- The strongest objection to the motivation will be "Think Shallow, Solve Deep" and Kuo et al. Both explain and fix depth degradation without attention selection. Kuo-style random latent counts and a fixed-point or settling objective should be baselines.
- In this project's own earlier measurements (memory notes, not external), SIM-CoT settles yet drifts as much as CODI, and copying KV entries alone reproduces the drift. That is evidence for a reweighting explanation over the stability explanation in 2608.18222. It is a differentiator, provided it replicates.

### Gaps
- Most 2026 papers (Dreamer, RiM, Abstract Token Curriculum, T-LoopFormer, 2608.18222) were read only at the abstract level. Their full texts might contain past-depth or attention-selection experiments that I did not see.
- I did not search OpenReview directly for ICLR 2026 or ICML 2026 submissions, and NeurIPS 2026 submissions are not public. Some unindexed workshop paper could be closer.

## Q1: Pruning, evicting, merging or compressing continuous thought tokens or latent reasoning steps

### Takeaway
Compression in latent CoT means making the chain shorter at training time (CoLaR, SLT, CCoT-style), choosing a length (adaptive or early-exit methods), or distilling from an evicted teacher cache (KaVa). I found no paper that evicts or merges the KV entries of already-generated continuous thoughts during the latent loop. Learned eviction exists only for explicit reasoning tokens or long context: TRIM-KV, IndexMem, ThinKV, Lethe, LazyEviction, Think Clearly, LightThinker, AbridgeKV.

### Cited Findings
- **CoLaR**: Tan et al., "Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains", arXiv 2505.16552. Embeddings of consecutive reasoning tokens are merged by a compression factor c, sampled at random during SFT. A test-time factor (e.g. CoLaR-5) is reported as a 14.1% improvement over baselines (the snippet cuts off before the metric). **Adjacent.** It merges target tokens into fewer latents. It does not select past latent KV entries. — [arXiv PDF](https://arxiv.org/pdf/2505.16552)
- **Selective Latent Thinking (SLT)**: arXiv 2605.25745, May 2026 (**2026**). It anticipates a short explicit trajectory, then compresses only the accepted span into a latent. **Adjacent.** — [arXiv PDF](https://arxiv.org/pdf/2605.25745)
- **KaVa** (see Q0 #8): eviction is applied to the teacher's explicit-CoT cache to make the latent target. — [arXiv html](https://arxiv.org/html/2510.02312v2)
- **AbridgeKV**: arXiv 2610.20641, October 2026 (**2026, very recent**). Long reasoning is compressed into the KV cache of learned pause tokens, in fixed 512-token blocks. **Adjacent.** It works on explicit reasoning. — [arXiv html](https://arxiv.org/html/2610.20641)
- **IndexMem**: arXiv 2605.25475, May 2026 (**2026**). A learned indexer decides which KV entries to keep, and a fixed-size latent memory absorbs evicted entries and adds a residual readout back to attention. It is trained end-to-end through continual training. It notes that "memory-as-tokens" variants can collapse toward a generic average. **Adjacent.** It targets long-context inference. — [arXiv html](https://arxiv.org/html/2605.25475v1)
- **TRIM-KV** (see Q0 #2): learned retention gate, lowest-score eviction, beats full cache on math. — [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/c3887741ee4e655494770be95303383f-Paper-Conference.pdf)
- **Think Clearly / Redundant Token Pruning**: EMNLP Findings 2025. It scores the importance of reasoning tokens and evicts redundant ones from the KV cache during generation. The authors report that pruning can also improve reasoning accuracy, not just compress. **Adjacent.** It works on explicit CoT. Its scoring uses self-summarization. — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1169.pdf)
- **ThinKV**, arXiv 2510.01290: classifies reasoning segments by thought type and gives each its own quantization and eviction policy. Reported near-lossless accuracy at under 5% of the original KV cache. **Adjacent.** — [alphaxiv](https://www.alphaxiv.org/abs/2510.01290)
- **Lethe**, arXiv 2511.06029: layer- and time-adaptive KV pruning for reasoning-intensive generation. **Adjacent.** — [arXiv PDF](https://www.arxiv.org/pdf/2511.06029)
- **LazyEviction**, arXiv 2506.15969: delays eviction because reasoning tokens regain attention after several steps. **Adjacent**, and relevant because decoded labels could flag latents that are needed again later. — [arXiv abs](https://arxiv.org/abs/2506.15969)
- **Markovian Thinker / Delethink**, arXiv 2510.06557, ICLR 2026. Reasoning is cut into fixed-size chunks and only a bounded carry-over is kept, so context stays constant. The method is RL-trained, and the authors report it is weaker on tasks that need persistent large memory. **Adjacent.** It works on explicit text. — [arXiv html](https://arxiv.org/html/2510.06557v3); [ICLR 2026 poster](https://www.iclr.cc/virtual/2026/poster/10011666)
- **PENCIL** (see Q0 #5): learned erasure of explicit thoughts. — [arXiv html](https://arxiv.org/html/2503.14337v2)
- **InftyThink**: iterative reasoning with intermediate summarization. It appeared only in passing in search results and was not fetched. **Adjacent.** — [mention in search result for PENCIL](https://arxiv.org/html/2503.14337v2)
- **Adaptive-length and early-exit latent reasoning** stops or chooses length. It does not select states.
  - FR-Ponder: per-step choice between emitting a token and applying a learned thought step. — [arXiv html](https://arxiv.org/html/2509.24238v1)
  - AdaPonderLM: gated pondering with token-wise adaptive depth, arXiv 2603.01914 (**2026**). — [arXiv PDF](https://arxiv.org/pdf/2603.01914)
  - Early-exit latent reasoning in VLA models, arXiv 2606.15099 (**2026**). — [arXiv html](https://arxiv.org/html/2606.15099)
  - LaST-R1, RL-chosen latent length, arXiv 2604.28192 (**2026**). — [arXiv PDF](https://arxiv.org/pdf/2604.28192)
  - LaTER: neither too few nor too many latent steps are desirable, arXiv 2605.07315 (**2026**). — [arXiv html](https://arxiv.org/html/2605.07315v1)
- **Reasoning in Memory (RiM)**: Aichberger and Hochreiter, arXiv 2605.30343, 28 May 2026 (**2026**). Fixed memory blocks of special tokens act as a latent workspace in one forward pass, with a two-stage curriculum. The abstract mentions no gating, eviction or selection. **Adjacent.** — [arXiv abs](https://arxiv.org/abs/2605.30343)
- **Dropout-GRPO**, arXiv 2606.10184, June 2026 (**2026**). One Bernoulli mask, held fixed across latent steps per rollout, adds variance so GRPO works on Coconut (GSM8K 27.29 to 29.01). This is noise for exploration, not selection of latent states. **Adjacent.** — [arXiv html](https://arxiv.org/html/2606.10184)

### Inferences
- "Latent thought eviction" and "latent token merging" over already-generated continuous thoughts look unclaimed. The explicit-CoT eviction papers (Think Clearly, TRIM-KV) set a precedent that eviction can raise reasoning accuracy, which supports the premise. They also make "apply KV eviction to latent tokens" an obvious step that reviewers will press on.
- Heuristic and attention-score eviction policies (H2O-style, LazyEviction, Lethe, Think Clearly) are natural Stage-0 baselines alongside recency and dedup.

### Gaps
- LightThinker appeared in a search summary as condensing each thought into a few cache tokens and dropping its raw KV, but I did not fetch it. I believe it is arXiv 2502.15589. Verify before citing.
- I did not fetch InftyThink. I believe it is arXiv 2503.06692. Verify.
- I did not search the exact phrases "latent thought dropout", "thinking with constant memory" or "thought retrieval" separately. Searches for "Markov chain of continuous thoughts", "latent tokens eviction", "redundant continuous thoughts prune", "latent thought merging compression", "bounded memory latent reasoning sliding window" and "pruning continuous thought tokens KV cache" found no latent-KV eviction paper.

## Q2: Selecting which latent or recurrent states a model attends to

### Takeaway
Related work selects across the depth axis of looped or recurrent models: depth attention (Dreamer), recursion-wise KV caches (MoR, T-LoopFormer), and bounded circular KV sharing (Huginn). Elastic-depth training (LoopFormer) is related too. None of these does learned keep/drop selection over the sequence of appended continuous-thought tokens in Coconut/CODI-style models. AGCLR is the only latent-CoT gating paper I found, and its gates act on a separate memory vector, not on attention.

### Cited Findings
- **Dreamer**: Knupp, Metzen, Bohn, Groh and Kersting, "Depth-Recurrent Attention Mixtures: Giving Latent Reasoning the Attention it Deserves", arXiv 2601.21582, 29 Jan 2026 (**2026**). It combines sequence attention, depth attention and sparse expert attention, and "alleviates the hidden-size bottleneck through attention along depth". It reports 2-8x fewer training tokens for the same accuracy. The abstract does not mention Coconut/CODI, selection of past states, extrapolation or dilution. **Adjacent.** Only the abstract was read. — [arXiv abs](https://arxiv.org/abs/2601.21582)
- **LoopFormer**: Jeddi, Ciccone and Taati, arXiv 2602.11451, 11 Feb 2026, ICLR 2026 (**2026**). Shortcut-consistency training conditions each loop on time and step size, so any loop budget gives consistent representations and performance scales smoothly with compute. No KV selection. **Adjacent.** — [arXiv abs](https://arxiv.org/abs/2602.11451); [ICLR poster](https://iclr.cc/virtual/2026/poster/10009450)
- **T-LoopFormer**, arXiv 2609.15160 (**2026**): per-loop KV caches with routing. Exited tokens must be masked so softmax does not give mass to empty slots. **Adjacent.** — [arXiv abs](https://arxiv.org/abs/2609.15160); [arXiv PDF](https://arxiv.org/pdf/2609.15160)
- **MoR**, arXiv 2507.10524, NeurIPS 2025: learned routers decide which tokens stay active, and so which are cached, at each recursion. Recursive KV sharing reuses the first recursion's KV. **Adjacent.** — [arXiv html](https://arxiv.org/html/2507.10524v3); [GitHub](https://github.com/raymin0223/mixture_of_recursions)
- **Huginn KV sharing** (see Q0 #4): fixed i mod k overwrite across recurrences with minimal performance loss. — [arXiv html](https://arxiv.org/html/2502.05171)
- **AGCLR** (see Q0 #1): read, forget and write gates over a memory vector inside a Coconut loop. No attention masking. — [arXiv html](https://arxiv.org/html/2606.07720)
- **Abstract Token Curriculum**: Gatmiry, Ghosh, Mirtaheri, Lee, Haghtalab, Abbe and Bartlett, arXiv 2609.19717, 17 Sep 2026 (**2026**). A curriculum yields continuous thoughts without direct supervision. Its theory says single-layer softmax attention focuses on the CoT tokens offering the "easiest path" to the next prediction. One search summary said attention concentrates on the most recent thought, but the abstract does not confirm this. **Adjacent.** — [arXiv abs](https://arxiv.org/abs/2609.19717)
- **Selective Attention** (see Q0 #3) and **TRIM-KV** (see Q0 #2) are the general learned-selection mechanisms. — [arXiv PDF](https://arxiv.org/pdf/2410.02703); [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/c3887741ee4e655494770be95303383f-Paper-Conference.pdf)

### Inferences
- In looped-depth models, each token's depth states are stacked in place. In latent CoT, each thought is a new sequence position. Selecting over appended latent positions is the less-explored axis, and it is where the idea can claim novelty.

### Gaps
- I did not read the full texts of Dreamer, LoopFormer or T-LoopFormer. I cannot rule out a depth-extrapolation experiment in them.
- I did not separately search "thought retrieval" or "latent state routing" phrasings.

## Q3: Decoding latents to intermediate results as a training signal (logit-lens supervision, step verification, process rewards on continuous thoughts)

### Takeaway
Step-level supervision of latents is well established: SIM-CoT's auxiliary decoder, LOTUS per-iteration supervision through the main LM head, PMPS structural process supervision, and local per-token supervision. Latent reward models exist that score latent trajectories (LatentRM, LTPO). But in every case the labels train the latents themselves or rank rollouts. I found no paper that uses decoded intermediate results as labels for an attention, retention or eviction gate.

### Cited Findings
- **SIM-CoT**, arXiv 2509.20317, ICLR 2026. During training, an auxiliary decoder aligns each implicit latent with one reasoning step on GSM8K-Aug, and the decoder is dropped at inference. It is motivated by latent homogenization and training collapse. **Partial overlap** on labels: step labels per latent, but via an auxiliary decoder and used to train latents, not gates. — [arXiv PDF](https://arxiv.org/pdf/2509.20317); [GitHub](https://github.com/InternLM/SIM-CoT)
- **LOTUS**: Fan et al., "Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers", arXiv 2606.31779 (**2026**). Supervising latent blocks h(t) directly through the main LM head per iteration, which is logit-lens-like, reaches 68.2% vs 70.0% for the post-loop default. Its Table 6 shows an R=6 model run at R=7 going from 70.0 to 69.3, "no further gain" (from the project's earlier reading). **Partial overlap.** — [arXiv html](https://arxiv.org/html/2606.31779)
- **PMPS**: "Structural Process Supervision for Latent Chain-of-Thought Reasoning", arXiv 2609.09928, Sep 2026 (**2026**). Process supervision on CODI-style latents. GPT-2 GSM8K: CODI 40.86 vs PMPS 44.35 (from the project's earlier reading). **Partial overlap** on labels. — [arXiv PDF](https://arxiv.org/pdf/2609.09928)
- **"What Makes Effective Supervision in Latent Chain-of-Thought"**, arXiv 2606.20075 (**2026**). On GSM8K-Aug, latent states from outcome-only supervision give no measurable benefit over direct answer prediction, while process supervision gives a more useful intermediate signal. **Adjacent.** It supports the value of intermediate labels. — [arXiv PDF](https://arxiv.org/pdf/2606.20075)
- **MUX**, arXiv 2607.18264 (**2026**): sides with local supervision of each latent token, since global supervision makes latents homogeneous and training collapses as their number grows. **Adjacent.** — [arXiv PDF](https://arxiv.org/pdf/2607.18264)
- **Latent Reasoning with Supervised Thinking States**, arXiv 2602.08332 (**2026**): chunk-level supervision that aligns CoT steps to positions in the query. **Adjacent.** — [arXiv PDF](https://arxiv.org/pdf/2602.08332)
- **Parallel Test-Time Scaling for Latent Reasoning Models**, arXiv 2510.07745: applies majority voting, best-of-N and beam search to latent reasoners. A search summary reports that Best-of-4 with a latent reward model (LatentRM) lifts Coconut on GSM8K from 34.0% to 46.7%. That number's attribution to this paper comes from a snippet and needs checking. **Adjacent to partial.** This is a learned verifier on continuous thoughts, used to rank rollouts, not to gate attention. — [arXiv html](https://arxiv.org/html/2510.07745)
- **Latent Thinking Optimization (LTPO)**, arXiv 2509.26314, ICLR 2026: optimizes latent thought sequences at test time with a latent reward model (from the search listing; not fetched). **Adjacent.** — [arXiv PDF](https://arxiv.org/pdf/2509.26314)
- **Latent RL on Coconut**: Özeren and Aßenmacher, arXiv 2512.11816, SwissText 2026. GRPO and a "Latent RL" method that acts on latent steps. RL-trained models still trail text CoT on math. Outcome reward only. **Adjacent.** — [arXiv abs](https://arxiv.org/abs/2512.11816v1)
- **Logit-lens readability of CODI latents (informal, not peer-reviewed)**:
  - LessWrong analyses report that plain logit lens on CODI's latents reads values mainly in late layers, and that intermediate values are recovered "weakly but significantly". — [LessWrong](https://www.lesswrong.com/posts/yetmqjCXEav9iaJGw/what-s-in-codi-s-latent-recurrent-thoughts)
  - A tuned lens shows the final answer on odd latents (1, 3, 5). — [GreaterWrong sprint #1](https://www.greaterwrong.com/posts/c4LYmzEC6cFQzrDZz/probing-codi-s-latent-reasoning-chain-with-logit-lens-and)
  - **Adjacent.** These are interpretability readouts, not training signals.
- **Causal caveat**: "Observable Patterns Are Not Explanations", arXiv 2606.12689 (**2026**), argues that decodable intermediate values have been over-read as evidence of a compressed scratchpad, and tests this with interventions. — [arXiv PDF](https://arxiv.org/pdf/2606.12689)

### Inferences
- Using decoded labels to supervise a gate, rather than the latents, appears new. A reviewer will push on label validity, given the causal-caveat paper and the weak logit-lens readability of CODI in the LessWrong posts. The planned check of decode labels against causal drop-one effects (memory note) directly answers that objection.
- SIM-CoT's decoder and LOTUS's LM-head supervision are both ready-made label sources, and natural ablations against plain logit-lens labels.

### Gaps
- I found no "latent step verification" paper that verifies each continuous thought against a gold intermediate value at inference. A search for "process reward model continuous latent thoughts step verification" returned only text-CoT PRMs such as ThinkPRM ([arXiv 2504.16828](https://arxiv.org/pdf/2504.16828)).
- I did not verify which paper (2510.07745 or 2509.26314) is the origin of the "34.0 to 46.7" Best-of-4 figure.

## Q4: Explaining degradation past trained latent steps by attention dilution; attention temperature or log-n scaling for latent steps

### Takeaway
I found no paper that attributes latent-CoT (Coconut/CODI/SIM-CoT) degradation past trained depth to attention dilution, and none that applies log-n or temperature scaling to latent steps. The nearest explanations for depth degradation are dynamical: settling vs drifting operators (2608.18222), fixed points in looped models (2604.11791), and the training-time remedy of random loop counts (Huginn, Kuo et al.). Log-n and softmax-temperature fixes exist only for sequence-length generalization: SSMax, Selective Attention temperature, 2506.16640.

### Cited Findings
- **Scalable-Softmax (SSMax)**: Nakanishi, arXiv 2501.19399. Names "attention fading": the softmax maximum shrinks as input size grows, which hurts length generalization. The fix multiplies logits by s·log n (n^s). Better length generalization and needle retrieval are reported, including when switching mid- or post-pretraining. Prior art for a log-n fix, but on sequence length, not latent steps. — [arXiv PDF](https://arxiv.org/pdf/2501.19399)
- **Selective Attention (temperature)**: Zhang et al., NeurIPS 2024. Position-aware temperature scaling, with a logarithmic rule τ = const + pow·log n derived from a power-law choice and linked to YaRN's temperature. — [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/14fc4a68da97a3d31eb11c642b0b10fc-Paper-Conference.pdf)
- **Long-context generalization with sparse attention**, arXiv 2506.16640, ICLR 2026. Argues that fixed-temperature softmax "inevitably disperses attention" as context grows. A log-n-based variant is reported to give up to 1000x length extrapolation on synthetic tasks. — [arXiv PDF](https://arxiv.org/pdf/2506.16640)
- **Critical scaling of inverse temperature**, arXiv 2605.12697 (**2026**): a unified classification of temperature schedules (explicit log, SSMax's s·log n per head, YaRN outside the window). — [arXiv PDF](https://arxiv.org/pdf/2605.12697)
- **Think Shallow, Solve Deep** (see Q0 #6): degradation is explained by drifting dynamics, and fixed with a terminal fixed-point objective, not attention. — [arXiv abs](https://arxiv.org/abs/2608.18222)
- **Blayney et al.**, "A Mechanistic Analysis of Looped Reasoning Language Models", arXiv 2604.11791 (**2026**). Looped LMs reach per-layer fixed points. Attention-head behaviour stabilizes at fixed points, and much attention mass sits on the first token as a sink. No CODI. — [arXiv html](https://arxiv.org/html/2604.11791v1)
- **Weak vs strong supervision of latent reasoning**: Cui et al., arXiv 2602.22441, 25 Feb 2026 (**2026**). Shortcut behaviour is widespread. Latent states implicitly prune and compress rather than search. Per the search summary, the top-10 attention targets of answer tokens are exclusively question tokens, not latent tokens. Stepwise alignment between latents and text may weaken as curricula add latent steps. — [arXiv html](https://arxiv.org/html/2602.22441); [arXiv abs](https://arxiv.org/abs/2602.22441)
- **Coconut** shows ProsQA accuracy rising from 0 to 6 continuous thoughts with the same weights, all within the trained range. — [arXiv PDF](https://arxiv.org/pdf/2412.06769)
- **Latent tokens often unnecessary**: Dilgren and Wiegreffe, ICLR 2026 Latent & Implicit Thinking Workshop / arXiv 2604.04902. Coconut and CODI rarely need their latent tokens for correct predictions, and the latent count is fixed rather than dynamic. — [OpenReview PDF](https://openreview.net/pdf/9354a22493eab2d3b815b7d7e3c7849904abae9a.pdf); [arXiv PDF](https://arxiv.org/pdf/2604.04902)
- **Liang and Pan**, arXiv 2602.00449 (**2026**): on sequential tasks CODI does not reliably sustain a full latent rollout as depth grows. No test of extra steps (from the project's earlier reading). — [arXiv abs](https://arxiv.org/abs/2602.00449)
- **Huginn** and **Kuo et al.**: random recurrence or loop counts in training, to support more steps at test time. — [Huginn html](https://arxiv.org/html/2502.05171); [Kuo et al. PDF](https://arxiv.org/pdf/2606.29983)

### Inferences
- An attention-dilution account of latent-CoT drift past trained depth appears unclaimed. The project's own copied-KV-entry result (memory note: exact KV copies reproduce SIM-CoT's drift, and attention on latents 1-6 falls from about .16 to .03) would be the first evidence of it. Log-n scaling of the readout over latent entries is an obvious, cheap prior-art baseline (SSMax) and should be tested next to the gate. Otherwise a reviewer will say a one-parameter temperature fix makes the gate unnecessary.
- 2608.18222 is the main rival explanation. The project's SIM-CoT result (settles, yet degrades as much as CODI) argues against "settling suffices" for appended-latent architectures. That contrast is a citable differentiator.

### Gaps
- No source uses "attention dilution" for latent CoT or looped models. Searches for "attention dilution" with latent reasoning, continuous thoughts or looped transformer returned only attention-sink and generic results.
- No source tests log-n or temperature scaling across latent steps or loop iterations. One search summary stated this explicitly after its own search.
- I found no paper that measures CODI, Coconut or SIM-CoT accuracy past the trained latent count other than LOTUS (one extra step, no gain). The query "Coconut CODI more latent tokens at inference than training accuracy degrades extrapolation" found none.

Queries tried, for the record:
- pruning continuous thought tokens latent CoT KV cache
- latent reasoning attention dilution more latent steps than training
- gated memory continuous latent thoughts select past thoughts
- logit lens supervision latent thoughts GSM8K-Aug
- adaptive latent reasoning length early exit
- latent thought merging compression Coconut CODI budget
- bounded memory latent reasoning sliding window
- process reward model continuous latent thoughts
- Coconut CODI more latent tokens than training
- log-n attention scaling looped transformer depth extrapolation
- sparse attention over latent thoughts learned selection
- Mixture-of-Recursions KV caching
- "latent tokens" eviction
- test-time scaling number of continuous thoughts
- step-level supervision decode latent SIM-CoT follow-up
- learned retention gate KV eviction
- LoopFormer elastic depth
- PENCIL InftyThink Delethink constant memory
- Markov chain of continuous thoughts
- continuous thought length generalization
- latent reasoning KV cache past thoughts attention mask
- RL continuous latent thoughts intermediate reward
- Think Shallow Solve Deep
- "attention dilution" latent/looped
- redundant continuous thoughts prune importance score
- Selective Attention Leviathan
- Scalable-Softmax log n
