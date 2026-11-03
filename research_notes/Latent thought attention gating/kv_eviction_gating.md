# KV-cache eviction, compression and learned token retention/gating (with emphasis on reasoning traces), as prior art for a learned gate over CODI-style latent-thought KV entries

Notes compiled 2026-11-02. **Tags** (relative to the planned method, a learned gate that evicts latent-thought KV entries in CODI-style models so that the answer readout attends to a bounded useful set when running 12/24 latent steps instead of the trained 6, supervised with verifiable intermediate-result labels): **supports** (evidence the premise holds), **motivates** (a problem the method addresses), **competes** (an alternative method or baseline to compare against), **threatens novelty** (overlaps with the core idea). **Verification markers:** [FT] = arXiv HTML full text read this session through a fetch tool that summarizes, so spot-check exact numbers before citing; [Abs] = arXiv abstract and metadata only, pulled from the arXiv API this session; [PK] = from prior knowledge and not re-verified this session (only the arXiv ID and title were confirmed). All arXiv IDs below were confirmed against the arXiv API on 2026-11-02. Papers dated 2026 are flagged **(2026)**.

## Q1. Heuristic eviction: what scores do H2O, StreamingLLM, Scissorhands, TOVA, SnapKV, PyramidKV, FastGen, Keyformer, Quest and KIVI use, and what do they find about which tokens matter?

### Takeaway
Heuristic policies score tokens in one of five ways:
1. **Position:** sinks plus a recency window (StreamingLLM).
2. **Accumulated attention over all past queries:** H2O, Scissorhands, Keyformer.
3. **Attention from the current query or a recent observation window:** TOVA, SnapKV, with PyramidKV and Ada-KV only reallocating budget.
4. **Per-head structural profiles:** FastGen, DuoAttention, RazorAttention.
5. **Geometry of keys and values:** KeyDiff key diversity, R-KV redundancy, VATP and VaSE value norms.

Quest and TokenSelect *select* entries per query but keep the full cache. The findings agree on four points: initial "sink" tokens must be kept; a small subset of tokens carries most of the attention mass; punctuation, separator and special tokens tend to persist; and importance is query-dependent and can come back later, which breaks one-shot greedy eviction.

### Cited Findings

**Position (sinks plus recency)**
- **StreamingLLM.** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) [Abs]
  - A plain sliding window fails once the text exceeds the cache. Keeping the KV of a few initial tokens ("attention sinks") recovers it, because initial tokens receive strong attention "even if they are not semantically important."
  - A dedicated sink placeholder token added in pre-training helps further. Reported: stable language modeling to 4M+ tokens, and up to 22.2× speedup over sliding-window recomputation. [Source](https://arxiv.org/abs/2309.17453)
  - Relevance: the recency baseline must keep sink and BOS entries, otherwise it is a strawman. **competes** (recency baseline)

**Accumulated attention**
- **H2O.** Zhang, Sheng, Zhou, Chen, et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs.* NeurIPS 2023 [PK venue]. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) [Abs]
  - A small set of "heavy hitter" tokens contributes most of the attention value. Their emergence "strongly correlates with the frequent co-occurrence of tokens," and removing them causes significant degradation.
  - The policy keeps a balance of recent tokens and heavy hitters, framed as dynamic submodular maximization with a guarantee.
  - Reported: 20% heavy hitters gives up to 29× throughput over DeepSpeed Zero-Inference and HF Accelerate, and 3× over FlexGen. [Source](https://arxiv.org/abs/2306.14048)
  - Score: attention accumulated over the generation, plus a recency window. In decoding this is "cumulative attention [tokens] receive during generation," as the Hold Onto That Thought benchmark reimplements it. [Source](https://arxiv.org/html/2512.12008) **competes** (this is the attention-score baseline)
- **Scissorhands.** Liu, Desai, Liao, Wang, et al. *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time.* NeurIPS 2023 [PK venue]. [arXiv:2305.17118](https://arxiv.org/abs/2305.17118) [Abs]
  - "Persistence of importance": only pivotal tokens that had substantial influence at one step will significantly influence future steps.
  - It keeps a fixed budget by storing pivotal tokens with higher probability. Reported: up to 5× KV reduction, and 20× when combined with 4-bit quantization. [Source](https://arxiv.org/abs/2305.17118)
  - [PK] Score: counts, over a history window, how often a token's attention falls below a threshold, then evicts high-count tokens while keeping recent ones. **competes**. This hypothesis is later contradicted for reasoning traces by LazyEviction and RaaS (see Q3).
- **Keyformer.** Adnan, Arunkumar, Jain, Nair, et al. *Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference.* MLSys 2024. [arXiv:2403.09054](https://arxiv.org/abs/2403.09054) [FT]
  - About 90% of attention weight falls on a subset of "key" tokens.
  - The score adds Gumbel noise to the unnormalized QKᵀ logits ("Keyformer adds this noise to the unnormalized logits derived from QK^T"). It then takes a Gumbel-softmax with temperature τ rising linearly from 1 to 2 over the generation, accumulated over prompt and generation steps per layer and head.
  - The cache keeps the w most recent tokens plus the top-scoring older tokens.
  - Motivation: evicting tokens removes their terms from the softmax normalizer, so "the attention weight of the discarded tokens is unevenly distributed" over the survivors. Reported: 2.1× lower latency and 2.4× higher throughput. [Source](https://arxiv.org/html/2403.09054) **competes**

**Current-query or observation-window attention**
- **TOVA.** Oren, Hassid, Yarden, Adi, Schwartz. *Transformers are Multi-State RNNs.* arXiv preprint (2024). [arXiv:2401.06104](https://arxiv.org/abs/2401.06104) [FT]
  - At each step, once the cache is full, it drops the single token with the lowest attention from the *current (newest) query*. Scores are averaged across the heads of a layer, so each layer keeps one token set; layer-wise averaging beat per-head scoring.
  - Findings on what is kept:
    - only 73–76% of kept tokens are recent;
    - the first token is kept to the end at every cache size;
    - punctuation persists, as do possessive endings (avg. 1,134 steps vs 249 overall at size 256) and plural proper nouns (321).
  - Plain window (FIFO) collapses: PPL 4812 at size 64 vs 9.53 for TOVA. TOVA comes within 0.4 PPL of full on PG-19 at 1/8 of the context. [Source](https://arxiv.org/html/2401.06104)
  - Relevance: TOVA is the natural "current-query attention" variant of the attention-score baseline. **competes**
- **SnapKV.** Li, Huang, Yang, Venkitesh, et al. *SnapKV: LLM Knows What You are Looking for Before Generation.* NeurIPS 2024 [PK venue]. [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) [Abs]
  - Each head consistently attends to specific prompt features during generation, and this pattern can be read from an "observation window" at the end of the prompt.
  - It selects "clustered important KV positions for each attention head" ([PK]: attention from the window is pooled with a 1-D pooling kernel). Prompt-only, fine-tuning-free.
  - Reported: 3.6× faster generation and 8.2× memory efficiency at 16K inputs. [Source](https://arxiv.org/abs/2404.14469)
  - Hold Onto That Thought builds a decoding variant, SnapKV-D, whose observation window slides every w=128 decoded tokens. [Source](https://arxiv.org/html/2512.12008) **competes**
- **PyramidKV.** Cai, Zhang, Gao, Liu, et al. *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling.* arXiv 2024. [arXiv:2406.02069](https://arxiv.org/abs/2406.02069) [Abs]
  - Attention "is scattering widely in lower layers, progressively consolidating within specific contexts, and ultimately focusing on critical tokens (a.k.a massive activation or attention sink) in higher layers." Budgets are therefore larger in lower layers and smaller in higher layers.
  - Reported: matches full KV on LongBench at 12% of the cache, and up to +20.5 points on TREC at 0.7%. [Source](https://arxiv.org/abs/2406.02069) **competes** (only an allocation axis)
- **Ada-KV.** Feng, Lv, Cao, Xie, et al. *Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation.* NeurIPS 2025. [arXiv:2407.11550](https://arxiv.org/abs/2407.11550) [Abs]
  - Derives a loss upper bound between pre- and post-eviction attention output, which explains what prior eviction methods optimize.
  - Allocates budget per head and plugs into existing selectors. [Source](https://arxiv.org/abs/2407.11550) **competes**

**Per-head structural profiling**
- **FastGen.** Ge, Zhang, Liu, Zhang, Han, Gao. *Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs.* ICLR 2024. [arXiv:2310.01801](https://arxiv.org/abs/2310.01801) [FT]
  - Each head is profiled at prompt encoding and assigned the cheapest policy that recovers the attention map at ratio T (0.95 in profiling). The policies:
    - special tokens only;
    - punctuation only;
    - locality (r_l = 0.3);
    - frequency / heavy hitter (r_f = 0.3);
    - greedy hybrids, which always include special tokens;
    - full cache.
  - Initial and final layers get more full-cache heads, while middle layers mostly attend to special tokens. Patterns are stable across decoding steps. Claim: "recover over 95% of attention scores with 35% cache compressed." [Source](https://arxiv.org/html/2310.01801) **competes**
- **RazorAttention.** Tang, Lin, Lin, Han, et al. *RazorAttention: Efficient KV Cache Compression Through Retrieval Heads.* arXiv 2024. [arXiv:2407.15891](https://arxiv.org/abs/2407.15891) [Abs]
  - Most heads focus locally and only a few "retrieval heads" attend to all tokens. It keeps a full cache for retrieval heads, drops remote tokens elsewhere, and adds a "compensation token." Reported: >70% KV reduction, training-free. [Source](https://arxiv.org/abs/2407.15891) **competes**

**Query-aware selection, no eviction**
- **Quest.** Tang, Zhao, Zhu, Xiao, Kasikci, Han. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* ICML 2024. [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) [Abs]
  - "The criticality of a token highly depends on the query." Quest tracks per-page minimal and maximal key values, estimates page criticality from the current query, and loads only the top-K pages.
  - Reported: up to 2.23× self-attention speedup and 7.03× lower latency. [Source](https://arxiv.org/abs/2406.10774)
  - It keeps the full cache, so it is O(N) in memory (as RaaS argues). [Source](https://arxiv.org/abs/2502.11147) **competes** (as an upper-bound style selector)
- **TokenSelect.** Wu, Pan, Wang, Chen, et al. *TokenSelect: Efficient Long-Context Inference and Length Extrapolation via Dynamic Token-Level KV Cache Selection.* EMNLP 2025. [arXiv:2411.02886](https://arxiv.org/abs/2411.02886) [Abs]
  - Despite appearing in "learned" lists, it is **training-free**. It measures per-head token criticality with QK dot products, combines heads by soft voting, and reuses selections across similar consecutive queries ("Selection Cache").
  - Reported: up to 23.84× attention speedup. [Source](https://arxiv.org/abs/2411.02886) **competes**

**Geometry of keys and values**
- **KeyDiff.** Park, Jones, Morse, Goel, et al. *KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments.* NeurIPS 2025. [arXiv:2504.15364](https://arxiv.org/abs/2504.15364) [Abs]
  - "Geometrically distinctive keys during LLM inference tend to have high attention scores." Eviction uses key similarity only, with no attention scores, so it is FlashAttention-compatible.
  - Reported: near-baseline accuracy on Math500 with DeepSeek-R1-Distill-Llama-8B. [Source](https://arxiv.org/abs/2504.15364)
  - Relevance: principled grounding for the "dedup by similarity" baseline. **competes**
- **VATP.** Guo, Kamigaito, Watanabe. *Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters.* EMNLP 2024. [arXiv:2406.12335](https://arxiv.org/abs/2406.12335) [Abs]
  - Value-vector norms are strongly non-uniform. Scoring by attention × ℓ1 value norm beats attention-only scoring on more than 12 of 16 LongBench tasks. [Source](https://arxiv.org/abs/2406.12335) **competes** (suggests a value-norm baseline)
- **SepLLM.** Chen, Shi, Li, Gao, et al. *SepLLM: Accelerate LLMs by Compressing One Segment into One Separator.* ICML 2025. [arXiv:2412.12094](https://arxiv.org/abs/2412.12094) [Abs]
  - Separator tokens (punctuation) get disproportionate attention, so segment information can be condensed into them. Reported: >50% KV reduction on GSM8K-CoT with Llama-3-8B. [Source](https://arxiv.org/abs/2412.12094) **supports** (structural anchors carry segment information)

**Quantization**
- **KIVI.** Liu, Yuan, Jin, Zhong, et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024. [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) [Abs]
  - Keys should be quantized per channel and values per token. Tuning-free 2-bit KV. Reported: 2.6× less peak memory, up to 4× larger batch, 2.35–3.47× throughput. [Source](https://arxiv.org/abs/2402.02750)
  - Orthogonal to eviction, with tiny relevance at 6–24 latent entries. **competes** (only as an orthogonal axis)

**Other attention-free or analysis work**
- **Expected Attention / KVPress.** Devoto, Jeblick, Jégou. *Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution.* arXiv 2025. [arXiv:2510.00636](https://arxiv.org/abs/2510.00636) [Abs]
  - Estimates the attention future queries will pay, in closed form from activation statistics.
  - Releases **KVPress**, a library with more than 20 eviction methods. Hold Onto That Thought extended it to decoding-phase compression. [Source](https://arxiv.org/abs/2510.00636); [Source](https://arxiv.org/html/2512.12008)
  - Relevance: KVPress is a ready-made harness for the heuristic baselines. **competes**
- **KVzip.** Kim, Kim, Kwon, Lee, et al. *KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction.* NeurIPS 2025 (oral). [arXiv:2505.23416](https://arxiv.org/abs/2505.23416) [Abs]
  - Scores a KV pair by how well the LLM can reconstruct the original context from the cache. Reported: 3–4× reduction. Query-aware evictors degrade "even at a 90% cache budget ratio under multi-query scenarios." [Source](https://arxiv.org/abs/2505.23416) **competes**
- **Transactional Attention (2026).** Basu. *Transactional Attention: Semantic Sponsorship for KV-Cache Retention.* arXiv 2026. [arXiv:2604.11288](https://arxiv.org/abs/2604.11288) [Abs]
  - "Dormant tokens," such as credentials that get near-zero attention but are needed later, defeat every attention, reconstruction or learned-retention-gate method. At K=16, six baselines (H2O, TOVA, SnapKV, StreamingLLM, PyramidKV, DynamicKV) score 0%.
  - Structural anchors "sponsor" adjacent value tokens. [Source](https://arxiv.org/abs/2604.11288)
  - Relevance: a needed-later intermediate result in a latent chain could be "dormant" under attention scores. This **motivates** label-based (verifiable) supervision.
- **Probabilistic interpretation (2026).** Geh, Chen, Israel, Grover, et al. *A Probabilistic Interpretation of KV Cache Eviction.* arXiv 2026. [arXiv:2608.28293](https://arxiv.org/abs/2608.28293) [Abs]
  - Formalizes KV eviction and proves it "computationally hard." Recasts it as expectation estimation and shows existing methods are "zero-variance biased estimators." [Source](https://arxiv.org/abs/2608.28293)
  - Relevance: justifies treating the logit-lens oracle as an approximation, not a true optimum. **supports**
- **Gated Attention.** Qiu, Wang, Zheng, Huang, et al. *Gated Attention for LLMs: Non-linearity, Sparsity, and Attention-Sink-Free.* arXiv 2025 (NeurIPS 2025 [PK venue/award]). [arXiv:2505.06708](https://arxiv.org/abs/2505.06708) [Abs]
  - A head-specific sigmoid gate after SDPA improves 15B MoE and 1.7B dense models trained on 3.5T tokens.
  - Its "query-dependent sparse gating... mitigates 'attention sink'" and improves long-context extrapolation. [Source](https://arxiv.org/abs/2505.06708)
  - Relevance: an architectural route to stopping readout attention from piling onto sinks. **motivates**

### Inferences
- The baseline suite maps cleanly onto prior work:
  - **Recency:** StreamingLLM, with sinks.
  - **Dedup:** KeyDiff key diversity or R-KV redundancy (Q3).
  - **Attention-score:** H2O accumulated, TOVA current-query, or SnapKV-D window. These three differ, and it is worth stating which one is used, or running all three.
  - **Random:** see Random Attention, Q3.
- The fixed-budget literature always pins sink and first tokens (StreamingLLM, TOVA, FastGen special tokens). CODI latents follow a question prefix and BOS. Every baseline should pin the question and BOS and evict only latent entries, otherwise baseline gaps will mostly reflect prefix protection (directly shown for reasoning in Random Attention, Q3).
- Several papers show value magnitude matters beyond attention (VATP here; VaSE and AvoKV-E in Q3). A value-norm heuristic baseline is cheap and plausibly strong for latent thoughts, given SCIT's finding that CODI-GPT2's computation is carried in value-cache suffixes (Q4).

### Gaps
- The exact Scissorhands and SnapKV scoring formulas (pooling kernel, history window, threshold) and the venues of H2O, Scissorhands, SnapKV and Gated Attention are from prior knowledge [PK]. Abstracts were re-verified, but the formula details were not.
- PyramidKV's final venue was not verified. TOVA's arXiv record says "preprint", and the venue was not checked.
- The question lists "Scissorhands" and "ScissorHands" separately. They are the same paper (2305.17118).

## Q2. Learned eviction and compression: DMC, Locret, trainable eviction, NAtS, Landmark, gist, AutoCompressors, NSA, MoBA, DuoAttention and 2025–2026 trainable sparse attention. How are the gates trained?

### Takeaway
Gate training falls into seven recipes:
1. **Gumbel-sigmoid or Gumbel-softmax relaxation** of a binary or categorical keep decision, with a one-sided compression-ratio loss plus LM or distillation loss (DMC, DMS, NAtS, ARM).
2. **Sigmoid retention scores used as additive log-biases** in attention, with a capacity hinge loss plus KL to the frozen model (TRIM-KV, Make Each Token Count).
3. **Supervised regression or ranking** toward an attention-derived "future importance" target (Locret, KVpop, ForesightKV, IndexMem, Judge Q).
4. **KL distillation of the model's own attention distribution** into a selector or indexer, which *selects* entries without evicting them (SeerAttention(-R), DSA). NSA and SAS instead learn end-to-end through the LM loss.
5. **Per-head continuous gates** trained by output distillation plus L1 sparsity (DuoAttention).
6. **RL from outcome reward or a KL reward** (NGC, RLKV, DistillCache, ForesightKV's GRPO stage).
7. **Attention-mask or architecture changes trained with the plain LM loss** (gist tokens, AutoCompressors, ICAE, Activation Beacon, Landmark, Selective Attention, FoX, Dynamic Context Pruning's sparse sigmoid).

No straight-through estimator showed up as the main recipe in the papers read here. **None of the papers found supervises the gate with verifiable intermediate-result labels.** Targets are attention-derived, output-distributional (KL), or outcome reward.

### Cited Findings

**Gumbel relaxations with a compression-ratio loss**
- **DMC.** Nawrot, Łańcucki, Chochowski, Tarjan, Ponti. *Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference.* ICML 2024. [arXiv:2403.09636](https://arxiv.org/abs/2403.09636) [FT]
  - **Decision:** per head, append vs merge, with no new parameters. α_t = round(sigmoid(k_t[0])) and ω_t = sigmoid(q_t[0]); a merge is a weighted average using running importance sums.
  - **Training:**
    - Gumbel-sigmoid relaxation ("stochastic reparametrization"), τ = 0.1;
    - bias c = 5 so α ≈ 0.0067 at start (no compression initially);
    - an additive mask during training so queries see only inference-time states;
    - a one-sided loss ℓ_CR = max(0, Σ(1−α) − n_l·n_h·n/CR)/(n_l·n_h·n), added to the LM loss;
    - target CR ramped linearly; 2–8% of pre-training data (24B–168B tokens for 2×–8×).
  - **Findings:** deeper layers compress most, and compression grows roughly logarithmically with sequence length. 4× on Llama-2-7B: MMLU 43.9 vs 44.6. It beats up-trained GQA, H2O and TOVA. [Source](https://arxiv.org/html/2403.09636)
  - **competes** (template for a differentiable keep/drop gate)
- **DMS.** Łańcucki, Staniszewski, Nawrot, Ponti. *Inference-Time Hyper-Scaling with KV Cache Compression.* NeurIPS 2025. [arXiv:2506.05345](https://arxiv.org/abs/2506.05345) [FT]
  - **Decision:** α_t from the attention input via a repurposed q/k neuron, no extra parameters.
  - **Training:**
    - Gumbel-sigmoid, bias −5;
    - an additive log(1−α) mask on QKᵀ;
    - a one-sided ℓ1 loss toward target α* annealed to 1 − 1/CR;
    - logit distillation from the original model;
    - 8× compression in about 1K steps (4× in 300, 8× in 700), vs about 44K steps for DMC.
  - **Delayed eviction:** a token marked at t stays visible until t + w (default w = 256). "Immediate eviction drastically deteriorates scores for every tested sliding window length."
  - **Reasoning results:** Qwen-R1 1.5B/7B/32B and Qwen3-8B; at 8×, AIME24 75.0 → 73.0 on Qwen3-8B.
  - The hyper-scaling gains differ between the abstract (+12.0 AIME24 for R1-32B) and the body (+12.5), so check the paper. [Source](https://arxiv.org/html/2506.05345); [Source](https://arxiv.org/abs/2506.05345)
  - **competes** (strongest trained-eviction baseline; its delayed eviction is directly applicable to latent steps)
- **NAtS.** Deng, Lindauer. *Neural Attention Search.* arXiv 2025. [arXiv:2502.13251](https://arxiv.org/abs/2502.13251) [FT]
  - **Token types,** learned per head:
    - Global: kept forever;
    - Local: kept until the next Global token;
    - Sliding Window: kept for W steps.
  - **Training:** a linear "Attention Score Layer" with Gumbel-Softmax sampling (GDAS-style; "allows the gradient to be backpropagated through the discrete choice"). Uses a multiplicative {0,1} mask because −∞ masks give invalid gradients. Sparsity is set by λ applied to Global/Local gradients.
  - **Settings:** trained from scratch (GPT-2-small style on PG-19) or as score-layer fine-tuning with output distillation on Llama-3.1-8B and Mistral-7B. Reported: about 3% KV in the sparsest LM, and RULER 4k 95.32 at a 50% budget vs 95.06 full. [Source](https://arxiv.org/html/2502.13251)
  - **competes**. "Local until next global" is close to a reduction rule on latent steps.
- **ARM (2026).** Zeng, Huang, Lu, Fang, et al. *ARM: Attention with Routed-Memory for Learnable Sparse Control.* ICML 2026. [arXiv:2609.24417](https://arxiv.org/abs/2609.24417) [Abs]
  - A fixed-size differentiable memory with a hierarchical router: Gumbel-Softmax slot selection plus sigmoid-gated updates that "softly combine new and stored information, avoiding hard eviction." A trained policy sets how much memory to access. [Source](https://arxiv.org/abs/2609.24417) **competes**

**Sigmoid retention with decay as a logit bias**
- **TRIM-KV.** Bui, Sharma, Lamba, Mishra, et al. *Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs.* arXiv Dec 2025. [arXiv:2512.03324](https://arxiv.org/abs/2512.03324) [FT]
  - **Gate:** a per-block MLP (d→512→h) on the token's hidden state gives β_t ∈ [0,1] per KV head, scored once at creation. Its influence decays as β_i^(t−i), added to the logits as (t−i)·log β_i. Bias is initialized to 18 (little forgetting).
  - **Loss:** forward KL to the frozen model plus CE, plus a capacity hinge (1/T)Σ_t(1/t)·max(0, Σ_i β_i^(t−i) − M), with M = 256 and λ = 1. Only the gates are trained, on OpenR1-Math-220k.
  - **Inference:** evict argmin β_j^(t−j) when over budget.
  - **Results:** beats H2O, SnapKV, R-KV, StreamingLLM, SeerAttention-R and Locret. Claims a 198.4% relative gain over R-KV/SnapKV at low budget. AIME24 on Qwen3-4B: 75.8 at a 4096 budget vs 65.5 full cache. Learned scores rediscover sinks, sliding windows (early layers), and periods as implicit "gist" tokens. [Source](https://arxiv.org/html/2512.03324)
  - **competes**. This is the closest gate *design* to the planned method; a TRIM-KV-style gate trained on CODI latents is a natural learned baseline. The beat-full-cache result also **supports** the premise that bounded retention can help.
- **Make Each Token Count (2026).** Bui, Nguyen, Cohan, Ying. *Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction.* arXiv May 2026. [arXiv:2605.09649](https://arxiv.org/abs/2605.09649) [Abs]
  - "Full-cache attention is not always optimal: in long contexts, irrelevant tokens can dilute attention away from useful evidence."
  - Retention gates plus a shared calibration projection give one global budget across layers, heads and modalities. Includes theory that retaining useful tokens reduces attention dilution. Matches or beats full cache. [Source](https://arxiv.org/abs/2605.09649)
  - **supports** (attention-dilution rationale for a bounded readout set) and **competes**

**Supervised regression or ranking to attention-derived targets**
- **Locret.** Huang, Yuan, Han, Xiao, et al. *Locret: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices.* arXiv 2024. [arXiv:2410.01805](https://arxiv.org/abs/2410.01805) [FT]
  - **Retaining head:** a small MLP σ([Q,K,V]W₁)W₂ (hidden 1024) per layer. It predicts a "causal importance score" (CIS): the max pre-softmax attention any answer token gives to the token, max-pooled over GQA groups.
  - **Loss:** Smooth-L1 regression plus an L2 smoothness term between adjacent predictions (α = 0.0025; "important segments... tend to be contiguous"). The backbone is frozen. Training: LongAlpaca, 3,000 steps, 0.47–0.80 A800 GPU-hours.
  - **Eviction:** under chunked prefill, keeping n_s "stabilizer" recent tokens; "the model fails entirely when stabilizers are absent." Reported: up to 20× compression at <10% loss. [Source](https://arxiv.org/html/2410.01805)
  - **competes**. The design (regress to answer-attention importance) is the closest analogue to supervising with labels; its labels are attention-derived, not verifiable.
- **KVpop (2026).** Hauzenberger, Schmidinger, Hartl, Stap, et al. *KVpop – Key-Value Cache Compression with Predictive Online Pruning.* arXiv Jul 2026. [arXiv:2607.05061](https://arxiv.org/abs/2607.05061) [FT]
  - **Scorer:** per-KV-head scorers on detached [k;v], either an MLP or a stateful mLSTM.
  - **Target:** mean post-softmax attention from queries ≥ w positions ahead, computed with a transposed-attention pass, no dense map.
  - **Loss:** a pairwise softplus "boundary" loss between the newly eligible token and the teacher's cutoff token, plus KL to a dense teacher. A delayed scorer scores a token only as it leaves the protected window.
  - **Results:** AIME/HMMT Qwen3-4B at 75%/88% compression retains 98%/97% of teacher (Table 1; the text says 95% and 94–97%, an inconsistency). DMS trails by 8–16 relative points at 88%. [Source](https://arxiv.org/html/2607.05061)
  - **competes**
- **ForesightKV (2026).** Dong, Liu, Li, Chen, et al. *ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution.* ICML 2026. [arXiv:2602.03203](https://arxiv.org/abs/2602.03203) [Abs]
  - A "Golden Eviction" algorithm uses future attention scores to find optimal evictions per step. The traces are distilled with a Pairwise Ranking Loss.
  - Eviction is then cast as an MDP and refined with GRPO "to mitigate the significant language modeling loss increase on low-entropy tokens." Beats prior methods on AIME24/25 with half the cache. [Source](https://arxiv.org/abs/2602.03203)
  - **competes**. Its "oracle trace, then distill" recipe parallels the planned oracle-then-gate design, except the oracle there is attention-based.
- **IndexMem (2026).** Yang, Gu, Xu, Li, et al. *IndexMem: Learned KV-Cache Eviction with Latent Memory for Long-Context LLM Inference.* arXiv May 2026. [arXiv:2605.25475](https://arxiv.org/abs/2605.25475) [Abs]
  - A learnable indexer predicts KV importance, and evicted tokens are folded into a compact online-updated "latent memory" that supplies residual readouts. Up to +25 on RULER under aggressive eviction. [Source](https://arxiv.org/abs/2605.25475) **competes**
- **Judge Q.** Liu, Wang, Xu, Ji, et al. *Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction.* AAAI 2026. [arXiv:2509.10798](https://arxiv.org/abs/2509.10798) [Abs]
  - A soft-token list is appended, and only the embedding layer is tuned, so that the soft tokens' attention map matches that of actually decoded tokens. Those queries then score KV importance. About +1 LongBench and +3 RULER. [Source](https://arxiv.org/abs/2509.10798)
  - **competes**. Analogous to a learned "readout probe query."

**KL distillation of attention into a selector (selection, not eviction)**
- **SeerAttention.** Gao, Zeng, Du, Cao, et al. *SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs.* arXiv 2024. [arXiv:2410.13276](https://arxiv.org/abs/2410.13276) [Abs]
  - An MoE-inspired AttnGate pools Q and K along the sequence, passes them through linear layers, and predicts block sparsity. Trained "in a lightweight self-distillation manner." [Source](https://arxiv.org/abs/2410.13276) **competes**
- **SeerAttention-R.** Gao, Guo, Cao, Xia, et al. *SeerAttention-R: Sparse Attention Adaptation for Long Reasoning.* arXiv 2025. [arXiv:2506.08889](https://arxiv.org/abs/2506.08889) [FT]
  - **Training:** the gate alone, frozen backbone, with KL to the column-wise 1-D max-pooled attention of the original model (GQA-shared). OpenR1-MATH-220k, about 0.4B tokens, 800 steps.
  - **Decode:** top-k blocks by budget or threshold, block size 64, last block always active. It keeps the full cache (offload suggested).
  - **Results (Qwen3-8B AIME24; full 74.5):** 56.6 / 72.3 / 74.2 / 75.1 at 2k / 4k / 6k / 8k budgets, vs Quest 13.3 / 44.2 / 52.5 / 59.6. [Source](https://arxiv.org/html/2506.08889) **competes**
- **DSA (DeepSeek Sparse Attention).** DeepSeek-AI. *DeepSeek-V3.2: Pushing the Frontier of Open LLMs.* arXiv Dec 2025. [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) [FT]
  - **Indexer:** a "lightning indexer" with I_{t,s} = Σ_j w_{t,j}·ReLU(q_{t,j}·k_s), FP8, few heads; top-k = 2048 tokens per query.
  - **Dense warm-up:** everything frozen except the indexer, with KL to the head-summed, L1-normalized main attention. 1,000 steps, about 2.1B tokens.
  - **Sparse stage:** KL restricted to the selected set, with the indexer input detached ("we detach the indexer input from the computational graph for separate optimization"). About 943.7B tokens. [Source](https://arxiv.org/html/2512.02556)
  - **competes** (illustrates the "distill attention, detach selector" recipe)
- **NSA.** Yuan, Gao, Dai, Luo, et al. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.* arXiv 2025 (ACL 2025 [PK venue/award]). [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) [FT]
  - **Branches:** compressed blocks (an MLP over blocks of length l, stride d), selected fine-grained blocks (top-n by compression-branch attention; l' = 64, n = 16 including 1 initial and 2 local blocks), and a sliding window. They are combined by gates g ∈ [0,1] "derived from input features via an MLP and sigmoid activation."
  - Argues that discrete selection in ClusterKV and MagicPIG "create[s] discontinuities in the computational graph," and that Quest/InfLLM-style selection being non-differentiable forces auxiliary losses. Pretrained end-to-end. [Source](https://arxiv.org/html/2502.11089) **competes**
- **MoBA.** Lu, Jiang, Liu, Du, et al. *MoBA: Mixture of Block Attention for Long-Context LLMs.* arXiv 2025. [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) [Abs]
  - Applies MoE principles to attention under a "less structure" principle, so the model "determine[s] where to attend autonomously." Switches between full and sparse attention; deployed for Kimi long context. [Source](https://arxiv.org/abs/2502.13189)
  - [PK] Router: parameter-free top-k over blocks by q·mean-pooled block keys, with the current block always attended. **competes**
- **InfLLM v2 (in MiniCPM4).** MiniCPM Team. *MiniCPM4: Ultra-Efficient LLMs on End Devices.* arXiv 2025. [arXiv:2506.07900](https://arxiv.org/abs/2506.07900) [Abs]
  - "InfLLM v2, a trainable sparse attention mechanism that accelerates both prefilling and decoding." [Source](https://arxiv.org/abs/2506.07900) **competes**

**More 2026 trainable sparse attention**
- **SAS (2026).** Li, Zhu, Gu, Hu, et al. *SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking.* arXiv Sep 2026. [arXiv:2609.13141](https://arxiv.org/abs/2609.13141) [Abs]
  - Criticizes hard top-K selectors trained by distilling dense attention: the ranking "is not directly aligned with their impact on predictions under a fixed attention budget."
  - Injects continuous selector scores into the attention logits "in log form" so the LM loss trains the selector, with normalized softmax gates. [Source](https://arxiv.org/abs/2609.13141)
  - **supports** (end-to-end, prediction-aligned gating beats attention-mimicking selectors) and **competes**
- **RTPurbo (2026).** Zhou, Li, Tang, Li, et al. *Full Attention Strikes Back: Transferring Full Attention into Sparse within Hundred Training Steps.* arXiv May 2026. [arXiv:2605.16928](https://arxiv.org/abs/2605.16928) [Abs]
  - Full KV is kept only for retrieval heads. A 16-dimensional token indexer with dynamic top-p is trained in a few hundred steps. [Source](https://arxiv.org/abs/2605.16928) **competes**
- **PHSA (2026).** Qiu, Wang, Chen, Zhang, et al. *Punctuation-aware Hybrid Trainable Sparse Attention for LLMs.* arXiv Jan 2026. [arXiv:2601.02819](https://arxiv.org/abs/2601.02819) [Abs]
  - Natively trainable sparse attention that uses punctuation as boundary anchors; beats InfLLM v2. [Source](https://arxiv.org/abs/2601.02819) **competes**

**Per-head gates**
- **DuoAttention.** Xiao, Tang, Zuo, Guo, et al. *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.* arXiv 2024 (ICLR 2025 [PK venue]). [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) [FT]
  - **Gate:** α ∈ [0,1] per KV head, initialized to 1. The head output is an α-blend of full and streaming (sink + recent) attention. The LLM is frozen.
  - **Loss:** L2 between the last hidden states of the full and mixed models on passkey tokens, plus L1 on α (λ = 0.05). 2,000 steps on synthetic multi-passkey retrieval in BookSum text. Gates are binarized by a sparsity quantile.
  - **Streaming heads:** keep e.g. 64 sinks + 256 recent tokens; performance "plateaus at 16 sink tokens and 64 recent tokens." Reported: up to 2.55× memory reduction (MHA). [Source](https://arxiv.org/html/2410.10819) **competes**
- **RLKV.** Du, Jiang, Tao, Liu, et al. *Which Heads Matter for Reasoning? RL-Guided KV Cache Compression.* arXiv 2025. [arXiv:2510.08525](https://arxiv.org/abs/2510.08525) [Abs]
  - "Existing token-dropping methods directly disrupt reasoning chains by removing intermediate steps." Uses RL "as a probe" to find which heads need full cache for reasoning quality, then compresses the others. 20–60% reduction, near-lossless. [Source](https://arxiv.org/abs/2510.08525) **competes**

**RL-trained policies**
- **DistillCache (2026).** Althoubi. *DistillCache: KL-Guided Adaptive KV-Cache Eviction.* arXiv Aug 2026. [arXiv:2608.08878](https://arxiv.org/abs/2608.08878) [Abs]
  - A policy network over attention statistics, value norms, entropy and position, trained with REINFORCE on a per-step KL-to-full-cache reward. Claims to beat ForesightKV and RLKV re-implementations. [Source](https://arxiv.org/abs/2608.08878) **competes**

**Forward-pass-only training**
- **Fast KVzip (2026).** Kim, Han, Yun. *Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction.* NeurIPS 2026. [arXiv:2601.17668](https://arxiv.org/abs/2601.17668) [Abs]
  - Lightweight "sink-attention gating modules" on frozen LLMs. Training "relies on forward passes... avoiding expensive backpropagation," with a task-agnostic reconstruction objective. Up to 70% eviction, near-lossless, including math. [Source](https://arxiv.org/abs/2601.17668) **competes**

**Architecture-level learned forgetting**
- **Dynamic Context Pruning.** Anagnostidis, Pavllo, Biggio, Noci, et al. *Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers.* arXiv 2023 (NeurIPS 2023 [PK venue]). [arXiv:2305.15805](https://arxiv.org/abs/2305.15805) [FT]
  - **Mechanism:** per-layer interaction queries and keys (r = 64) give pairwise drop scores through an α-sigmoid (sparse, entmax-like). α is annealed from 1 to 8 on a cosine schedule and becomes a step function at inference. Scores enter attention as log I.
  - **Training:** drops are irreversible through a cumulative product ("will remain dropped for all subsequent tokens"), with a γ-weighted sparsity regularizer. Fine-tuned GPT-2 models.
  - **Results:** up to about 80% of context pruned with minimal PPL change. "Most pruning is triggered by punctuation." [Source](https://arxiv.org/html/2305.15805)
  - **competes**. A GPT-2 learned-drop precedent at the same scale as CODI-GPT2.
- **Selective Attention.** Leviathan, Kalman, Matias. *Selective Attention Improves Transformer.* ICLR 2025. [arXiv:2410.02703](https://arxiv.org/abs/2410.02703) [FT]
  - **Mechanism:** parameter-free. It reuses head 0's logits as S, applies ReLU, zeroes the BOS column and the diagonal, shifts so masking applies only to future tokens, takes the cumulative sum F, and uses softmax(QKᵀ/√d − F).
  - **Pruning:** per-layer budgets K_l, dropping the token with the highest F. An optional L_mem auxiliary loss gives 16×/25×/47× attention-memory reduction at 512/1024/2048 context with equal PPL.
  - "Unneeded elements in the attention's context degrade performance." [Source](https://arxiv.org/html/2410.02703)
  - **supports** (removing unneeded context improves modeling) and **competes**
- **FoX.** Lin, Nikishin, He, Courville. *Forgetting Transformer: Softmax Attention with a Forget Gate.* ICLR 2025. [arXiv:2503.02130](https://arxiv.org/abs/2503.02130) [Abs]
  - A data-dependent forget gate down-weights unnormalized attention scores, beating the Transformer on long-context LM and length extrapolation. [Source](https://arxiv.org/abs/2503.02130)
- **Adaptive Computation Pruning (ACP).** Lin, Obando-Ceron, He, Courville. *Adaptive Computation Pruning for the Forgetting Transformer.* COLM 2025. [arXiv:2504.06949](https://arxiv.org/abs/2504.06949) [Abs]
  - Many FoX heads "forget quickly." ACP prunes dependencies strongly decayed by the gate, with "provably safe" thresholds. About 70% fewer attention FLOPs and no degradation. [Source](https://arxiv.org/abs/2504.06949)
  - **competes** (a learned decay gate that makes pruning safe; structurally the same as TRIM-KV's β^(t−i))

**Compression into learned tokens (LM loss with modified attention masks)**
- **Gist tokens.** Mu, Li, Goodman. *Learning to Compress Prompts with Gist Tokens.* NeurIPS 2023. [arXiv:2304.08467](https://arxiv.org/abs/2304.08467) [Abs]
  - Trained "with no additional cost over standard instruction finetuning by simply modifying Transformer attention masks." Up to 26× prompt compression. [Source](https://arxiv.org/abs/2304.08467)
  - **competes** (alternative of compressing rather than evicting)
- **AutoCompressors.** Chevalier, Wettig, Ajith, Chen. *Adapting Language Models to Compress Contexts.* EMNLP 2023. [arXiv:2305.14788](https://arxiv.org/abs/2305.14788) [Abs]
  - Summary vectors trained with an unsupervised LM objective across segments. Sequences up to 30,720 tokens. [Source](https://arxiv.org/abs/2305.14788)
- **ICAE.** Ge, Hu, Wang, Wang, et al. *In-context Autoencoder for Context Compression in an LLM.* ICLR 2024. [arXiv:2307.06945](https://arxiv.org/abs/2307.06945) [Abs]
  - Memory slots pretrained with autoencoding plus LM objectives; 4× compression with about 1% extra parameters. [Source](https://arxiv.org/abs/2307.06945)
- **Activation Beacon.** Zhang, Liu, Xiao, Shao, et al. *Long Context Compression with Activation Beacon.* arXiv 2024. [arXiv:2401.03462](https://arxiv.org/abs/2401.03462) [Abs]
  - Compresses per-layer KV activations into beacon tokens, trained by compression-based autoregression with randomly sampled compression ratios. 8× KV reduction. [Source](https://arxiv.org/abs/2401.03462)
- **Landmark Attention.** Mohtashami, Jaggi. *Landmark Attention: Random-Access Infinite Context Length for Transformers.* NeurIPS 2023. [arXiv:2305.16300](https://arxiv.org/abs/2305.16300) [Abs]
  - A landmark token per block; attention is trained to use it to select relevant blocks ("retrieval of blocks directly through the attention mechanism"). Extends LLaMA-7B past 32k. [Source](https://arxiv.org/abs/2305.16300) **competes**

**Choosing a compression configuration**
- **KV-Kaizen (2026).** Monteiro, Béthune, Filippova, Laguna, et al. *KV-Kaizen: Learning Context-Adaptive Cache Compression Choices.* arXiv Sep 2026. [arXiv:2609.37988](https://arxiv.org/abs/2609.37988) [Abs]
  - A learned per-layer selector over depth sharing, precision and rank, as an alternative to eviction ("a one-off decision to discard content may prove detrimental later"). [Source](https://arxiv.org/abs/2609.37988) **competes** (tangential)

### Inferences
- The most transferable training recipes for a latent-thought gate:
  - (a) DMS-style Gumbel-sigmoid with an additive log(1−α) mask, a one-sided budget loss and delayed eviction;
  - (b) TRIM-KV-style β with β^(t−i) log-bias, a capacity hinge and KL to the full model;
  - (c) KVpop- or ForesightKV-style supervised ranking against an oracle.

  The planned method's distinctive piece is the *label source*: verifiable intermediate results decoded via logit lens, instead of future attention, KL, or outcome reward. That lets the gate keep entries that attention-based targets would miss. This is the "dormant token" failure that Transactional Attention shows for all attention-based and learned-retention methods.
- SAS argues that distilling dense attention (SeerAttention, DSA) misaligns the ranking with prediction impact. That supports using a task-aligned target (intermediate-result correctness or answer loss) rather than mimicking CODI's own readout attention.
- For latent CoT with only 6–24 entries, the systems motivations of most of these papers (memory, throughput) do not apply. The relevant precedents are those claiming *accuracy* gains from bounded retention: TRIM-KV beats full cache, Make Each Token Count argues attention dilution, and Selective Attention says unneeded elements degrade performance.

### Gaps
- No paper found trains an eviction gate on *verifiable intermediate-step correctness labels*. This is a negative result from arXiv keyword search and the papers read; it is not proof of absence.
- No paper found clearly uses a straight-through estimator as the primary gate recipe. Detailed recipes for NAtS's sparsity gradient (Appendix C) and NSA's gate MLP were not read.
- MoBA router details, and the NSA, Gated Attention and DuoAttention venues, are [PK].
- The DMS and KVpop headline numbers have abstract/body/table inconsistencies, flagged above.

## Q3. Eviction on reasoning traces: Hold Onto That Thought, LazyEviction, Lethe, ThinKV, R-KV, SeerAttention-R, RaaS, and step- or thought-level eviction (Think Clearly, PENCIL, InftyThink and others). How does PENCIL decide what to drop?

### Takeaway
For explicit long CoT, the evidence points six ways:
1. **Elongation:** eviction in untrained models lengthens traces and can cause degenerate loops (Hold Onto That Thought, AbridgeKV, KV-Rescue, VaSE).
2. **Prompt protection:** pinning the prompt matters more than the selection score. Random eviction plus a pinned prompt ties the best scorers, because traces are redundant in text and across heads (Random Attention).
3. **Recurring importance:** importance comes back ("milestone," "recurrence," "thought revisiting"), so lagged or delayed eviction beats greedy eviction (RaaS, LazyEviction, BeaconKV, DMS, AvoKV-E).
4. **Redundancy:** penalizing redundancy among keys helps on repetitive self-verification (R-KV).
5. **Thought structure:** treating thoughts or steps as units helps (ThinKV, TAM, StepKV, Think Clearly).
6. **Trained compression is best at tight budgets:** PENCIL, LightThinker, Breadcrumbs, InftyThink, Delethink, Accordion, AbridgeKV, NGC.

**PENCIL** drops via a fixed pattern-matching reduction rule, C [CALL] T [SEP] A [RETURN] ⇒ C A, that erases the whole thought T. The *learned* part is only the LM emitting the special tokens, which it learns from solver-generated scaffolded traces.

### Cited Findings

**Benchmarks and diagnostics**
- **Hold Onto That Thought.** Liu, Palnitkar, Rabbani, Jae, et al. *Hold Onto That Thought: Assessing KV Cache Compression On Reasoning.* arXiv Dec 2025. [arXiv:2512.12008](https://arxiv.org/abs/2512.12008) [FT]
  - **Setup:** StreamingLLM, H2O, SnapKV-D, R-KV, KNorm and ShadowKV at budgets of 128–512, with up to 2,048 new tokens. Models: Llama-3.1-8B-Instruct and R1-Distill-Qwen-7B/14B, Nemotron-Nano-8B and R1-Distill-Llama-8B. Eight datasets including GSM8K and MATH-500.
  - **Results:** H2O and SnapKV-D dominate for reasoning models; for the non-reasoning model "no singular strategy" wins. Example, R1-Distill-Qwen-7B on GSM8K: SnapKV-D about 0.67–0.71, H2O 0.21–0.52, StreamingLLM 0.02–0.44, KNorm 0.00–0.16.
  - **Elongation:** "lower budgets are capable of triggering longer reasoning traces," with KNorm worst (some outputs never terminate).
  - **Tokens that matter:** H2O and SnapKV-D retain critical tokens (names, quantities, entities) at higher rates. [Source](https://arxiv.org/html/2512.12008)
  - **competes** (an attention-score baseline evaluation protocol) and **supports** (heavy-hitter tracking is meaningful on reasoning traces)
- **Semantic Integrity Matters.** Liu, Tang, Chen, Dong, et al. *Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression.* ICML 2026. [arXiv:2502.01941](https://arxiv.org/abs/2502.01941) [Abs]
  - KVFundaBench: retrieval tasks stay robust, but reasoning shows "severe Task-Dependent Degradation under aggressive compression due to disrupted CoT links." Proposes ShotKV, which keeps few-shot examples as indivisible units. [Source](https://arxiv.org/abs/2502.01941)
  - **motivates** (reasoning chains are fragile under token-level eviction)
- **Random Attention (2026).** Wang, Qiu, Zhao, Qian, et al. *Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning.* arXiv Sep 2026. [arXiv:2609.03430](https://arxiv.org/abs/2609.03430) [FT]
  - **Procedure:** pin the prompt, then draw i.i.d. uniform scores per KV head and keep the top-K.
  - **Setup:** Qwen3-4B/14B/32B and Phi-4-reasoning on MATH500, GPQA-D, AIME25/26, HMMT and LCB, at about 4× compression.
  - **Headline:** ties the best scorer. MATH500 on Qwen3-4B at K=1024: Random 0.874, TriAttention 0.864, R-KV 0.810, VaSE 0.809, SnapKV 0.703, full 0.939. 32–43% higher vLLM throughput.
  - **Prompt protection:** "Losing the prompt is catastrophic and cutting the trace at random is not." SnapKV goes 0.703 → 0.829 once the prompt is pinned; recency alone goes 0.246 → 0.843.
  - **Redundancy explains it:** in text ("the model restates what it still needs") and across heads (a planted fact is recovered 3% from the best single head vs 99% from all 8).
  - **Where scoring still wins:** rare once-stated facts (passcode: R-KV 84% vs Random 0%). [Source](https://arxiv.org/html/2609.03430)
  - **competes**. This defines the random baseline protocol. It also **motivates** a latent-CoT test: with 12–24 latent entries and no textual restating, the redundancy argument may fail and selection may matter.

**Training-free reasoning-specific eviction**
- **R-KV.** Cai, Xiao, Sun, Luo, et al. *R-KV: Redundancy-aware KV Cache Compression for Reasoning Models.* arXiv 2025. [arXiv:2505.24133](https://arxiv.org/abs/2505.24133) [FT]
  - **Importance:** attention from the last α = 8 tokens (GQA max-pooled), smoothed by a max over a 2W window.
  - **Redundancy:** cosine similarity of L2-normalized keys (diagonal zeroed). Similar pairs above a threshold T are linked, keeping only the β most recent, and R_i = softmax(mean similarity).
  - **Combination:** Z = λI − (1−λ)R with λ = 0.1. Both pure attention (λ = 1) and pure redundancy (λ = 0) perform worst. Compression every 128 decode steps.
  - **Results:** R1-Llama-8B is lossless at 10% on AIME24 and reaches 105% of full at 16%. "Repetitive content often receives disproportionately high attention scores." R1 outputs show 5–7× higher n-gram repetition than ground truth. [Source](https://arxiv.org/html/2505.24133)
  - **competes**. This is the reference implementation for the dedup baseline; its result that attention-only is worst explains why dedup and attention should both be tested.
- **LazyEviction.** Zhang, Zhang, Ma, Zhang, et al. *LazyEviction: Lagged KV Eviction with Attention Pattern Observation for Efficient Long Reasoning.* ACL 2026 (long), pp. 36335–36352. [arXiv:2506.15969](https://arxiv.org/abs/2506.15969); [ACL Anthology](https://aclanthology.org/2026.acl-long.1683/) [FT]
  - **Token Importance Recurrence:** ">95%" of tokens show recurrent attention. The Maximum Recurrence Interval (MRI) is the longest gap between re-activations (for Qwen on MATH500, 80% of tokens have MRI < 175 in about 8k-token outputs).
  - **Policy:** evict every W steps (W = the MRI covering 80% of tokens) and always keep the W most recent tokens. Priority is H1 (recency of last activation relative to MRI) plus H2 (smaller MRI).
  - **Results:** 50–70% KV reduction. GSM8K 50% budget, DS-Llama-8B: 80.06 vs 81.73 full. Baselines: TOVA, H2O, RaaS, R-KV. [Source](https://arxiv.org/html/2506.15969); [Source](https://aclanthology.org/2026.acl-long.1683/)
  - **competes**
- **Lethe.** Zeng, Zhao, Yang, Hou, et al. *Lethe: Layer- and Time-Adaptive KV Cache Pruning for Reasoning-Intensive LLM Serving.* AAAI 2026. [arXiv:2511.06029](https://arxiv.org/abs/2511.06029) [Abs]
  - Layer-wise budgets come from estimated attention redundancy. Multi-round pruning during generation uses "Recency-Aware Selective Retention" (RASR), which combines recency with relevance from evolving attention. Up to 2.56× throughput. [Source](https://arxiv.org/abs/2511.06029) **competes**
- **ThinKV.** Ramachandran, Neseem, Sakr, Venkatesan, et al. *ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models.* ICLR 2026 (oral). [arXiv:2510.01290](https://arxiv.org/abs/2510.01290) [FT]
  - **Thought types:** reasoning (R), execution (E) and transition (T), classified from attention sparsity. Two KDE thresholds are calibrated offline on 4 layers, and labels refresh every 128 steps.
  - **Importance:** measured by *counterfactual KL*, ranking R ≫ E ≫ T. Hybrid quantization at about R4/E4/T2 bits; the text also says 8/4/2, an inconsistency.
  - **Eviction:** at each transition thought, earlier segments are annealed down a retention ladder {64, 32, 16, 8, 4}. Kept tokens are picked by K-means on post-RoPE keys.
  - **Results:** "near-lossless accuracy with less than 5% of the original KV cache"; up to 5.8× throughput over R-KV. [Source](https://arxiv.org/html/2510.01290)
  - **competes** (thought-level eviction). Its counterfactual-KL importance is the nearest prior to "usefulness labels."
- **RaaS.** Hu, Huang, Wang, Li, et al. *RaaS: Reasoning-Aware Attention Sparsity for Efficient LLM Reasoning.* arXiv 2025. [arXiv:2502.11147](https://arxiv.org/abs/2502.11147) [FT]
  - **Token types:** "milestone tokens" (like lemmas) get high attention, are used, then fade; "phoenix tokens" (usually prefill/query tokens) are low for a while and then re-emerge.
  - **Policy:** LRU-like. Tokens with attention above the median get the newest timestamp, and the oldest are evicted. Prefill tokens are non-evictable. 16-token pages.
  - Matches Quest accuracy at O(L) memory. Fails at small budgets when the prefill eats the budget. [Source](https://arxiv.org/html/2502.11147)
  - **supports** (intermediate results act as milestones that become unimportant, which is exactly what the gate should learn) and **competes**
- **SeerAttention-R** (learned selection for reasoning; see Q2). [Source](https://arxiv.org/html/2506.08889) **competes**
- **Think Clearly.** Choi, Lee, Tack, Song, et al. *Think Clearly: Improving Reasoning via Redundant Token Pruning.* arXiv 2025; a search result lists it as EMNLP 2025 Findings, not verified. [arXiv:2507.08806](https://arxiv.org/abs/2507.08806) [Abs]
  - Attention in reasoning paths is scattered, and "incorrect answers exhibit greater attention sparsity."
  - Inserts an instruction to conclude the current step and an end-of-thinking token, scores tokens by attention *to that token*, and prunes low-contributing *chunks* before individual tokens (structure-aware). Then it removes the injected instruction and resumes.
  - Training-free accuracy *gains* on AIME and AMC. [Source](https://arxiv.org/abs/2507.08806); [search listing](https://preview.aclanthology.org/author-page-yu-wang-polytechnic/2025.findings-emnlp.1169.pdf)
  - **supports** (pruning distraction can raise accuracy). Its "attention from a summary/readout probe token" score is a strong extra baseline for latent CoT: score latents by the answer-readout position's attention.
- **Crystal-KV (2026).** Wang, Tang, Gong, Li, et al. *Crystal-KV: Efficient KV Cache Management for CoT LLMs via Answer-First Principle.* arXiv Jan 2026. [arXiv:2601.16986](https://arxiv.org/abs/2601.16986) [Abs]
  - Maps answer preferences into think-stage attention to separate "SlipKV" (keeps the flow but may mislead) from "CrystalKV" (contributes to answer correctness). Attention-based LRFU eviction plus adaptive budgets. [Source](https://arxiv.org/abs/2601.16986)
  - **competes**. "Answer-first" is conceptually close to "keep what the readout needs."
- **BeaconKV (2026).** Kim, Kim, Shim, Choi. *BeaconKV: KV Cache Compression Guided by Beacon Queries for Efficient Large Reasoning Model Inference.* ICML 2026. [arXiv:2609.04971](https://arxiv.org/abs/2609.04971) [Abs]
  - "Thought Revisiting Tokens" re-attend to distant context such as early plans, so recent queries are poor proxies. Their queries cluster, and cluster representatives ("beacon queries") are kept to score KV. Up to 5.8× memory reduction. [Source](https://arxiv.org/abs/2609.04971) **competes**
- **VaSE (2026).** Chang, Fu, Fu, Yang, et al. *Value-Aware Stochastic KV Cache Eviction for Reasoning Models.* arXiv Jun 2026. [arXiv:2606.03928](https://arxiv.org/abs/2606.03928) [Abs]
  - Evicting the few large-magnitude value states causes "repetitive reasoning loops." Stochastic eviction improves accuracy "by increasing cache diversity." It beats the best eviction method by >4% at 4× on Qwen3. [Source](https://arxiv.org/abs/2606.03928)
  - **competes**. Strengthens the case for value-norm and stochastic baselines.
- **AvoKV-E (2026).** Yu, Zhu, Chen, Wang, et al. *AvoKV-E: Payload-Aware KV Cache Eviction for Long Reasoning.* arXiv Oct 2026. [arXiv:2610.03007](https://arxiv.org/abs/2610.03007) [Abs]
  - Low-attention entries can carry large value payloads. New states "can appear stale before later queries have had a chance to read them." It delays eligibility, then ranks by read pressure, key redundancy and value payload. [Source](https://arxiv.org/abs/2610.03007) **competes**
- **ReasonAlloc (2026).** Liu, Shi, Li, Fei, et al. *ReasonAlloc: Hierarchical Decoding-Time KV Cache Budget Allocation for Reasoning Models.* arXiv Jun 2026. [arXiv:2606.11164](https://arxiv.org/abs/2606.11164) [Abs]
  - An offline layer "Reasoning Wave" pattern plus online head reallocation; beats R-KV, SnapKV and Pyramid-RKV at 128–512 budgets. [Source](https://arxiv.org/abs/2606.11164) **competes**
- **KV-Rescue (2026).** Cheong, Lim, Yun, Yoo. *KV-Rescue: Recovering Reasoning LM KV Eviction Loss via Stepwise Interleaving.* arXiv Aug 2026. [arXiv:2608.15797](https://arxiv.org/abs/2608.15797) [Abs]
  - Aggressive eviction causes "runaway degeneration." The loss is "an information gap... rather than a capability gap": an oracle choosing between an evicted 7B and a full-context 1.5B recovers 79% of the gap. [Source](https://arxiv.org/abs/2608.15797) **motivates**
- **KARA (2026).** Han, Wu, Yu, Isayev. *KARA: Efficient Reasoning LLM Serving via Sliding-Window KV Cache Compression.* arXiv May 2026. [arXiv:2607.01237](https://arxiv.org/abs/2607.01237) [Abs]
  - Compresses only the recent window, scoring with bidirectional attention, and expands picks into chunks ("Token2Chunk"). [Source](https://arxiv.org/abs/2607.01237) **competes**
- **TAM (2026).** Liu, Chong, Zhang, Zheng, et al. *Thought-Aware KV Cache Compaction for Reasoning via Adaptive Attention Matching.* arXiv Jun 2026. [arXiv:2608.12331](https://arxiv.org/abs/2608.12331) [Abs]
  - Segments CoT into thoughts, allocates budget by segment importance and size, and protects pivotal high-attention anchors. Proves the allocation is optimal under a convex error model with bounded cumulative error. [Source](https://arxiv.org/abs/2608.12331) **competes**
- **MorphKV.** Ghadia, Kumar, Jain, Nair, et al. *Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in LLMs.* ICML 2025. [arXiv:2503.00979](https://arxiv.org/abs/2503.00979) [Abs]
  - Constant-size cache for long responses, ranked by correlation with recent tokens' attention; "eliminates early-token bias." [Source](https://arxiv.org/abs/2503.00979) **competes**
- **StepKV (2026).** Feng, Liu, Li, Yu, et al. *StepKV: Step-Aware KV Cache Compression for LLM Agents.* arXiv Aug 2026. [arXiv:2609.22158](https://arxiv.org/abs/2609.22158) [Abs]
  - Reasoning steps are "first-class retention units." Step utility comes from trajectory signals and is combined with token saliency. Names the failure "Reasoning Continuity Disruption": early decisions get little recent attention but remain essential. [Source](https://arxiv.org/abs/2609.22158) **competes**

**Trained step- or thought-level compression and erasure**
- **PENCIL.** Yang, Srebro, McAllester, Li. *PENCIL: Long Thoughts with Short Memory.* ICML 2025. [arXiv:2503.14337](https://arxiv.org/abs/2503.14337); code: [github.com/chr26195/PENCIL](https://github.com/chr26195/PENCIL) [FT]
  - **How it decides what to drop:**
    - Special tokens [CALL], [SEP] and [RETURN] (plus [EOS]).
    - Reduction rule: **C [CALL] T [SEP] A [RETURN] ⇒ C A**, applied whenever the last generated token is [RETURN] and the pattern matches. T must contain no [CALL]; C may contain [CALL] (nesting); A may contain [CALL] (tail recursion).
    - The rule is a fixed pattern match that erases the *entire* thought T and keeps only the answer A. Generation and reduction alternate recursively.
    - What gets erased is decided implicitly by *where the model emits the special tokens*, learned from training traces ("based on patterns learned from training"). There is no separate learned erasure scorer.
  - **KV handling:** after reduction, A is spliced into the prefix, so A's KV must be recomputed; this is included in the FLOP estimate.
  - **Training:**
    - Solver traces (DPLL for SAT, recursive QBF, constraint propagation for Einstein) are scaffolded with the special tokens.
    - Each trace is split into reduced sub-sequences, and loss is computed only on tokens generated since the last reduction.
    - CoT baselines get the same special tokens.
  - **Results:**
    - SAT n=10: 100% vs 50% CoT.
    - QBF n=10: 100% vs CoT 73% in the table (the text says 61%).
    - Einstein 5×5: 97% vs 25%, with max sequence length 151,192 → 3,335.
    - Models: 10.63M (6-layer) and 25.19M (8-layer), both with 2048 context.
  - Theory: simulates Turing machines with optimal time and space. [Source](https://arxiv.org/html/2503.14337); [Source](https://arxiv.org/abs/2503.14337)
  - **competes** and **supports**. A step-level erase-intermediate-results rule with structural labels is the closest conceptual analogue to "keep only entries that encode useful intermediate results." PENCIL uses explicit tokens, exact rules and recomputation, not a learned soft gate over latent KV.
- **InftyThink.** Yan, Shen, Liu, Jiang, et al. *InftyThink: Breaking the Length Limits of Long-Context Reasoning in LLMs.* ICLR 2026. [arXiv:2503.06692](https://arxiv.org/abs/2503.06692) [Abs]
  - Interleaves short reasoning segments with progress summaries, giving a "sawtooth memory pattern." OpenR1-Math is reconstructed into 333K iterative examples. Qwen2.5-Math-7B gains 3–11% on MATH500, AIME24 and GPQA-D. [Source](https://arxiv.org/abs/2503.06692) **competes**
- **LightThinker.** Zhang, Zhu, Sun, Luo, et al. *LightThinker: Thinking Step-by-Step Compression.* EMNLP 2025 (oral). [arXiv:2502.15589](https://arxiv.org/abs/2502.15589) [Abs]
  - Trains the model on "when and how" to compress thoughts into gist tokens using constructed data and specialized attention masks, then discards the original chain. Introduces a "Dependency" metric. [Source](https://arxiv.org/abs/2502.15589)
  - **competes**. AbridgeKV reports that LightThinker, retrained on its data, "loses 10 points." [Source](https://arxiv.org/html/2610.20641)
- **Breadcrumbs Reasoning.** Monea, Feldman, Padmanabhan, Brantley, et al. *Breadcrumbs Reasoning: Memory-Efficient Reasoning with Compression Beacons.* arXiv 2025. [arXiv:2510.13797](https://arxiv.org/abs/2510.13797) [FT]
  - **Mechanism:** a beacon token every c tokens (c = 2–32). The preceding c tokens' KV are evicted after the beacon is encoded, and training simulates this by masking.
  - **Training:** joint PPO for an uncompressed teacher, plus online token-level KL distillation into the compressed student on the teacher's rollouts.
  - **Setup:** Qwen2.5-1.5B and Phi-4-mini on Countdown, LinSys and StarGraph.
  - **Results:** retains 65.1–89.8% of performance at 2–32× less memory. TOVA and StreamingLLM produce zero correct LinSys outputs. [Source](https://arxiv.org/html/2510.13797) **competes**
- **Markovian Thinker / Delethink.** Aghajohari, Chitsaz, Kazemnejad, Chandar, et al. *The Markovian Thinker: Architecture-Agnostic Linear Scaling of Reasoning.* arXiv 2025. [arXiv:2510.06557](https://arxiv.org/abs/2510.06557) [Abs]
  - Fixed 8K chunks: at each boundary the context is reset and a short learned carryover is kept, trained by RL. An R1-Distill-1.5B thinks up to 24K tokens, matching LongCoT-RL at a 24K budget. [Source](https://arxiv.org/abs/2510.06557) **competes**
- **Accordion-Thinking (2026).** Yang, Guo, Huang, Wang, et al. *Accordion-Thinking: Self-Regulated Step Summaries for Efficient and Readable LLM Reasoning.* arXiv Feb 2026. [arXiv:2602.03249](https://arxiv.org/abs/2602.03249) [Abs]
  - The model learns to summarize steps and discard prior thoughts ("Fold" mode). Under RL, the Fold–Unfold accuracy gap "progressively narrows and eventually vanishes." [Source](https://arxiv.org/abs/2602.03249) **competes**
- **Neural Garbage Collection (2026).** Li, Hamid, Fox, Goodman. *Neural Garbage Collection: Learning to Forget while Learning to Reason.* arXiv Apr 2026. [arXiv:2604.18002](https://arxiv.org/abs/2604.18002) [FT]
  - **Eviction as an action:** every δ tokens (256–350), blocks of 32 tokens, scored by the model's own attention (last 5 queries, no new parameters), are sampled by Gumbel-top-k. A fraction ε = 50% is evicted per round, and peak cache converges to L·δ/ε.
  - **Training:** Dr. GRPO on a binary outcome reward, with replayed attention masks. The eviction rate is written into the prompt.
  - **Results:** R1-Distill-Qwen-1.5B. Countdown at 2.4× peak reduction: 49.6% vs 21.2% for the next-best heuristic (SnapKV, KeyDiff, KNorm, StreamingLLM). Math at 3–5× reduction. [Source](https://arxiv.org/html/2604.18002)
  - **competes** (end-to-end learned eviction in reasoning). Explicit tokens only.
- **TokenSkip.** Xia, Leong, Wang, Li, Li. *TokenSkip: Controllable Chain-of-Thought Compression in LLMs.* EMNLP 2025. [arXiv:2502.12067](https://arxiv.org/abs/2502.12067) [Abs]
  - Text-level skipping of less important CoT tokens (40% fewer on GSM8K, <0.4% drop with Qwen2.5-14B). Not KV eviction. [Source](https://arxiv.org/abs/2502.12067) **motivates**
- **AbridgeKV (2026)**, **KVpop**, **ForesightKV**, **TRIM-KV**, **DMS** and **RLKV**: trained compression or eviction evaluated on reasoning; see Q2 and Q4.

### Inferences
- **Protocol for the planned baselines:**
  - (i) Pin the question prefix and BOS in all baselines (Random Attention, RaaS).
  - (ii) Report a delayed-eviction variant, where an entry becomes evictable only w latent steps after creation (DMS, KVpop, LazyEviction, AvoKV-E).
  - (iii) Implement dedup as R-KV's key-cosine redundancy, with and without the λ-mix with attention.
  - (iv) Run random with several seeds, per head vs shared across heads. Random Attention's result depends on per-head independent draws exploiting cross-head redundancy.
  - (v) Add a "readout-probe attention" baseline (Think Clearly / Crystal-KV style): score latents by the attention of the answer-readout or a probe position.
  - (vi) Add a value-norm baseline (VATP, VaSE, AvoKV-E).
- **Untrained models degrade under eviction.** AbridgeKV, Hold Onto That Thought and KV-Rescue all show this, so any learned gate trained *with* masked caches gets an unfair advantage over fixed-weight heuristics from train/test-consistency alone. A control that trains CODI with random or recency masks (no learned gate) would separate the "gate selectivity" effect from the "trained under eviction" effect. NGC's ablations and AbridgeKV's comparisons use similar logic.
- **What the gate should learn.** RaaS's milestone tokens, LazyEviction's recurrence and BeaconKV's revisiting tokens suggest intermediate results are read, fade, and are sometimes re-read. A gate trained on "this entry decodes to a gold intermediate result" may need to keep entries the readout re-reads late, which is something recency would drop.
- **PENCIL is the conceptual prior** for "erase intermediate computation once its result is consumed." Its decision uses structural tokens and an exact rule. A learned soft gate over latent KV is a different mechanism, so it does not seem to collide with PENCIL on novelty, but it should be framed relative to PENCIL.

### Gaps
- No paper titled "Memory-efficient reasoning via thought eviction" or "dynamic step pruning" was found in arXiv keyword searches or in a web search. The closest matches are ThinKV (thought-level eviction), Think Clearly (step/chunk pruning) and StepKV. [Source](https://arxiv.org/abs/2510.01290); [Source](https://arxiv.org/abs/2507.08806)
- Random Attention does not analyze when scoring matters for very short traces (an appendix on "short generations" was not read). That regime is the one most relevant to 6–24 latents.
- Per-budget numbers for RaaS, Think Clearly and LightThinker are in figures that were not read.
- Lethe's RASR formula was not read.

## Q4. Has any work applied KV eviction or gating to continuous/latent thought tokens or pause tokens? (closest competitors)

### Takeaway
I found **no paper that applies learned KV eviction or gating to CODI- or Coconut-style latent-thought KV entries so that the answer readout attends to a bounded set when running more latent steps than trained, supervised with intermediate-result labels.** The nearest neighbors fall into four clusters:
- **(a) Gated memory inside Coconut:** AGCLR, a learned write/read/forget-gated residual stream on GPT-2, with no KV eviction and no step extrapolation. Partial **novelty threat**.
- **(b) Bounded or gated KV across recurrent latent iterations in looped or recurrent-depth LMs:** MELT's gated shared cache, Huginn's modulo-k cache sharing, FlashLoop, ResidualQuant, PLT, T-LoopFormer. These **compete** on "bounded memory as latent depth grows." Huginn is the only one also showing extrapolation beyond trained depth.
- **(c) Pause or anchor tokens used as compressed KV for explicit reasoning:** AbridgeKV, Bottlenecked Transformers, Sep-Latent (attention decay into separator anchors for *implicit* reasoning; abstract only), Breadcrumbs, LightThinker.
- **(d) Latent-CoT mechanistic work** showing *which* latent KV entries carry the computation in CODI (SCIT, logit-lens bridge states, causal step structure). This **supports** both the oracle and the gate premise.

### Cited Findings

**Closest competitors (gating or memory management over latent thoughts)**
- **AGCLR (2026).** Farhan, Chaudhary. *Why Limit the Residual Stream to Layers and Not Tokens? Persistent Memory for Continuous Latent Reasoning.* arXiv Jun 2026. [arXiv:2606.07720](https://arxiv.org/abs/2606.07720) [FT]
  - **Problem:** "concept bottleneck": "at each reasoning pass, intermediate hidden states are overwritten, causing the model to lose critical facts computed in earlier steps as reasoning depth increases."
  - **Mechanism:** a persistent vector c_t across latent passes, with sigmoid read, forget and write gates (d×d matrices) on GPT-2. The gated state becomes the next latent input.
  - **Training:** Coconut-style curriculum. The paper does not specify gate supervision, intermediate labels or KV changes, and does not test more passes than trained. Single seed.
  - **Results:** GSM8K 34.0 vs Coconut 31.4 (CoT 40.6); HotpotQA EM 14.0 vs 10.4; ProsQA 96.0 vs 92.0. Ablation: no forget gate drops HotpotQA EM to 8.4. [Source](https://arxiv.org/html/2606.07720)
  - **threatens novelty (partial).** It is "learned gates that prune context in continuous latent reasoning." The differences: (1) residual memory, not KV eviction; (2) no attention-readout boundedness; (3) no extrapolation in step count; (4) no verifiable intermediate labels; (5) Coconut, not CODI.
- **MELT (2026).** Conchello Vendrell, Padres Masdemont, Grillo, Ros-Giralt, et al. *Memory-Efficient Looped Transformer: Decoupling Compute from Memory in Looped Language Models.* arXiv May 2026. [arXiv:2605.07721](https://arxiv.org/abs/2605.07721) [FT]
  - Starts from Ouro LoopLMs, whose per-loop KV "grow[s] linearly with reasoning depth." MELT keeps "a single KV cache per layer that is shared across reasoning loops... updated over time via a learnable gating mechanism," with z = σ(xW_z + hU_z + b) and h_t = z⊙h_{t−1} + (1−z)⊙x_t, plus a per-loop sliding window.
  - **Training:** chunk-wise distillation from the LoopLM (per the abstract, "interpolated transition, followed by attention-aligned distillation").
  - **Results:** 3.98× less KV than Ouro-1.4B at 32k, accuracy on par. Loop count is fixed at inference, and extrapolation is not evaluated. [Source](https://arxiv.org/html/2605.07721); [Source](https://arxiv.org/abs/2605.07721)
  - **competes / threatens novelty (partial):** "gated KV update over latent iterations."
- **Huginn (recurrent depth).** Geiping, McLeish, Jain, Kirchenbauer, et al. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* arXiv 2025. [arXiv:2502.05171](https://arxiv.org/abs/2502.05171) [FT]
  - **KV sharing:** training-free; "set a fixed KV-cache budget for the recurrence at every token k, and at iteration i, read and write the cache entry i mod k." MT-Bench 5.86 with a small budget vs 5.63 standard (the text's "44" is probably a typo for 4).
  - **Extrapolation:** trained at a mean of 32 recurrences; at r=64 GSM8K improves to 47.23% (flexible). Saturation is task-dependent. [Source](https://arxiv.org/html/2502.05171)
  - **competes**. A modulo-k recency-overwrite is a "recency" heuristic over latent-iteration KV, and it works zero-shot when extrapolating depth. That is a strong precedent the planned recency baseline should mirror.
- **Looped-transformer KV compression**
  - **FlashLoop (2026).** Yang, Liu. *FlashLoop: Fast and Memory-Efficient Looped Transformers via Lazy Updates.* arXiv Sep 2026. [arXiv:2609.29812](https://arxiv.org/abs/2609.29812) [Abs]
    - "As recurrence proceeds, state changes become increasingly concentrated on a small subset of tokens," and cross-loop KV residuals become quantizable. Training-free; up to 6× KV reduction. [Source](https://arxiv.org/abs/2609.29812)
  - **ResidualQuant (2026).** Kim, Lee, Cho, Han, et al. *ResidualQuant: KV Cache Quantization for Looped Transformers with 2-Bit Residuals.* arXiv Oct 2026. [arXiv:2610.10381](https://arxiv.org/abs/2610.10381) [Abs]
    - "KV states across loops are highly similar." Uses final-loop KV as a reference with 2-bit residuals. [Source](https://arxiv.org/abs/2610.10381)
  - **PLT.** Wu, Chen, Luo, Yan, et al. *Parallel Loop Transformer for Efficient Test-Time Computation Scaling.* arXiv Oct 2025. [arXiv:2510.24824](https://arxiv.org/abs/2510.24824) [Abs]
    - Shares first-loop KV across loops, with gated sliding-window attention. [Source](https://arxiv.org/abs/2510.24824)
  - **T-LoopFormer (2026).** Yu, Zhang, Cui, Zhao. *T-LoopFormer: Token-Level Elastic-Depth Looped Transformers with Dynamic Routing.* arXiv Sep 2026. [arXiv:2609.15160](https://arxiv.org/abs/2609.15160) [Abs]
    - Per-token loop counts via a router, with a recursion-wise KV cache. [Source](https://arxiv.org/abs/2609.15160)
  - Tag for this group: **supports** (cross-iteration KV is highly redundant, so a dedup baseline could be strong when running extra latent steps) and **competes**.
- **Bottlenecked Transformers.** Oomerjee, Fountas, Bou-Ammar, Wang. *Bottlenecked Transformers: Periodic KV Cache Consolidation for Generalised Reasoning.* arXiv May 2025. [arXiv:2505.16950](https://arxiv.org/abs/2505.16950) [Abs]
  - An auxiliary "Cache Processor" does non-causal in-place KV rewrites at newline step boundaries. It consolidates recent entries and "reconsolidates a small, top-k attention-selected set of prior entries," motivated by information-bottleneck theory.
  - Beats vanilla and *pause-token-augmented* baselines by up to +6.6pp on math. [Source](https://arxiv.org/abs/2505.16950)
  - **competes** (learned KV rewriting at reasoning-step granularity, framed as latent-space computation)
- **Sep-Latent (2026).** Xu, Huang, Zhou, Lin, et al. *Sep-Latent: Separator Tokens as Natural Compression Anchors for Implicit Reasoning.* arXiv Oct 2026. [arXiv:2610.22924](https://arxiv.org/abs/2610.22924) [Abs]
  - Coconut and CODI "rely on fixed-length latent sequences and suffer from representational collapse."
  - An "Attention-Decay mechanism" forces the model to "aggregate and compress reasoning logic into these anchors," with multi-level distillation and adaptive termination. GSM8k-Aug 56.8% at about 5.4 tokens. [Source](https://arxiv.org/abs/2610.22924)
  - **threatens novelty (partial, abstract only).** It is attention shaping inside implicit reasoning, but aimed at compressing explicit text into anchors, not at bounding attention over extra latent steps. The full text should be checked for whether "attention decay" evicts or down-weights latent entries.
- **AbridgeKV (2026).** Cai, Yang, Kontonis, Papailiopoulos, et al. *AbridgeKV: Compressing Long Reasoning into the KV Cache of Learned Pause Tokens.* arXiv Oct 2026. [arXiv:2610.20641](https://arxiv.org/abs/2610.20641) [FT]
  - **Mechanism:** each 512-token block of generated text is compressed into the KV ("code rows") of up to 256 learned pause tokens (mean 128 kept), then the raw block KV is freed. Code rows take positions *inside* the replaced block (+6.2 avg@8 vs appended). Per-layer quotas come from measured value.
  - **Training:** plain next-token loss; distillation did not help. Qwen3-4B-Base on 2B tokens.
  - **Results:** a quarter cache gives 43.0 vs 43.8 full KV on 446 competition problems. "A model never trained on a cache with entries removed generates longer, often up to the length limit": 38–58% of samples hit 32K under simple eviction. R-KV is within noise at 4096 but writes 40–69% more tokens. Sink+recency, random and H2O-style baselines are included. [Source](https://arxiv.org/html/2610.20641)
  - **competes**. Pause tokens are compressors of explicit text here, not latent reasoning steps. Its baseline set (sink+recency, random, H2O-style, R-KV) closely matches the planned baselines.
- **KaVa.** Kuzina, Pioro, Whatmough, Ehteshami Bejnordi. *KaVa: Latent Reasoning via Compressed KV-Cache Distillation.* ICLR 2026. [arXiv:2510.02312](https://arxiv.org/abs/2510.02312) [FT]
  - **Teacher side:** the teacher's CoT KV is compressed with an R-KV adaptation (S = λI + (1−λ)R, λ = 0.1, importance from answer tokens' attention to CoT keys; eviction "only applied during training").
  - **Student side:** a PCCoT student's 24 latent KV entries are matched to it with L1 or MSE, added on top of CODI's distillation loss.
  - **Results:** LLaMA3.2-1B GSM8k-Aug 56.5 vs CODI 53.9; GSM8k-Aug-NL 55.7 vs 50.1. [Source](https://arxiv.org/html/2510.02312)
  - **supports / motivates.** It shows latent-thought KV can be aligned to an R-KV-selected subset of CoT KV, so attention+redundancy selection is meaningful in CODI-land. It applies eviction to the *teacher*, not to latent entries at inference.
- **SPOT (2026).** Chu, Shao, Liu, Hao, et al. *SPOT: Span-level Pause-of-Thought for Efficient and Interpretable Latent Reasoning in LLMs.* arXiv Mar 2026. [arXiv:2603.06222](https://arxiv.org/abs/2603.06222) [Abs]
  - Compresses CoT into latent pause tokens with Sinkhorn span-level alignment and a "Frozen-Head Decoding Constraint" that keeps latents decodable by the LM head. [Source](https://arxiv.org/abs/2603.06222)
  - **supports** (logit-lens-decodable latents are trainable, which is what the oracle and labels need)
- **Dynamic K-Head (2026).** Li, Chen, Li, Cai. *Dynamic K-Head: Difficulty-Supervised Budget Selection for Continuous Latent Reasoning.* arXiv Oct 2026. [arXiv:2610.17679](https://arxiv.org/abs/2610.17679) [Abs]
  - In continuous CoT, "accuracy need not improve monotonically with K." A question-conditioned budget selector on CoDI (Llama-3.2-1B, GSM8K) reaches 54.44% at 12.03 tokens vs 52.31% at 15.21 tokens for fixed K=6. [Source](https://arxiv.org/abs/2610.17679)
  - **motivates.** Non-monotonic K is the symptom the gate targets. This paper is a *budget-selection* alternative, not a gate.

**Mechanistic evidence about which latent KV entries matter in CODI**
- **SCIT (2026).** Ding, Huang, Yang. *SCIT: Testing Causal Cache Carriers in Latent Chain-of-Thought Models.* EMNLP 2026. [arXiv:2608.27265](https://arxiv.org/abs/2608.27265) [FT]
  - **Carrier:** on CODI-GPT2 (6 latent steps), counterfactual arithmetic transfers "primarily through value-cache suffix trajectories rather than hidden states, keys, reusable answer slots, or single-token triggers."
  - **Where:** the strongest block is layers 8–9 values (0.875–0.908 target win). A 1-token suffix reaches 0.203, while k≥3 suffixes reach 0.984–1.000. Late-value corruption drops target win to 0.211.
  - **Caveat:** checkpoint-specific; newly trained CODI seeds "fail to reproduce" the late 8–9 value carrier. [Source](https://arxiv.org/html/2608.27265)
  - **supports** (the readout depends on a small, localized set of latent KV entries, largely values; a gate or oracle should consider values and multi-step suffixes)
- **Do Latent-CoT Models Think Step-by-Step? (2026).** Liang, Pan. arXiv Jan 2026. [arXiv:2602.00449](https://arxiv.org/abs/2602.00449) [Abs]
  - On CODI with known intermediate states, logit lens and probes find "faithful bridge states across latent-thought positions" in short-horizon tasks, with "late fusion at the answer readout." At greater depth CODI "does not reliably sustain a full latent rollout." [Source](https://arxiv.org/abs/2602.00449)
  - **supports** (validity of a logit-lens gold-intermediate oracle on short horizons) and **motivates** (degradation at depth)
- **Dynamics Within Latent CoT (2026).** Li, Bai, Chen, Li, et al. *Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure.* ICML 2026. [arXiv:2602.08783](https://arxiv.org/abs/2602.08783) [Abs]
  - Do-interventions on Coconut and CODI steps: "latent-step budgets behave less like homogeneous extra depth and more like staged functionality with non-local routing." [Source](https://arxiv.org/abs/2602.08783) **supports** (not all latent steps are equally needed)
- **Interpreting Latent CoT Reasoning as Dynamical Systems (2026).** Duraipandian, Boyane, Nagesh, Francis, et al. ICML 2026 FoGen Workshop. [arXiv:2607.09698](https://arxiv.org/abs/2607.09698) [Abs]
  - "CODI behaves as a stable attractor, while COCONUT behaves as an unstable expanding system." [Source](https://arxiv.org/abs/2607.09698)
  - **motivates.** If CODI settles into an attractor, extra latent steps may add near-duplicate entries, which favors the dedup baseline (see Inferences).

**Context: latent and pause-token methods themselves, with no eviction**
- Coconut: Hao et al., *Training LLMs to Reason in a Continuous Latent Space.* [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
- CODI: Shen et al., *CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation.* [arXiv:2502.21074](https://arxiv.org/abs/2502.21074)
- Pause tokens: Goyal et al., *Think before you speak: Training Language Models With Pause Tokens.* [arXiv:2310.02226](https://arxiv.org/abs/2310.02226)
- CCoT: Cheng and Van Durme, *Compressed Chain of Thought.* [arXiv:2412.13171](https://arxiv.org/abs/2412.13171)
- IDs and titles verified [Abs]. No eviction or gating component in these papers is known to me [PK].
- **Almost Free State Prediction Separation (2026).** Langford, Godey, Monea, Artzi, et al. [arXiv:2609.03807](https://arxiv.org/abs/2609.03807) [Abs]
  - "A free pause token: a prediction stream that writes no keys or values at all." Tangential: a pause token that never enters the KV cache. [Source](https://arxiv.org/abs/2609.03807)
- **Cache Merging (2026).** Baquero, Brito. *Cache Merging as a Convergent Replicated State for Multi-Agent Latent Reasoning.* [arXiv:2607.01308](https://arxiv.org/abs/2607.01308) [Abs]
  - Order-invariant merging of per-agent latent KV. Tangential. [Source](https://arxiv.org/abs/2607.01308)

### Inferences
- **The open slot.** The combination is: (i) KV-level eviction or gating of *latent-thought entries* at inference in CODI; (ii) the goal of keeping the readout's attention bounded and useful when *extrapolating* latent steps (6 → 12/24); and (iii) supervision from verifiable intermediate results decoded via logit lens. None of the papers found does it.
  - AGCLR covers learned gating in latent reasoning without KV, extrapolation or labels.
  - MELT and Huginn cover bounded KV over latent iterations, without learned selection by usefulness or labels (Huginn is a fixed modulo rule; MELT's loop count is fixed).
  - Sep-Latent covers attention shaping in implicit reasoning, with unclear eviction (abstract only).
  - NGC, TRIM-KV, KVpop and ForesightKV cover learned eviction on *explicit* reasoning.
- **Reviewer expectations.** The likely asks are:
  - comparisons to a TRIM-KV-style learned retention gate trained without intermediate labels (KL plus capacity only);
  - a DMS-style Gumbel-sigmoid gate;
  - Huginn's modulo-k cache sharing as the zero-shot latent-iteration baseline;
  - an AGCLR-style gated memory, if the claim is framed as "gating in latent reasoning."
- **Implications from SCIT.** Its finding (late, multi-step value-cache suffix in middle-to-late layers; checkpoint-specific) implies three things. (a) The gate might need to be per-layer, or at least act on values. (b) Per-entry logit-lens labels on the residual stream may not match the KV objects that actually carry the computation. (c) Results should be replicated across CODI seeds.
- **Dedup may be strong.** The looped-transformer findings (cross-loop KV "highly similar"; changes concentrate on few tokens) and the "CODI is a stable attractor" result suggest that extra latent steps past 6 may produce near-duplicate KV. That predicts dedup will be a strong heuristic, and that random may look good for redundancy reasons, as in Random Attention. Both are worth testing before claiming the learned gate's advantage.

### Gaps
- The search for latent-thought eviction ran through arXiv API keyword queries and two web searches, with terms including latent reasoning, continuous thought, pause tokens, Coconut, CODI, looped and recurrent-depth plus KV cache, and thought or step eviction. It may miss works that use other vocabulary (e.g., "memory tokens," "register tokens," "scratchpad pruning," "latent working memory"), venue-only papers without arXiv versions, and OpenReview submissions under review.
- Sep-Latent, Bottlenecked Transformers, FlashLoop, ResidualQuant, T-LoopFormer, PLT, Dynamic K-Head, the 2602.x mechanistic papers, and the dynamical-systems paper were seen only as abstracts.
- AGCLR and MELT were read via summarized HTML. MELT's summary did not find the "interpolated transition / attention-aligned distillation" description that the abstract states.
- Venue status for Huginn, Breadcrumbs, R-KV, SeerAttention-R, RaaS, NAtS, Locret and MoBA was not verified beyond arXiv metadata.
