# Attention dilution as sequence length grows, and the fixes proposed for it (for latent-step gating in CODI-style models)

Notes compiled 2026-11-02. Scope: why softmax attention spreads out as more keys are added, the fixes proposed for it, and whether any of it has been applied to latent, pause, or recurrent thought tokens.

Tag legend (relative to our idea, which is a learned keep/drop or soft log-gate on attention logits over latent KV entries, motivated by readout attention spreading out when CODI runs more latent steps than it was trained on):
- **[supports]**: evidence that the mechanism we hypothesize is real or that our diagnosis is right.
- **[motivates]**: theory or background that gives a reason to try the gate.
- **[competes]**: an alternative fix that could solve the same problem, which the gate should be compared against.
- **[threatens novelty]**: the same mechanism, or the same application, is already published.

Evidence level: "(full text checked)" means I pulled the arXiv PDF and read or grepped the relevant passages. "(abstract only)" means the claim comes from the arXiv abstract or metadata. "2026 preprint" marks work first posted in 2026.

---

## Q1. Theory: why softmax attention spreads out with more keys, and the attention-level fixes (log-n scaling, entropy invariance, SSMax, sharpness limits, Selective attention, Diff Transformer, stick-breaking, sigmoid attention)

### Takeaway
The theory is settled and has a precise form. If the logits stay bounded, every softmax weight falls as Θ(1/n) when keys are added (Veličković et al., Lemma 2.1). Several groups independently arrive at a fix that multiplies the logits by about log n (Chiang & Cholak 2022; Su 2021; SSMax 2025; Qwen; YaRN uses roughly (log n)²), and a 2026 ICLR paper proves that log n is the critical scaling. The alternatives fall into three families: (a) temperature or log-n multipliers on the logits; (b) additive, data-dependent negative biases on the logits (Selective attention, FoX; see Q3); (c) replacing softmax with something unnormalized or sparse (sigmoid, stick-breaking, α-entmax, Diff attention). None of these is aimed at latent thought tokens.

### Cited Findings

**Root theory**
- Hahn, M. (2020). "Theoretical Limitations of Self-Attention in Neural Sequence Models." TACL. arXiv:1906.06755. Claim: with both soft and hard attention, self-attention cannot model periodic finite-state languages or hierarchical structure unless the number of layers or heads grows with input length. This is the lemma that Chiang & Cholak build on. **[motivates]** (abstract only) — [arXiv:1906.06755](https://arxiv.org/abs/1906.06755)
- Chiang, D. & Cholak, P. (2022). "Overcoming a Theoretical Limitation of Self-Attention." ACL 2022. arXiv:2202.12172. Claims (full text checked):
  - Under Hahn's lemma, the classification confidence of a transformer on languages like FIRST decays toward 1 bit of cross-entropy as strings get longer. Layer norm can push cross-entropy arbitrarily close to zero. — [arXiv:2202.12172](https://arxiv.org/abs/2202.12172)
  - When a model must "focus on a single position" (FIRST), it fails to generalize from short to long strings. The remedy is "multiplying attention logits by log n", which restores perfect test cross-entropy and accuracy at every training length. — [arXiv:2202.12172](https://arxiv.org/abs/2202.12172)
  - English–Vietnamese MT, training on ≤20 tokens and testing on >20 tokens: log-n scaling raises BLEU from 11.4 to 12.4 (p<0.01). With matched train/test lengths it does nothing (32.6 vs 32.5). — [arXiv:2202.12172](https://arxiv.org/abs/2202.12172)
  - **[motivates / competes]**. This is the canonical "log-n scaling" baseline. It is global (all keys), not per-entry.
- Veličković, P., Perivolaropoulos, C., Barbero, F., Pascanu, R. (2024/2025). "Softmax is not Enough (for Sharp Size Generalisation)." ICML 2025. arXiv:2410.01104. Claims (full text checked):
  - Lemma 2.1 (softmax must disperse): if the logits are bounded in [m, M], then as n→∞ every coefficient is Θ(1/n). Concretely, softmax_θ(e)_k ≤ (1/n)·exp(δ/θ), where δ = M − m is the logit spread. — [arXiv:2410.01104](https://arxiv.org/abs/2410.01104)
  - Theorem 2.2: in transformers over a finite vocabulary the logits are bounded in every layer, so dispersion "must happen everywhere in a Transformer for sufficient problem sizes". Even for max-finding, "any learned circuitry must disperse as the number of items grows at test time." — [arXiv:2410.01104](https://arxiv.org/abs/2410.01104)
  - Prop. 3.1: sharpness requires large weight norms, so setting θ=0 on a trained model is problematic. Their fix is *adaptive temperature*: an inverse degree-4 polynomial maps the entropy of the attention distribution to θ, applied at inference time only and never raising θ above 1. It is streamable and compatible with FlashAttention. — [arXiv:2410.01104](https://arxiv.org/abs/2410.01104)
  - **[supports / competes]**. Lemma 2.1 is the formal version of our "attention on the first 6 latents falls as steps are added". Adaptive temperature is a training-free baseline the gate should beat.
- Chen, S., Lin, Z., Polyanskiy, Y., Rigollet, P. (2025). "Critical attention scaling in long-context transformers." ICLR 2026. arXiv:2510.05554. Claims (full text checked):
  - Without scaling, attention scores collapse toward uniform as n grows, which produces rank collapse (tokens cluster to a single direction). — [arXiv:2510.05554](https://arxiv.org/abs/2510.05554)
  - With logits multiplied by β_n there is a phase transition. Scaling that is too weak collapses all tokens; scaling that is too strong turns attention into the identity. The critical value is β_n ≍ log n. — [arXiv:2510.05554](https://arxiv.org/abs/2510.05554)
  - Their Table 1 lists the β_n each method uses: YaRN (log n)², Qwen log n, SSMax log n, SWAN-GPT log n. — [arXiv:2510.05554](https://arxiv.org/abs/2510.05554)
  - **[motivates]**. It also warns that over-sharpening has a failure mode of its own.

**Entropy-invariant and length-scaled softmax**
- Su, Jianlin (2021). "从熵不变性看Attention的Scale操作" ("The Scale Operation of Attention from the Perspective of Entropy Invariance"), kexue.fm blog, Dec 2021. Author blog. **I could not fetch it (HTTP 403).** The derivation is cited as the basis for InfoScale (Li et al. 2025, co-authored by Su), whose comparison table lists a log_512(n)·QKᵀ/√d_k scaling. So the formula here is as reported secondhand in that table. **[motivates]** — [kexue.fm/archives/8823](https://kexue.fm/archives/8823); [arXiv:2501.08570](https://arxiv.org/abs/2501.08570)
- Li, K., Kong, Y., Xu, Y., Su, J., et al. (2025). "Information Entropy Invariance: Enhancing Length Extrapolation in Attention Mechanisms." arXiv:2501.08570. Claims:
  - InfoScale is a training-free scaling for dot-product attention, and CosScale is its counterpart for cosine attention. Together they reach state of the art on GAU-α at 64× the training length.
  - The paper names "attention score dilution" by newly added tokens as "a key challenge", and notes that a large CosScale approximates windowed attention.
  - **[motivates / competes]** (abstract plus grep of full text) — [arXiv:2501.08570](https://arxiv.org/abs/2501.08570)
- Nakanishi, K. M. (2025). "Scalable-Softmax Is Superior for Attention." arXiv:2501.19399, a single-author preprint (venue not stated on arXiv). Claims (full text checked):
  - It names the problem "attention fading": the max softmax element goes to 0 as n grows because the denominator grows while the numerator does not. — [arXiv:2501.19399](https://arxiv.org/abs/2501.19399)
  - SSMax: z_i ↦ exp((s·log n)·z_i) / Σ_j exp((s·log n)·z_j), with s a learnable scalar per layer and head. The form came from fitting learned per-length multipliers p_n, which turned out to follow p_n ≈ a₁ log n + a₂. — [arXiv:2501.19399](https://arxiv.org/abs/2501.19399)
  - With SSMax, attention stays focused as long as z_max exceeds the other logits by about 1/s, whatever n is. — [arXiv:2501.19399](https://arxiv.org/abs/2501.19399)
  - Models swapped to SSMax mid-training or after pretraining still recover part of the length-generalization benefit. — [arXiv:2501.19399](https://arxiv.org/abs/2501.19399)
  - **[competes]**. This is the strongest global-scaling baseline. It sharpens all keys, and it does not care which keys are "extra".
- Peng, B., Quesnelle, J., Fan, H., Shippole, E. (2023). "YaRN: Efficient Context Window Extension of Large Language Models." arXiv:2309.00071. Claims (full text checked):
  - Besides NTK-by-parts interpolation, YaRN puts a temperature t on the logits: softmax(qᵀk / (t√|D|)). For LLaMA/Llama 2 the recommended value is √(1/t) = 0.1·ln(s) + 1, where s is the context-extension factor. For example, s=8 gives ≈1.208.
  - The value came from fitting the perplexity-minimizing t across extension factors, and the authors call this "the property of increased entropy".
  - **[competes / motivates]** — [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
- Anson, B., Wang, X., Aitchison, L. (2025). "Scale-invariant Attention." NeurIPS 2025. arXiv:2505.17083. Claims (full text checked):
  - It proposes two conditions for long-context attention: "scale-invariant total attention" and "scale-invariant attention sparsity".
  - Under a Gaussian-logit assumption, a position-dependent affine transform L_t ↦ a_t·L_t + m_t achieves them. Both a_t and m_t depend on log(t/τ+1), with τ≈10 working best.
  - It improves zero-shot short-to-long validation loss and long-context retrieval, and is compared against "LogN" scaling.
  - **[competes]**. The additive m_t term is a position-dependent logit bias, close in form to a log-gate.
  - [arXiv:2505.17083](https://arxiv.org/abs/2505.17083)
- Vasylenko, P., Pitorro, H., Martins, A. F. T., Treviso, M. (2025). "Long-Context Generalization with Sparse Attention." ICLR 2026. arXiv:2506.16640. Claims (full text checked):
  - It formally defines "attention dispersion". Softmax entropy grows to Θ(log n), while α-entmax keeps entropy bounded at O(log s), where s is the number of nonzero entries.
  - ASEntmax(z) = α-entmax((δ + β(log n)^γ)·z), with a learnable temperature.
  - Reported results: 1000× length extrapolation on synthetic associative recall (65K tokens after training on 64), and better LM perplexity and retrieval at 8× training length than softmax, SSMax, and fixed-temperature entmax.
  - **[competes]**. Exact zeros are a "hard drop" learned inside the attention itself.
  - [arXiv:2506.16640](https://arxiv.org/abs/2506.16640)

**Mechanisms that let a model ignore unneeded keys**
- Leviathan, Y., Kalman, M., Matias, Y. (2024). "Selective Attention Improves Transformer." ICLR 2025. arXiv:2410.02703. Claims (full text checked):
  - "Unneeded elements in the attention's context degrade performance." — [arXiv:2410.02703](https://arxiv.org/abs/2410.02703)
  - Mechanism: a token can *mask* earlier tokens. The steps are:
    - A masking matrix S comes from an existing head, so the method is parameter-free.
    - S is passed through ReLU, so masking can only reduce attention, never boost it.
    - The <BOS> column and the diagonal are zeroed out.
    - S is accumulated by cumulative sum, applied only to later queries, giving F.
    - F is *subtracted from the attention logits*: softmax(QKᵀ/√d − F).
    
    The intuition is that a token masks another "after absorbing its contents". — [arXiv:2410.02703](https://arxiv.org/abs/2410.02703)
  - Results: on C4 it matches standard transformers with about 2× more heads and attention parameters. With pruning, contexts of 512, 1,024 and 2,048 need 16×, 25× and 47× less attention memory at the same perplexity. — [arXiv:2410.02703](https://arxiv.org/abs/2410.02703)
  - **[threatens novelty (mechanism) / competes]**. This is the closest published mechanism to a learned soft log-gate that drops latent entries. An extra latent that has "absorbed" an earlier one could mask it this way.
- Ye, T., Dong, L., Xia, Y., Sun, Y., Zhu, Y., Huang, G., et al. (2024). "Differential Transformer." ICLR 2025 (oral). arXiv:2410.05258. Claims (full text checked):
  - Transformers "overallocate attention to irrelevant context", which the paper calls "attention noise". Its fix computes attention as the difference of two softmax maps, with a learnable λ = exp(λq1·λk1) − exp(λq2·λk2) + λ_init.
  - It reports gains in long-context modeling, key-information retrieval, and hallucination.
  - **[competes]**. It is an architecture change that requires pretraining.
  - [arXiv:2410.05258](https://arxiv.org/abs/2410.05258)
- Tan, S., Yang, S., Courville, A., Panda, R., Shen, Y. (2024). "Scaling Stick-Breaking Attention: An Efficient Implementation and In-depth Study." ICLR 2025 (per header in PDF). arXiv:2410.17980. Claims (full text checked):
  - Mechanism: β_{i,j} = σ(q_jᵀk_i/√d). The weight for key i is β_{i,j} multiplied by the "remaining stick" ∏(1−β) over keys closer to the query.
  - There is no normalization over all n keys and no positional embedding. The result is a built-in recency bias.
  - A model trained at 2¹¹ context "perform[s] well at 2¹⁴".
  - **[competes]**. Caveat for latents: its recency bias would favor the *latest* (extra) latents, which is probably the wrong direction for us.
  - [arXiv:2410.17980](https://arxiv.org/abs/2410.17980)
- Ramapuram, J., Danieli, F., Dhekane, E., Weers, F., Busbridge, D., Ablin, P., et al. (2024). "Theory, Analysis, and Best Practices for Sigmoid Self-Attention." arXiv:2409.04431. Claims (full text checked):
  - Sigmoid attention is elementwise, with no sum-to-one constraint. It needs a bias b = −log(n) on the logits to stay stable and to "make sense of SigmoidAttn for any sequence length". With that bias, LayerScale, and QK-norm, it matches softmax at 1B scale.
  - FlashSigmoid gives a 17% inference-kernel speedup over FlashAttention-2 on H100.
  - **[motivates]**. Even without normalization, the community converges on an additive −log n correction. That is the additive mirror of our log(k+1) duplication identity.
  - [arXiv:2409.04431](https://arxiv.org/abs/2409.04431)

### Inferences
- **The duplication identity is the same algebra as SSMax and log-n scaling.** Under softmax, k+1 identical keys with logit z contribute (k+1)e^z = e^{z+log(k+1)}. So "duplicate one latent k times" is exactly "add log(k+1) to its logit", as the brief states. The literature treats the opposite correction (−log n additive, as in sigmoid; ×log n multiplicative, as in Chiang/SSMax) as the principled fix. This gives our proposed log-gate an analytic target in the duplicate case: g_j = −log(cluster size).
- **Global sharpening may not be enough for CODI's extra latents.** SSMax, YaRN, Chiang & Cholak, and adaptive temperature all multiply every logit. They help only when the useful keys already have *higher* logits than the added ones (SSMax: focus holds when z_max exceeds the rest by about 1/s). If the extra latents are near-duplicates of the trained latents (see the attractor evidence in Q5), their logits will be similar, and sharpening will not separate them. A per-entry gate (Selective-attention style) can. This is the cleanest way to differentiate our idea from the log-n family.
- **Arithmetic check on the reported ~4× drop.** If the latent block keeps roughly constant total mass, and latents 7–24 get logits about equal to latents 1–6, then the share of latents 1–6 falls from 6/6 to 6/24, which is exactly a 4× drop. That fits Lemma 2.1's Θ(1/n) dispersion *within the latent block*. It is suggestive only: it assumes the prompt and sink mass stay fixed.
- The critical-scaling result (too much scaling turns attention into the identity) means any learned gate or temperature should be regularized or bounded. Selective attention's constraints (no boosting, never mask BOS or the sink, never mask self) are a ready-made template.

### Gaps
- I could not fetch Su's blog (403). The exact formula and date come from the InfoScale citation and table, not the primary post.
- The SSMax paper is a single-author arXiv preprint; I did not find a peer-reviewed venue.
- I did not check whether Veličković et al.'s adaptive temperature or SSMax have been evaluated on reasoning tasks where the extra keys are model-generated (as opposed to longer inputs).

---

## Q2. Attention sinks (StreamingLLM, Gu et al., gated attention), and what they mean for a gate over latent entries

### Takeaway
Sinks are how softmax models *park* attention mass they do not want to spend. Gu et al. show that a sink behaves like a learned key bias, and that sinks disappear when softmax normalization is removed. Gated attention (Qiu et al., NeurIPS 2025 best paper, deployed in Qwen3-Next) shows that an output gate removes the need for sinks and improves long-context extrapolation, because the model no longer relies on a fragile sink mass that shifts under length changes. For our idea this has two consequences. First, a gate over latents should never touch the sink or BOS entry. Second, a learned "null" logit (as in gpt-oss or off-by-one) is a cheap alternative that caps how much mass goes to latents. It does not fix how mass is split among the latents themselves.

### Cited Findings
- Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. (2023). "Efficient Streaming Language Models with Attention Sinks." ICLR 2024. arXiv:2309.17453. Claims:
  - Window attention fails once the text exceeds the cache, but keeping the KV of the first few tokens "largely recover[s]" performance, because those tokens act as sinks "even if they are not semantically important".
  - StreamingLLM runs Llama-2, MPT, Falcon, and Pythia stably up to 4M tokens, with up to 22.2× speedup over sliding-window recomputation. A dedicated placeholder sink token added in pretraining helps further.
  - **[motivates]** (abstract only) — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- Gu, X., Pang, T., Du, C., Liu, Q., Zhang, F., Du, C., et al. (2024). "When Attention Sink Emerges in Language Models: An Empirical View." ICLR 2025 (Spotlight). arXiv:2410.10781. Claims (full text checked):
  - Sinks appear in all LMs tested, including small ones, and they emerge in pretraining after effective optimization on enough data. — [arXiv:2410.10781](https://arxiv.org/abs/2410.10781)
  - "Attention sink acts more like key biases, storing extra attention scores, which could be non-informative and not contribute to the value computation." It "(at least partially) stems from tokens' inner dependence on attention scores as a result of softmax normalization." — [arXiv:2410.10781](https://arxiv.org/abs/2410.10781)
  - Adding learnable key biases k* (with v* = 0) moves the sink from the first token onto the bias, and "LMs need key biases". Sigmoid attention without normalization grows no sinks up to 1B parameters. With normalization, sinks re-emerge even under sigmoid. — [arXiv:2410.10781](https://arxiv.org/abs/2410.10781)
  - **[supports / motivates]**. It suggests a design: a learned key-bias or null slot that latent-readout heads can dump mass into.
- Qiu, Z., Wang, Z., Zheng, B., Huang, Z., Wen, K., Yang, S., et al. (2025). "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free." arXiv:2505.06708. NeurIPS 2025 Best Paper per [Alibaba's announcement](https://www.alibabacloud.com/blog/alibaba-qwen-wins-%E2%80%9Cneurips-2025-best-paper-award%E2%80%9D-for-breakthrough-in-attention-mechanisms_602708) and the [authors' repo](https://github.com/qiuzh20/gated_attention). Claims (full text checked):
  - Setup: over 30 gating variants on 15B MoE and 1.7B dense models trained on 3.5T tokens. The best is a head-specific sigmoid gate on the SDPA output (position "G1"), which "consistently improves performance", stabilizes training, and tolerates larger learning rates. — [arXiv:2505.06708](https://arxiv.org/abs/2505.06708)
  - Why it works: (1) it adds non-linearity on top of the low-rank value/output mapping; (2) the gate scores are query-dependent and sparse (mean SDPA-output gate score is 0.116). — [arXiv:2505.06708](https://arxiv.org/abs/2505.06708)
  - Sink: the baseline sends on average 46.7% of attention to the first token, and the gated model sends 4.8%. In one baseline layer the first-token share drops from 83% to 4%. — [arXiv:2505.06708](https://arxiv.org/abs/2505.06708)
  - Long context: after training at 32k and extending to 128k with YaRN, gated models "outperform the baseline significantly" at 64k and 128k. The text claims "a gain of over 10 points on RULER". The authors' hypothesis is that baseline models "rely on attention sinks to adjust the distribution of attention scores", which breaks when the context changes. — [arXiv:2505.06708](https://arxiv.org/abs/2505.06708)
  - Deployed in Qwen3-Next-80B-A3B per the [authors' repo](https://github.com/qiuzh20/gated_attention) and the [Alibaba blog](https://www.alibabacloud.com/blog/alibaba-qwen-wins-%E2%80%9Cneurips-2025-best-paper-award%E2%80%9D-for-breakthrough-in-attention-mechanisms_602708).
  - **[competes / motivates]**. Note that the G1 gate scales the *head output*, not individual keys, so it cannot re-weight latents relative to one another.
- OpenAI (2025). "gpt-oss-120b & gpt-oss-20b Model Card." arXiv:2508.10925. Claim (full text checked): "Each attention head has a learned bias in the denominator of the softmax, similar to off-by-one attention and attention sinks, which enables the attention mechanism to pay no attention to any tokens." **[competes]** It is a learned null logit. — [arXiv:2508.10925](https://arxiv.org/abs/2508.10925)
- Miller, E. (2023). "Attention is off by one." Blog, cited by gpt-oss and by Gu et al. as softmax-off-by-one. Clearly a blog. **[motivates]** — [evanmiller.org](https://www.evanmiller.org/attention-is-offby-one.html) (not fetched; cited via [arXiv:2508.10925](https://arxiv.org/abs/2508.10925))
- Zuhri, Z. M. K., Fuadi, E. H., Aji, A. F. (2025). "Softpick: No Attention Sink, No Massive Activations with Rectified Softmax." arXiv:2504.20966. Claim: softpick is a rectified, not-sum-to-one replacement for softmax that gives a 0% sink rate at 340M and 1.8B parameters and sparse attention maps. **[competes]** (abstract only) — [arXiv:2504.20966](https://arxiv.org/abs/2504.20966)
- Coconut attention example: when analyzing which tokens Coconut's answer attends to on ProsQA, Cui et al. (2026) "exclude the top-1 attended token, which consistently acts as an attention sink". The latent-reasoning models studied there do have sinks. **[supports]** (full text checked; 2026 preprint) — [arXiv:2602.22441](https://arxiv.org/abs/2602.22441)

### Inferences
- **A gate over latents should leave the sink alone.** Selective attention explicitly never masks <BOS>, and Gu et al. show the sink is load-bearing. If CODI-GPT-2's readout parks mass on BOS, part of the "attention falls 4×" effect may be mass moving between the sink and the latents, not only within the latents. That is worth measuring separately: sink share versus latent-block share versus within-latent share, as functions of the number of steps.
- **What sink-based fixes do and do not cover.** A learned null logit or off-by-one is a cheap one-parameter-per-head baseline. It lets the readout put less total mass on the latent block as the block grows, but it cannot restore the *relative* weight of latents 1–6 over latents 7–24. Our per-entry gate targets the relative problem. The output gate in gated attention targets neither directly.
- Gated attention's explanation (sinks are a fragile way to control mass and break under length shifts) maps onto the latent-step setting: the readout tuned its sink and latent mass balance for exactly 6 latents.

### Gaps
- I found no study of sink behavior in CODI or COCONUT readout as a function of the number of latent steps.
- I did not verify the exact RULER numbers in gated attention's Table 5. The ">10 points" figure is the paper's own summary.

---

## Q3. Forget gates, ALiBi, and other learned additive biases on attention logits (closest in mechanism to a soft log-gate)

### Takeaway
The soft log-gate we propose, logit_ij + log g_j with g_j ∈ (0,1], is a known form. FoX adds Σ log f_l (a data-dependent cumulative log-decay). ALiBi is the data-independent special case. Selective attention subtracts a learned, accumulated, non-negative mask. The mechanism by itself is therefore **not novel**. What may be new is (i) using it on latent-thought KV entries, (ii) letting the gate depend on a latent's redundancy or position relative to the *trained* depth rather than on distance, and (iii) targeting test-time depth extrapolation.

### Cited Findings
- Lin, Z., Nikishin, E., He, X. O., Courville, A. (2025). "Forgetting Transformer: Softmax Attention with a Forget Gate." ICLR 2025. arXiv:2503.02130. Claims (full text checked):
  - Each timestep gets a scalar forget gate f_t = σ(w_fᵀx_t + b_f). Attention becomes softmax(QKᵀ + D), where D_ij = Σ_{l=j+1..i} log f_l. That is a data-dependent, non-positive additive bias that "down-weight[s] the unnormalized attention scores". — [arXiv:2503.02130](https://arxiv.org/abs/2503.02130)
  - ALiBi is the special case with a data-independent gate f_t = exp(−m_h). FoX needs no positional embeddings, is compatible with FlashAttention, and beats the Transformer on long-context LM and length extrapolation, while matching it on long-context downstream tasks. — [arXiv:2503.02130](https://arxiv.org/abs/2503.02130)
  - **[threatens novelty (mechanism) / competes]**.
- Lin, Z., Obando-Ceron, J., He, X. O., Courville, A. (2025). "Adaptive Computation Pruning for the Forgetting Transformer." COLM 2025. arXiv:2504.06949. Claim: many FoX heads forget quickly. Pruning dependencies that the gate has strongly decayed, with a provably safe threshold, removes about 70% of attention FLOPs with "no performance degradation". **[competes]** This turns a soft log-gate into a hard drop. (abstract only) — [arXiv:2504.06949](https://arxiv.org/abs/2504.06949)
- Press, O., Smith, N. A., Lewis, M. (2021). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." ICLR 2022. arXiv:2108.12409. Claim: ALiBi biases query–key scores with a penalty proportional to distance. A 1.3B model trained at 1024 extrapolates to 2048 with the same perplexity as a sinusoidal model trained at 2048, while training 11% faster with 11% less memory. **[competes]** (abstract only) — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Leviathan et al. (2024), Selective attention (see Q1): softmax(QKᵀ/√d − F), where F is a ReLU'd, cumulative, learned (head-reused) mask that cannot boost attention and never masks BOS or self. **[threatens novelty (mechanism)]** — [arXiv:2410.02703](https://arxiv.org/abs/2410.02703)
- Anson et al. (2025), Scale-invariant attention (see Q1): a position-dependent additive m_t plus multiplicative a_t on the logits. **[competes]** — [arXiv:2505.17083](https://arxiv.org/abs/2505.17083)
- Gu et al. (2024): learnable key biases k* act as an always-present extra logit slot (see Q2). **[competes]** — [arXiv:2410.10781](https://arxiv.org/abs/2410.10781)

**Learned keep/drop over KV entries for reasoning (efficiency-motivated, but sometimes accuracy-improving)**
- Łańcucki, A., Staniszewski, K., Nawrot, P., Ponti, E. M. (2025). "Inference-Time Hyper-Scaling with KV Cache Compression." NeurIPS 2025. arXiv:2506.05345. Claims:
  - DMS (Dynamic Memory Sparsification) learns which tokens to evict, with delayed eviction. It reaches 8× compression after 1K training steps, with better accuracy than training-free sparse attention.
  - It improves Qwen-R1 32B by 12.0 points on AIME 24 at an equal number of memory reads.
  - **[competes]** It is a learned drop gate on reasoning KV. (abstract only) — [arXiv:2506.05345](https://arxiv.org/abs/2506.05345)
- Huang, Y., Yuan, B., Han, X., Xiao, C., Liu, Z. (2024). "Locret: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads." arXiv:2410.01805. Claim: learnable "retaining heads" score the causal importance of KV units for eviction, giving up to 20× compression with <10% loss and costing <1 GPU-hour of training. **[competes]** (abstract only) — [arXiv:2410.01805](https://arxiv.org/abs/2410.01805)
- Gao, Y., et al. (2025). "SeerAttention-R: Sparse Attention Adaptation for Long Reasoning." arXiv:2506.08889. Claim: a lightweight, self-distilled gating plug-in learns attention sparsity for long reasoning decodes and is near-lossless on AIME with a 4K token budget. **[competes]** (abstract only) — [arXiv:2506.08889](https://arxiv.org/abs/2506.08889)

### Inferences
- **Map of the design space.** FoX's log f_l gate is cumulative over *intervening* tokens, so it encodes recency. Selective attention's F is cumulative over *masking* tokens, so it encodes "has this been absorbed". A latent log-gate could be:
  - (a) per-entry and static: log g_j with g_j from the latent's own hidden state;
  - (b) Selective-attention style: later latents mask earlier ones they have absorbed;
  - (c) FoX style: decay with latent index.
  
  Option (c) would actively *hurt* if, as in CODI, the useful information sits in the early latents and later latents drift. Options (a) and (b) are the plausible novel variants. They should be framed as "applying additive logit gating (cf. FoX, Selective attention) to latent thought entries", not as a new attention mechanism.
- **Hard drop beats soft gate on compute.** The DMS, Locret, and SeerAttention-R results show learned drop gates can be trained cheaply (1K steps; <1 GPU-hour). That makes a cheap post-hoc fine-tune of only a gate on a frozen CODI model plausible.

### Gaps
- I found no paper that applies FoX, Selective attention, or ALiBi-style biases to continuous-thought KV entries specifically (see Q5).
- I did not check whether Selective attention's masking survives when sequences extend past training length. Its experiments vary context size but train at each size.

---

## Q4. Length-generalization failures attributed to attention dispersion (lost in the middle, temperature fixes, over-squashing, position coupling)

### Takeaway
The evidence pointing to dispersion is substantial:
- NoPE models fail exactly when attention flattens, and a single temperature fixes it.
- Llama 4 ships an SSMax-style temperature on its NoPE layers (per vLLM's code).
- Redundant-token pruning *raises* reasoning accuracy.

There is also a serious **threat**. Du et al. (2025) show that length hurts even when the extra tokens are *masked out of attention*. That points to positional distance, not dilution. Because CODI on GPT-2 uses learned absolute position embeddings, extra latent steps also shift the readout's positions. The duplication experiment helps separate the two, but a masked-latent control is needed.

### Cited Findings
- Wang, J., Ji, T., Wu, Y., Yan, H., Gui, T., Zhang, Q., et al. (2024). "Length Generalization of Causal Transformers without Position Encoding." arXiv:2404.12224. Claims (full text checked):
  - NoPE's length-generalization failures "always co-occur with the distraction of attention distributions: the attention heads begin to allocate their weights to tokens evenly".
  - Scaling attention logits by 1.2 (λ from 1/√d to 1.2/√d), with all parameters frozen, extends a 2K-trained NoPE model to over 4K. Head-specific temperature search extends it further.
  - The same trick does *not* move the entropy inflection point for RoPE models.
  - **[supports / competes]** — [arXiv:2404.12224](https://arxiv.org/abs/2404.12224)
- Llama 4 "attention temperature tuning": vLLM's Llama 4 implementation applies an inference-time, position-dependent query scaling on NoPE layers only. The code comments cite arXiv:2501.19399 (SSMax), and the defaults are floor_scale 8192 and attn_scale 0.1. This is code-level evidence from search-result summaries of vLLM docs; I did not check Meta's documentation. **[supports]** — [vLLM llama4 source docs](https://docs.vllm.ai/en/v0.10.1/api/vllm/model_executor/models/llama4.html)
- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." TACL. arXiv:2307.03172. Claim: performance is highest when relevant information sits at the beginning or end of the context and drops significantly in the middle, even for long-context models. **[motivates]** (abstract only) — [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
- Hsieh, C.-Y., Chuang, Y.-S., Li, C.-L., Wang, Z., Le, L. T., Kumar, A., et al. (2024). "Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization." ACL Findings 2024. arXiv:2406.16008. Claim: lost-in-the-middle traces to a U-shaped positional attention bias that holds "regardless of their relevance". Calibrating that bias out improves RAG by up to 15 points. **[competes]** It is a calibration-style correction on attention. (abstract only) — [arXiv:2406.16008](https://arxiv.org/abs/2406.16008)
- Barbero, F., Banino, A., Kapturowski, S., Kumaran, D., Araújo, J. G. M., Vitvitskyi, A., et al. (2024). "Transformers need glasses! Information over-squashing in language tasks." arXiv:2406.04267. Claims (full text checked):
  - Theorem 4.2 (informal): take a sequence and the same sequence with its last token repeated. If positional information decays with distance, their final-token representations "become arbitrarily close as n increases". Low-precision formats make this worse. — [arXiv:2406.04267](https://arxiv.org/abs/2406.04267)
  - Separately, there is "over-squashing": earlier tokens have many more paths to the final representation. — [arXiv:2406.04267](https://arxiv.org/abs/2406.04267)
  - **[supports]**. Repeated or near-duplicate tokens provably wash out distinctions, which is the representation-level twin of the duplicate-KV logit effect.
- Vasylenko et al. (2025), ASEntmax (Q1): "as sequence length increases, non-informative tokens accumulate attention probability mass, leading to dispersion and representational collapse" (abstract). **[supports]** — [arXiv:2506.16640](https://arxiv.org/abs/2506.16640)
- Cho, H., Cha, J., Awasthi, P., Bhojanapalli, S., Gupta, A., Yun, C. (2024). "Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure." NeurIPS 2024. arXiv:2405.20671. Claim: giving the same position ID to "relevant" tokens (digits of equal significance) lets models trained on 1–30-digit addition generalize to 200 digits (6.67×). **[competes]** It is an alternative fix for the *positional* part of the problem, for example giving extra latents re-used position IDs. (abstract only) — [arXiv:2405.20671](https://arxiv.org/abs/2405.20671)
- Du, Y., Tian, M., Ronanki, S., Rongali, S., Bodapati, S., Galstyan, A., et al. (2025). "Context Length Alone Hurts LLM Performance Despite Perfect Retrieval." Findings of EMNLP 2025. arXiv:2510.05381. Claims (full text checked):
  - Across 5 LLMs, performance drops 13.9%–85% as input length grows, even with perfect retrieval. — [arXiv:2510.05381](https://arxiv.org/abs/2510.05381)
  - The drop persists when distractors are whitespace, and when all distractor tokens are *masked*, so the model attends only to evidence and question and the only remaining difference is distance. With 30K masked tokens the drop is at least 7.9% for both Llama and Mistral, and Llama3 on HumanEval drops 50% with masking versus 19.4% with whitespace. — [arXiv:2510.05381](https://arxiv.org/abs/2510.05381)
  - Their mitigation is to have the model recite the evidence before answering. — [arXiv:2510.05381](https://arxiv.org/abs/2510.05381)
  - **[threatens]** the pure-dilution explanation. It shows that positional distance alone can degrade performance.
- Choi, D., Lee, J., Tack, J., Song, W., Dingliwal, S., Jayanthi, S. M., et al. (2025). "Think Clearly: Improving Reasoning via Redundant Token Pruning." Findings of EMNLP 2025. arXiv:2507.08806. Claims (abstract only):
  - In long reasoning traces, "attention scores are widely scattered", and incorrect answers show this more.
  - Evicting redundant reasoning tokens, chosen by their attention from an injected end-of-thinking token, then resuming generation "significantly improves" accuracy without training. Gains are largest on AIME and AMC.
  - **[supports / competes]** Removing reasoning-token KV improves accuracy by "removing distraction", but for explicit CoT.
  - [arXiv:2507.08806](https://arxiv.org/abs/2507.08806); [ACL Anthology](https://preview.aclanthology.org/new-sigs/2025.findings-emnlp.1169/)
- Cai, Z., et al. (2025). "R-KV: Redundancy-aware KV Cache Compression for Reasoning Models." arXiv:2505.24133. Claim: with 16% of the KV cache it reaches 105% of full-cache performance, so dropping redundant reasoning tokens can *exceed* full attention. **[supports]** (abstract only) — [arXiv:2505.24133](https://arxiv.org/abs/2505.24133)
- Liu, M., et al. (2025). "Hold Onto That Thought: Assessing KV Cache Compression On Reasoning." arXiv:2512.12008. Claims: H2O and a decoding-enabled SnapKV dominate for reasoning models, and eviction at low budgets can lengthen reasoning traces. **[motivates]** (abstract only) — [arXiv:2512.12008](https://arxiv.org/abs/2512.12008)

### Inferences
- **Separating dilution from position in CODI, using the masking control from Du et al.:**
  - (1) Run 24 latent steps but mask latents 7–24 from the answer readout. If accuracy returns to the 6-step level, dilution is the cause. If it does not, positional shift or latent-state drift is.
  - (2) Run 24 steps but give the answer tokens the position IDs they would have at 6 steps. This works because GPT-2 uses learned absolute position embeddings, so extra latents push the answer tokens to positions never seen at that offset during training.
  - (3) The existing duplication experiment holds position fixed if duplicates are inserted at the *same* position ID. Confirm that this was how it was implemented, otherwise duplication also shifts positions.
- The NoPE result (temperature fixes NoPE but not RoPE) suggests the dilution fix works best when position is not the binding constraint. That is another reason to run the positional control before claiming dilution.
- Think Clearly and R-KV are explicit-CoT analogues of our claim: removing extra reasoning entries from attention can *increase* accuracy. They are the most citable "supports" for the premise that a drop gate over thought tokens can help, beyond saving compute.

### Gaps
- The Llama 4 temperature formula comes from search-result summaries of vLLM code. I did not verify the exact expression or Meta's stated rationale.
- I did not open the Lost-in-the-Middle or Found-in-the-Middle full texts.
- I did not verify the venue of Transformers-need-glasses; it is not in the arXiv metadata.

---

## Q5. Has any of this been applied to latent or continuous thought tokens, pause tokens, or recurrent steps?

### Takeaway
**I found no paper that (a) diagnoses the failure of CODI or COCONUT past the trained number of latent steps as softmax dilution over latent KV entries, or (b) applies log-n scaling, temperature, sinks, or learned additive log-gates to latent thought entries.** The nearest work:
- SIM-CoT: training with more latents collapses them into homogeneous states. The paper blames weak supervision, not attention.
- The 2026 dynamical-systems analysis: CODI's latents behave as a "stable attractor", which makes near-duplicates plausible.
- Coconut attention analyses: the answer mostly attends to the question, with a sink at the top.
- Learned stop and halting policies for latent reasoning: they choose *how many* latents to make, not which to attend to.
- Explicit-CoT KV pruning and compression (LightThinker, PENCIL, DMS, Think Clearly).
- Pause tokens: performance degrades gracefully, not sharply, when more pauses are used at inference than in training.

The application angle looks open. The mechanism does not.

### Cited Findings
**Latent CoT models and their behavior as the number of latents grows**
- Shen, Z., Yan, H., Zhang, L., Hu, Z., Du, Y., He, Y. (2025). "CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation." arXiv:2502.21074 (EMNLP 2025 main per the [ACL Anthology PDF](https://aclanthology.org/2025.emnlp-main.36.pdf)). Claim: the first implicit-CoT method to match explicit CoT on GSM8K at GPT-2 scale, with 3.1× compression. (abstract only) — [arXiv:2502.21074](https://arxiv.org/abs/2502.21074)
- Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., et al. (2024). "Training Large Language Models to Reason in a Continuous Latent Space" (Coconut). COLM 2025. arXiv:2412.06769. Claim: it feeds the last hidden state back as the next input embedding, and continuous thoughts can encode multiple next steps (BFS-like). (abstract only) — [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
- Wei, X., Liu, X., Zang, Y., Dong, X., Cao, Y., Wang, J., et al. (2025). "SIM-CoT: Supervised Implicit Chain-of-Thought." arXiv:2509.20317. Claims (full text checked):
  - Increasing the number of implicit tokens makes training unstable and can make it collapse. Coconut collapses at 8 or 16 implicit tokens. — [arXiv:2509.20317](https://arxiv.org/abs/2509.20317)
  - Failed models' latents become "homogeneous", inter-latent distance falls sharply, and operator information is lost. — [arXiv:2509.20317](https://arxiv.org/abs/2509.20317)
  - The fix is step-level supervision through an auxiliary decoder, giving +8.2% for Coconut on GPT-2 and +3.0% for CODI on LLaMA-3.1-8B. — [arXiv:2509.20317](https://arxiv.org/abs/2509.20317)
  - **[supports / competes]**. Homogeneous latents are, for attention, near-duplicate keys (see the Q1 inference). Note that this is a *training-time* number-of-latents effect, not test-time extrapolation.
- Duraipandian, S. I., Boyane, S. S., Nagesh, M., Francis, J., Vaidheeswaran, A., Zhu, K. (2026). "Interpreting Latent CoT Reasoning as Dynamical Systems." ICML 2026 Workshop (FoDGM). arXiv:2607.09698. **2026 preprint.** Claims (full text grepped):
  - "CODI behaves as a stable attractor, while COCONUT behaves as an unstable expanding system". In CODI, "latent representations converge as timestep" increases.
  - SIM-CoT supervision tightens both behaviors without changing their dynamics.
  - **[supports]**. If CODI's extra steps converge to an attractor, they are near-duplicate KV entries, exactly the case our duplication experiment simulates.
  - [arXiv:2607.09698](https://arxiv.org/abs/2607.09698)
- Cui, Y., Dai, Z., He, B., et al. (2026). "How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision?" arXiv:2602.22441. **2026 preprint.** Claims (full text checked):
  - Latent reasoning methods show "pervasive shortcut behavior".
  - In a Coconut ProsQA example, after excluding the top-1 attended token ("consistently acts as an attention sink"), the top-10 attended tokens for each answer token come "exclusively" from the input question, not from latent tokens. A GSM8K example shows a different pattern.
  - **[motivates]**. The readout may not depend on latents in the way assumed. Measure this for CODI.
  - [arXiv:2602.22441](https://arxiv.org/abs/2602.22441)
- Zhang, Y., Tang, B., Ju, T., Duan, S., Liu, G. (2025). "Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought." EMNLP 2026 Main (per arXiv comment). arXiv:2512.21711. Claim: COCONUT tokens show minimal sensitivity to steering and act as "uninterpretable placeholders", promoting shortcuts. CODI is not tested. **[motivates / threatens premise]** (abstract only) — [arXiv:2512.21711](https://arxiv.org/abs/2512.21711)
- Liu, W., Min, D., Cheng, L. (2026). "Latent Thoughts Tuning." ICML 2026 (per arXiv comment). arXiv:2602.10229. **2026 preprint.** Claim: current latent paradigms "suffer from feature collapse and instability due to distribution mismatch when recurrently reusing hidden states". It proposes Context-Prediction-Fusion. **[supports]** (abstract only) — [arXiv:2602.10229](https://arxiv.org/abs/2602.10229)

**Pause and filler tokens with more steps at test time than in training**
- Goyal, S., Ji, Z., Rawat, A. S., Menon, A. K., Kumar, S., Nagarajan, V. (2023). "Think before you speak: Training Language Models With Pause Tokens." ICLR 2024. arXiv:2310.02226. Claims (full text checked):
  - Setting inference pauses M_inf ≠ M_ft (the fine-tuning pause count) is "a severe test-time distribution shift". Even so, with M_ft = 10, pause-trained models degrade "gracefully" across M_inf ∈ [5, 25].
  - Zero inference pauses breaks the pretrained-and-finetuned pause model "spectacularly".
  - There is a task-specific optimal number of pauses.
  - **[threatens premise partially / motivates]**. Pause tokens share one learned embedding, so extra pauses are close to duplicates, yet degradation is mild. CODI's sharper drop may therefore come from latent-state drift and not only from count dilution.
  - [arXiv:2310.02226](https://arxiv.org/abs/2310.02226)
- Pfau, J., et al. (2024). "Let's Think Dot by Dot: Hidden Computation in Transformer Language Models." arXiv:2404.15758. Filler tokens can carry hidden computation. I did not find a dilution analysis in it. **[background]** (title and metadata only) — [arXiv:2404.15758](https://arxiv.org/abs/2404.15758)

**Recurrent and looped alternatives that do not grow the KV cache per step**
- Geiping, J., et al. (2025). "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach." arXiv:2502.05171. It scales test-time compute by iterating a recurrent block in depth, so extra iterations do not append sequence-dimension KV entries the way Coconut and CODI steps do. A search-summary source says Huginn "reuses a bounded set of recurrent cache entries"; I did not verify this in the paper. **[competes]** (abstract-level) — [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)
- Fan, Y., Svete, A., Lee, K. (2026). "Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers" (LOTUS). arXiv:2606.31779. **2026 preprint.** Claim: a looped padded transformer processes K latent blocks in parallel for R iterations, so the KV count is fixed by K while depth grows with R. It is the first latent-CoT method to close the gap to explicit CoT at 3B. **[competes]** (abstract and intro read). Note: our project memory says the LOTUS baseline is paused. — [arXiv:2606.31779](https://arxiv.org/abs/2606.31779)
- Fan, Y., et al. (2024). "Looped Transformers for Length Generalization." ICLR 2025. arXiv:2409.15647. **[background]** (metadata only) — [arXiv:2409.15647](https://arxiv.org/abs/2409.15647)

**Learned policies over how many latents to make, or which thoughts to keep**
- Ning, A., Kuo, Y.-L., Gomes, G. (2025). "Learning When to Stop: Adaptive Latent Reasoning via Reinforcement Learning." arXiv:2511.21581. Claim: post-SFT RL on Llama 3.2 1B with GSM8K-Aug cuts total latent reasoning length by 52% "with no penalty to accuracy". **[competes]** It is a stop gate rather than an attention gate. (abstract only) — [arXiv:2511.21581](https://arxiv.org/abs/2511.21581)
- Zeng, B., et al. (2026). "Pretraining with Token-Level Adaptive Latent Chain-of-Thought." arXiv:2602.08220. **2026 preprint.** Claim: variable-length latent CoT per token with token-wise adaptive halting, which improves perplexity with fewer FLOPs. A search summary says it uses a router with continuation probabilities and KV reuse; not verified. **[competes]** (abstract only) — [arXiv:2602.08220](https://arxiv.org/abs/2602.08220)
- Xie, H., Liu, J., Qiao, Z., Vanschoren, J. (2026). "Selective Latent Thinking: Adaptive Compression of LLM Reasoning Chains." arXiv:2605.25745. **2026 preprint.** Claim: confidence-based gating decides which spans to compress to latents. Compressing arbitrary spans cuts Qwen3-4B GSM-Hard accuracy from 54.7% to 17.3%, while keeping numerical tokens explicit keeps 51.8%. **[adjacent]** (abstract only) — [arXiv:2605.25745](https://arxiv.org/abs/2605.25745)
- Amos, I., Caciularu, A., Geva, M., Globerson, A., Herzig, J., Shani, L., et al. (2026). "Latent Reasoning with Supervised Thinking States." arXiv:2602.08332. **2026 preprint.** Claim: thinking tokens generated every few input tokens are added back into the embeddings, and the method extrapolates to longer sequences on state tracking. **[adjacent]** (abstract only) — [arXiv:2602.08332](https://arxiv.org/abs/2602.08332)

**Explicit-thought KV compression and erasure (the closest "drop thought entries" precedents)**
- Zhang, J., et al. (2025). "LightThinker: Thinking Step-by-Step Compression." EMNLP 2025 (oral). arXiv:2502.15589. Claim: it compresses thought steps into gist tokens and *discards the original thought tokens via specialized attention masks*, reducing peak memory and latency "while maintaining competitive accuracy". **[competes / threatens novelty (thought-KV dropping)]** (abstract only) — [arXiv:2502.15589](https://arxiv.org/abs/2502.15589)
- Yang, C., Srebro, N., McAllester, D., Li, Z. (2025). "PENCIL: Long Thoughts with Short Memory." ICML 2025. arXiv:2503.14337. Claim: a learned reduction mechanism recursively erases intermediate thoughts, and a 25M model with 2048 context solves Einstein's puzzle. **[competes]** (abstract only) — [arXiv:2503.14337](https://arxiv.org/abs/2503.14337)
- Yan, Y., et al. (2025). "InftyThink." ICLR 2026. arXiv:2503.06692. Claim: interleaving reasoning segments with summaries bounds context, with 3–11% gains for Qwen2.5-Math-7B. **[competes]** (abstract only) — [arXiv:2503.06692](https://arxiv.org/abs/2503.06692)
- DMS (arXiv:2506.05345), Think Clearly (arXiv:2507.08806), and R-KV (arXiv:2505.24133): see Q3 and Q4. They are all on explicit tokens.

### Inferences
- **Novelty position.** "Latent-step extrapolation failure in CODI is (partly) softmax dilution over latent KV entries, shown causally by a duplication or log-bias intervention; fixed by a learned per-entry log-gate over latent keys" does not appear in anything I found. The two parts should be pitched separately:
  - (i) The *diagnosis*, with the duplication = +log(k+1) identity as a causal probe. This looks open.
  - (ii) The *fix*, which must be framed against FoX, Selective attention, SSMax, and DMS as prior mechanisms.
- **Baselines a reviewer will ask for**, all cheap and mostly training-free:
  - (a) SSMax or log-n temperature on readout-to-latent logits only;
  - (b) Veličković-style adaptive temperature;
  - (c) hard mask of latents beyond the trained depth;
  - (d) a window keeping only the last 6 latents;
  - (e) a learned null or sink logit (gpt-oss style);
  - (f) the positional controls from Q4.
  
  If (c) alone recovers accuracy, the "learned gate" has to show it beats a fixed mask, for example by choosing *which* latents to keep when the useful ones are not simply the first 6.
- **Competing explanation.** The pause-token result (graceful degradation up to 2.5× trained pauses) is a natural comparison point. Pauses are identical input embeddings at different positions, so they are a "pure count" condition. If CODI degrades much more sharply, latent-state drift (CODI's attractor and odd-even cycle, per our earlier notes) is a competing explanation that the dilution story has to rule out or include.
- SIM-CoT's homogenization and the CODI "stable attractor" finding together give a mechanism by which extra steps become near-duplicates. Through the softmax identity, that turns directly into a log-count boost for the attractor state's mass. That is a clean bridge from the dynamics literature to the attention-dilution literature, and I did not see it made explicitly anywhere.

### Gaps
- The search for "latent thoughts plus attention temperature, gating, or sinks" was limited to a few web searches and arXiv lookups. Neither found a direct precedent, but a targeted OpenReview search (ICLR 2026, NeurIPS 2025, ICML 2026) and a forward-citation check of SIM-CoT and the dynamical-systems paper are still needed.
- No source I found measures readout attention over latents as a function of test-time step count for CODI or COCONUT. The "CODI attention is diffuse and bypasses latents" claim surfaced only in a search-tool summary, and I could not trace it to a primary source, so it is excluded from the findings.
- MUX (arXiv:2607.18264, 2026) shows a "published" date of 2026-05-19 in arXiv metadata that does not match its 2607 ID. I treated it as unverified and did not use it.
- I did not verify the Huginn KV-cache behavior or the router details of 2602.08220 beyond their abstracts and search summaries.
