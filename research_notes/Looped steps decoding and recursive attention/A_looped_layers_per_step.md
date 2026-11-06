# Idea A: looping a block of layers inside each latent CoT step (CODI / SIM-CoT GPT-2), so extra serial compute adds no KV entries

Notes compiled 2026-11-05. They build on, and do not repeat, `research_notes/Latent thought attention gating/looped_depth_extrapolation.md` and `novelty_threats.md`. Findings marked **[prior notes]** come from those files and were not re-fetched today. The other tags are:
- **[main text]**: I read the arXiv HTML. Usually this was only the first ~100k characters, and the finding says so where it matters.
- **[abstract]**: I saw only the abstract page.
- **[snippet]**: from a search-engine summary only. Treat these as unverified.

Every 2026 item is an arXiv preprint unless a venue is given. Project measurements are cited by local file path.

Two corrections to the brief:
- **PCCoT.** "PCCoT trained GPT-2 from scratch with 12 latents and got 48.24" is wrong on all three counts. In PCCoT's Table 1, **48.24 ± 1.61 is the "Continuous CoT" baseline row**. PCCoT itself scores **49.48 ± 0.31 with c = 24 latents and T = 3 Jacobi iterations**, and it is **LoRA-fine-tuned from pretrained GPT-2**, not trained from scratch. The c sweep, including 12 vs 6, appears only in Figures 2–3 with no numbers in the text. — [PCCoT HTML](https://arxiv.org/html/2506.18582)
- **LOTUS venue.** A search summary called LOTUS "ICLR 2026". Its arXiv id is 2606 (June 2026) and v2 is dated 13 Jul 2026, so the venue claim is probably wrong. I treat LOTUS as a 2026 preprint.

---

## Q1. Which papers loop layers (or the whole model) per token or per latent step of a pretrained or retrofitted LM, and what math gains did they get?

### Takeaway
Retrofitting loops into a pretrained LM does work on math. GSM8K gains of +5 to +12 points are reported:
- McLeish et al.: TinyLlama 46.2 → 52.0.
- ETD: OLMo-2-1B 44.05 → 56.56.
- Think-at-Hard: Qwen3-0.6B 58.2 → 63.6 / 67.6.

The large gains came with **tens of billions of training tokens** (McLeish, ETD), or with a **selective** loop that iterates only about 7% of tokens (TaH). **Iterating every token twice and then fine-tuning made GSM8K worse by about 6 points** (TaH's AlwaysThink baseline). That is the single most important caution for idea A.

The only looped *latent-CoT* result on our exact setting is LOTUS (GPT-2, GSM8K-Aug): 44.1 vs CODI 42.0 and SIM-CoT 42.6. It uses parallel latent blocks with whole-model looping, not loops inside sequential CODI steps.

### Cited Findings

**Retrofitting recurrence into pretrained LMs (closest in recipe to "fine-tune a published checkpoint")**

- **McLeish et al., "Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence"**, arXiv 2511.07384 v1, 10 Nov 2025. No venue listed. Code: github.com/mcleish7/retrofitting-recurrence. **[main text, first 100k chars]** — [arXiv HTML](https://arxiv.org/html/2511.07384v1)
  - Models: TinyLlama-1.1B (3T tokens), OLMo-2-0425-1B and Llama-3.2-1B.
  - Layer layout:
    - TinyLlama is (prelude, recurrent, coda) = (4,8,4): layers 0–3, then 10–17, then 18–21. Layers 4–9 are **dropped**, leaving about 700M parameters.
    - OLMo and Llama are (4,6,4), with layers 4–5 dropped.
    - The layout came from a small search. It beat ShortGPT-style pruning on training loss.
  - Injection: concatenate the prelude output e with the recurrent state s, then apply a linear 2h→h adapter. s₀ is random noise. An additive adapter gave higher training loss but similar eval accuracy.
  - Recurrence count is sampled from a Poisson-lognormal with a curriculum. The mean ramps to 32 over the first 75% of training on a 1-sqrt schedule. Backprop runs through only the last 8 block applications.
  - Optimizer: Muon. AdamW diverged to NaN on the recurrent models.
  - Data: about 50B tokens of Nemotron-CC-Math, on AMD MI300A.
  - **GSM8K (Table 1), test recurrence 1 → 32:** (4,8,4) single-phase 25.6 → **52.0**; two-phase 26.5 → 51.2. The static (non-recurrent) TinyLlama post-trained on the same data scores **46.2** / 45.2.
  - MATH: 8.8 → 14.5, against 14.4 static.
  - In Table 2, gains "largely level off between 16 and 32 recurrences." Recurrences past 32 were not tested, and the authors list deeper extrapolation as open.
  - A "healing" phase on FineWeb-Edu adds about 5 points of Arc-C. The authors attribute this to repairing the distribution shift that layer removal causes.
- **Koishekenov, Lipani, Cancedda (FAIR/UCL), "Encode, Think, Decode (ETD)"**, arXiv 2510.07358. Listed on iclr.cc/virtual/2026 per a search result; the v1 HTML names no venue. **[main text]** — [arXiv HTML](https://arxiv.org/html/2510.07358v1); [ICLR listing](https://www.iclr.cc/virtual/2026/10016653)
  - Layer choice: average angular distance between layer inputs on C4, with Kneedle knee detection to set the encoder and decoder boundaries. For OLMo-2-1B (16 layers) this gives **7-4×k-5**.
  - The thinking block is applied repeatedly with **no input injection described** and **no KV handling described**.
  - Training: integrated into OLMo-2 mid-training, 5×10¹⁰ tokens. Each k is a separate model.
  - **GSM8K:** k=1 44.05, k=2 51.10, k=3 54.36, k=4 55.50, **k=5 56.56**.
  - MATH is noisy and non-monotone: 4.57 → 6.22 at k=3, then 3.73 at k=4 and 4.33 at k=5.
  - At matched FLOPs, the selected middle block beat both "loop all layers" (0-16×2-0) and a broad middle loop (2-12×2-2) on most categories. Among block positions, 7-4×2-5 was best (math 28.27, vs 26.8 for 5-4×2-7 and 25.6 for 1-4×2-11).
  - Adaptive depth: a per-token sigmoid router accumulates halting mass, stops at 1−ε (ε=0.01), and allows at most 10 iterations. It beats no recursion and sometimes beats fixed k=5 with fewer iterations. Exact numbers are in figures only.
- **Think-at-Hard (TaH): "Dynamic Looped Transformers for Improved Reasoning"**, arXiv 2511.08577, **ICML 2026**, v4 28 Aug 2026. **[main text, ~100k chars + tail]** — [arXiv abs](https://arxiv.org/abs/2511.08577); [HTML](https://arxiv.org/html/2511.08577)
  - Base models: Qwen3-Base 0.6B / 1.7B / 4B. The whole backbone is re-run. The next-depth input is a logit-weighted embedding softmax(W_outᵀy)·E, plus residual connections across iterations whose exact form is not given.
  - **Depth-aware LoRA**: depth 1 uses the base weights θ, and depths >1 use θ+Δ.
  - **Duo-causal attention**: the token at (i, d) attends to {x_j^(k): j ≤ i, k ≤ d}. Each depth keeps its own KV cache, the caches are flattened into one sequence, and position ids use the original token index. It is implemented with an additive mask and needs no custom kernels.
  - A decider MLP over shallow, middle and final hidden states (threshold 0.9) iterates about 7% of tokens. The maximum is 2 iterations; ablations go to 3–4, but I did not extract those numbers.
  - Training: 100K Open-R1 Mixture-of-Thoughts examples, 5 epochs, lr 4e-5, batch 128. Stage 1 trains backbone + LoRA at oracle depth; stage 2 trains the decider.
  - **GSM8K (Table 2):**
    - 0.6B: Standard 58.2, **AlwaysThink 52.6**, TaH 63.6, TaH+ 67.6.
    - 1.7B: 80.2, **74.2**, 84.8, 84.5.
    - 4B: 91.7, AlwaysThink OOM, 91.0, 91.7.
  - **MATH500:** 0.6B 39.9 / 37.8 / 46.2 / 51.8; 1.7B 67.8 / 63.2 / 71.4 / 72.6.
  - "Latent overthinking": a second pass fixes 8.7% of first-pass predictions but turns 2.1% of correct ones wrong.
  - In injected-error tests (Table 23), overthinking dominates the accuracy loss. The fitted model is acc = 0.81 − 1.41·underthink − 2.73·overthink.
  - AlwaysThink decode cost is 2.27× / 2.19× Standard.
- **Relaxed Recursive Transformers** (Bae et al., ICLR 2025). Pretrained Gemma is converted to a recursive model with layer-wise LoRA. A **recursive Gemma 1B uptrained on 15B tokens beats full-size TinyLlama and Pythia**. Full-rank LoRA matches the full model. The best init is "Average" looped layers plus SVD-initialized LoRA. No GSM8K numbers found. **[snippet]** — [arXiv PDF](https://arxiv.org/pdf/2410.20672); [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/54d6a55225cebbdc16fbb0e45c5bdf2b-Abstract-Conference.html)
- **T2MLR, "Transformer with Temporal Middle-Layer Recurrence"** (Cai, Zhu, Dong, He, Arora), arXiv 2607.15178, Jul 2026. **[abstract]** — [arXiv abs](https://arxiv.org/abs/2607.15178)
  - It caches a middle-layer representation of the previous token and feeds it into an earlier layer at the current position, making it a recurrence across tokens.
  - "Recurrence on as little as 20% of the network often beats full-layer recurrence."
  - Adding the pathway to a **pretrained 1.7B and briefly fine-tuning improves math reasoning**. The abstract gives no numbers.
- **LoopUS, "Recasting Pretrained LLMs into Looped Latent Refinement Models"** (Park et al.), arXiv 2605.11011, May 2026. **[abstract]** — [arXiv abs](https://arxiv.org/abs/2605.11011)
  - Post-training split into encoder / looped block / decoder, guided by "staged representation dynamics."
  - Adds an **input-dependent selective gate against hidden-state drift**, **random deep supervision**, and a confidence head for early exit.
  - Models and numbers are not in the abstract.
- **Relit** (Panwar, Singh, Bansal), arXiv 2608.08113, Aug 2026. A frozen TinyLlama-1.1B gets a trainable recursive block with gradient-isolated loops. Evaluated on logic benchmarks (GLoRE, ProofWriter, RuleTaker); GSM8K is not mentioned. **[abstract]** — [arXiv abs](https://arxiv.org/abs/2608.08113)

**Looped / recurrent-depth LMs trained from scratch (architecture references)**
- **Huginn-3.5B** (Geiping et al. 2025). **[main text, first 100k chars]** — [arXiv HTML](https://arxiv.org/html/2502.05171)
  - Layout (2, 4, 2). Injection by concat + adapter (2h→h); "concatenation works best at scale, addition for smaller models." s₀ is truncated normal.
  - **Sandwich RMSNorm** was required. Failed runs either collapsed (token correlation → 1) or "learned to ignore the incoming state" (same perplexity at 1 and 32 recurrences).
  - r is sampled from a log-normal Poisson (mean 32, σ=½). Backprop runs through only the last 8 iterations.
  - GSM8K: r=32 8-shot CoT 34.80/42.08 (flexible/strict, with system prompt); EMA model at r=64 gives 47.23 flexible.
- A search snippet attributed to Huginn reports GSM8K-CoT of 9.02/10.24 for an early recurrent checkpoint at 32 steps vs 1.82/2.20 for a fixed-depth baseline, with gains "plateaued quickly" from 4 to 256 steps. **[snippet]** — [arXiv PDF](https://arxiv.org/pdf/2502.05171)
- Lu et al. probe Huginn and find limited interpretable latent CoT. A snippet gives explicit CoT 24.87% vs best latent 4.62% at 256 recurrences. **[snippet]** — [arXiv 2507.02199](https://arxiv.org/pdf/2507.02199)
- **Ouro / LoopLM** (2510.25741). Trained at T=4. GSM8K 78.92 for the 1.4B with full caches. Past T=4 it degrades (AIME24 64.7 → 39.0 at T=8). **[prior notes]** — [arXiv](https://arxiv.org/abs/2510.25741)
- **Mixture-of-Recursions** (Bae et al., NeurIPS 2025). Trained from scratch with a token router over recursion depth. A 167M model with 2 recursions (expert-choice) reaches 43.1% average few-shot vs 42.3% for a 315M vanilla model. Recursive KV sharing gives 41.9%. **[snippet / prior notes]** — [arXiv](https://arxiv.org/abs/2507.10524); [NeurIPS](https://neurips.cc/virtual/2025/poster/118085)
- **Inner Thinking Transformer** (Chen et al.), arXiv 2502.13842, Feb 2025.
  - Components: adaptive token routing, **residual thinking connections** (accumulating each iteration's output) and **thinking step encoding**.
  - A 162M model reaches 96.5% of a 466M Transformer, uses 43.2% less data, and beats Transformer/Loop variants on 11 benchmarks. No GSM8K reported. **[abstract]** — [arXiv HTML](https://arxiv.org/html/2502.13842v2)
- **PonderLM** (Zeng et al.), **ICLR 2026**, arXiv 2505.20674.
  - Several forward passes per token; each feeds back a probability-weighted mix of embeddings.
  - Learned by self-supervised pretraining on GPT-2, Pythia and LLaMA.
  - PonderPythia-2.8B beats Pythia-6.9B. No GSM8K numbers found. Code: github.com/LUMIA-Group/PonderingLM. **[snippet]** — [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/94572a66c6b0710c5263f15d8ef6b463-Abstract-Conference.html); [arXiv HTML](https://arxiv.org/html/2505.20674v3)
- **CoTFormer** (Mohtashami, Pagliardini, Jaggi), arXiv 2310.10845, v2 Aug 2024 ("…Budget-Adaptive Computation Cost at Inference").
  - It argues that how past tokens are attended "distinguishes CoT from simply running the model several times." Repeats are interleaved as tokens, so later passes attend to earlier passes. Token-wise variable depth.
  - A secondary summary says a 24-layer CoTFormer with interleave 2 beats a 48-layer Transformer. Venue not confirmed. **[abstract / snippet]** — [arXiv abs](https://arxiv.org/abs/2310.10845)

**2026 looped-model papers named in the brief** (detail in prior notes)
- **Kuo, Chayti, Reizinger, Brendel, Jaggi**, arXiv 2606.29983, **NeurIPS 2026**. Randomized loop counts K = clip(n+Δ) with Δ ~ U{±5} raise OOD addition from 22.1 to 30.9 and cut the std from 9.9 to 5.9. RL-Halt reaches 45.0. ±10 is worse than ±5. Algorithmic tasks only. **[prior notes]** — [arXiv](https://arxiv.org/abs/2606.29983)
- **Viakhirev et al., "Think Shallow, Solve Deep,"** arXiv 2608.18222. Settling, marginal and drifting regimes, plus a terminal fixed-point loss. Huginn LoRA "latent anchoring" restores stability, but "no treated model exceeds the base's peak competence." **[prior notes]** — [arXiv](https://arxiv.org/abs/2608.18222)
- **LOTUS** (Fan, Svete, Lee), arXiv 2606.31779 v2, 13 Jul 2026. See Q2 for numbers. **[main text, first 100k chars]** — [arXiv HTML](https://arxiv.org/html/2606.31779)
- **Blayney et al.**, arXiv 2604.11791. Cyclic fixed points per layer. Input injection is needed for non-degenerate fixed points. Stable residual stream ⇒ stable attention (Prop. 4.2). **[prior notes]** — [arXiv](https://arxiv.org/abs/2604.11791)
- **InfiLoop**, arXiv 2610.11570. A learned-decay residual over past loop states keeps improving to about 25k layer evaluations on Sudoku-Extreme. With no decay, accuracy is 38–40% vs 82–90% with decay. **[prior notes]** — [arXiv HTML](https://arxiv.org/html/2610.11570)
- **RecurTrace**, arXiv 2609.03379. A gated, recency-biased window over a layer's own past loop outputs; W=3 is best. Loops are sampled from a clamped log-normal Poisson (mean about 3.9, clamped at 8). At 16 loops, accuracy is a stable 47.5% vs a 54.7% two-loop peak. **[prior notes]** — [arXiv HTML](https://arxiv.org/html/2609.03379)
- **STARS**, arXiv 2605.26733, ICML 2026: Jacobian spectral-radius regularization plus random loops. **LoopFormer**, arXiv 2602.11451, ICLR 2026: shortcut-consistency conditioning on time and step size. **Parcae**, arXiv 2604.12946: constrained injection; test-time returns saturate. **[prior notes]** — [STARS](https://arxiv.org/abs/2605.26733); [LoopFormer](https://arxiv.org/abs/2602.11451); [Parcae](https://arxiv.org/abs/2604.12946)
- **"Loop, Think, & Generalize"** (Kohli, Parthasarathy, Sun, Yao), arXiv 2604.07822, **COLM 2026**. Recurrent-depth models trained from scratch on implicit multi-hop reasoning extrapolate from 5 to 10 hops when inference recurrence grows. "Overthinking, where excessive recurrence degrades predictions," limits the deepest compositions. No numbers in the abstract. **[abstract]** — [arXiv abs](https://arxiv.org/abs/2604.07822)
- **LatentLoop**, arXiv 2610.19975 (per its repo). A looped MoE pretrained from scratch that runs the middle half twice. LM benchmarks only. Peripheral. **[repo page]** — [GitHub](https://github.com/lcqysl/LatentLoop)

**Project baselines (local, CODI GPT-2, gsm8k_valid n=500)** — `results/latent_gate/stage1/codi/report.txt`
- Published CODI: 206 / 192 / 173 at 6 / 12 / 24 steps.
- ft6 (answer-only fine-tune, 32,768 examples): 209 / 201 / 180.
- ft12: 200 / 210 / 203.
- So 12 steps of serial compute trained in gives 210, against 209 for ft6 at 6 steps: **zero gain from extra steps**.

### Inferences
- **The gain profile is "big with lots of data, small or negative with a short fine-tune."** McLeish and ETD used about 50B tokens. TaH used about 100K long examples × 5 epochs and *selective* depth. Our budget is about 10–30M tokens (Q5), which is three to four orders of magnitude below the retrofits that gained 6–12 GSM8K points.
- **TaH's AlwaysThink result is the closest analogue to "loop every latent step k times."** It is the strongest piece of negative prior evidence: iterating all positions hurt by 5.6–6.0 GSM8K points after the same fine-tune. Idea A differs in two ways that may matter: only 6 latent positions are looped, and the answer tokens are not. TaH looped answer tokens, where overthinking flips correct predictions.
- **The positive evidence is about depth inside a single position helping math** (McLeish r=1 → 32: 25.6 → 52.0; ETD k=1 → 5: +12.5). This is the mechanism idea A needs. None of it was measured in a latent-CoT model.

### Gaps
- McLeish appendix tables 3–5 (exact OLMo and Llama-1B GSM8K numbers) were in the unread last 43k characters.
- TaH's ablations for max iterations 3–4, no-LoRA, no-residual and duo-causal vs same-depth KV are in parts of the page I did not extract.
- No GSM8K numbers exist for Relaxed Recursive, MoR, ITT, PonderLM or CoTFormer. They are pretraining or LM-benchmark papers.
- ETD does not say how it handles the KV cache across iterations.

---

## Q2. Has anyone combined looped or recurrent depth with continuous latent CoT (Coconut, CODI, SIM-CoT, PCCoT, CoLaR)? Closest competitors

### Takeaway
I found **no paper that loops a block of layers inside each *sequential* CODI/Coconut-style latent step**, writing one KV entry per step. That also means nobody tests "depth inside a step vs more steps" at matched compute in an appended-latent model. But looping for latent CoT is **not new**:
- **LOTUS** (2606.31779) loops the whole model over parallel latent blocks on exactly our setting (GPT-2 + Llama-3.2-1B/3B, GSM8K-Aug) and beats CODI and SIM-CoT.
- **PCCoT** iterates (Jacobi) the full model over a fixed latent set.
- **Penelope** (2607.25915) confines recurrence to a decoder interval of a pretrained model for latent reasoning. It is the closest in spirit and I read only its abstract.
- **TaH, ETD and McLeish** loop within a token of a pretrained LM, but not in latent CoT.

A reviewer will read idea A as "LOTUS / TaH-style looping applied to CODI." Novelty would have to rest on two things: the **controlled steps-vs-depth comparison**, motivated by the dilution diagnosis, and a **KV-neutral inner loop** on sequential latents.

### Cited Findings
1. **LOTUS: Fan, Svete, Lee, "Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers," arXiv 2606.31779 v2 (13 Jul 2026).** This is the closest competitor. **[main text, first 100k chars]** — [arXiv HTML](https://arxiv.org/html/2606.31779)
   - Design: K=6 latent blocks of c learnable `<lat>` tokens, with c=13 on GPT-2 (78 latent positions) and c=25 on Llama. The question is encoded once into a KV cache that is reused.
   - The base model is looped R=6 times, with h⁽ᵗ⁾ = f(E + h⁽ᵗ⁻¹⁾ | C_pre), i.e. **input injection by addition**. Latents are processed **in parallel**, and gradients flow through all R. The fetched text does not say which layers loop.
   - Training data: GSM8K-Aug (385k). Losses: answer CE plus a parallel CE from each post-loop latent through the LM head to the gold CoT-step tokens.
   - **GSM8K accuracy:**

     | Method | GPT-2 | Llama-1B | Llama-3B |
     |---|---|---|---|
     | Explicit CoT | 42.7 | 58.4 | 71.5 |
     | CODI | 42.0 | 52.7 | 60.8 |
     | CODI + SIM-CoT | 42.6 | 56.1 | 62.3 |
     | **LOTUS** | **44.1 ± 0.7** | 57.3 ± 0.2 | 70.0 ± 0.9 |
     | LOTUS + CODI | 43.1 ± 1.2 | 58.3 ± 0.8 | 70.6 ± 0.2 |

   - "LOTUS + CODI" adds only CODI's distillation loss, **not CODI's sequential latent decoding**.
   - Test-time R for a model trained at R=6 (3B): 22.7 / 40.0 / 55.0 / 63.5 / 68.7 / **70.0** / 69.3 for R = 1–7.
   - A search summary says that a Coconut-style regime inside LOTUS's looped architecture underperforms both LOTUS variants. Not checked in the text. **[snippet]** — [arXiv PDF](https://arxiv.org/pdf/2606.31779)
   - A search-result paper describes CODI as decoding "only a single latent per step," faster than looped LOTUS but less accurate. **[snippet]** — [arXiv HTML](https://arxiv.org/html/2606.31779v2)
2. **PCCoT** (Wu, Teng, Tu, EMNLP 2025). **[main text]** — [arXiv HTML](https://arxiv.org/html/2506.18582)
   - A fixed set of c=24 latents is overwritten across T=3 Jacobi iterations of the full model. LoRA on pretrained GPT-2; only the special-token embeddings are otherwise trainable.
   - GPT-2 GSM8K: PCCoT **49.48 ± 0.31**, Continuous CoT 48.24 ± 1.61, CODI 43.7 (from the literature), CoT 44.1.
   - **[prior notes]** Iterations past about T=6 "fluctuate heavily."
3. **Penelope: "Localized Latent Recurrence for Efficient Structured Reasoning"** (Chen et al.), arXiv 2607.25915, 28 Jul 2026. **[abstract]** — [arXiv abs](https://arxiv.org/abs/2607.25915)
   - Aimed at pretrained decoder-only models. Recurrence is confined to "a chosen decoder interval," so the full decoder does not rerun.
   - The lower prefix runs once to build a "problem-conditioned boundary memory." That memory is refined by time-modulated GRU dynamics and recurrent readout states, with a progressive CoT-to-latent curriculum.
   - Reported as "competitive with established latent-reasoning models" at lower latency. The abstract names no base model, benchmark or numbers.
   - **Must be read before claiming novelty.** It is the nearest "localized recurrence for latent reasoning in a pretrained decoder."
4. **TaH** (ICML 2026; Q1). It loops inside a token of a pretrained LM with depth-aware LoRA and duo-causal KV. It is not latent CoT, but a reviewer will cite it for the mechanism. — [arXiv abs](https://arxiv.org/abs/2511.08577)
5. **ETD / McLeish / LoopUS / T2MLR / Relaxed Recursive** (Q1). These retrofit loops into pretrained LMs at the token level. None involves Coconut, CODI or SIM-CoT latents. — [ETD](https://arxiv.org/html/2510.07358v1); [McLeish](https://arxiv.org/abs/2511.07384); [LoopUS](https://arxiv.org/abs/2605.11011); [T2MLR](https://arxiv.org/abs/2607.15178)
6. **Huginn latent-CoT probing** (Lu et al., COLM 2025 workshop) finds "limited evidence of interpretable latent CoT" in a recurrent-depth LM. This concerns recurrent depth *as* latent CoT, not combined with it. **[prior notes]** — [arXiv](https://arxiv.org/abs/2507.02199)
7. **Other adjacent work**
   - LLoCoT (2610.11472): a fixed latent workspace refined by a looped transformer, code tasks only. **[prior notes]** — [arXiv](https://arxiv.org/abs/2610.11472)
   - AGCLR (2606.07720): gated memory inside a Coconut loop on GPT-2, GSM8K 34.0 vs 31.4. **[prior notes]** — [arXiv](https://arxiv.org/html/2606.07720)
   - "Trading Depth for Time in Recurrent Transformers" (2609.21605): thought tokens (time) vs loops (depth) at matched FLOPs. One latent thought token between vocabulary tokens overtakes a two-loop transformer as compute grows. **[prior notes, abstract]** — [arXiv](https://arxiv.org/abs/2609.21605)
   - Zhang (2605.30757): loops add compute but no persistent memory, while appended CoT tokens add memory. **[prior notes]** — [arXiv](https://arxiv.org/abs/2605.30757)
   - "Interpreting Latent CoT Reasoning as Dynamical Systems" (2607.09698, ICML 2026 FoGen workshop): CODI behaves "like a stable attractor," COCONUT "like an unstable expanding system," and SIM-CoT tightens both. Analysis only, no looping. — [arXiv abs](https://arxiv.org/abs/2607.09698)
   - "Prioritize the Process…" (2602.10520) rewards latent thought trajectories in looped LMs with RL. Title only. **[snippet]** — [arXiv PDF](https://arxiv.org/pdf/2602.10520)
8. **Searches that returned no direct combination:**
   - "looped transformer layers inside continuous thought latent CoT Coconut CODI recurrent depth"
   - "recurrent depth within each latent thought step continuous reasoning GSM8K-Aug 2026"
   - "continuous thought depth recurrence looped latent reasoning combined Coconut deeper latent steps"
   - "CODI latent tokens loop middle layers multiple times per latent step deeper continuous thoughts"
   - One search summary stated outright: "I didn't find a CODI variant with middle-layer looping." — [search results incl. LoopUS/T2MLR](https://arxiv.org/html/2605.11011v1)

### Inferences
- **Novelty is thin on mechanism and real on the question.** "Loop layers to deepen latent reasoning" belongs to LOTUS. "Loop selectively inside tokens of a pretrained LM with LoRA" belongs to TaH and ETD. What is unclaimed is narrower:
  - (i) **Inner loops on sequential, appended latents with KV overwrite**, so depth costs no attention dilution.
  - (ii) The **matched-compute comparison of steps vs depth** (ft12 vs 6 × k=2) in a CODI-style model.
  - (iii) Tying (ii) to the project's measured dilution mechanism: exact KV copies reproduce the drift, and attention on latents 1–6 drops from .158 to .033.

  Item (ii) is publishable even if idea A gives only a small gain, provided it is clean. Note that "Trading Depth for Time" found the opposite preference (time beat depth) in its setting, so a result showing depth beating time in appended-latent CoT would be an informative contrast.
- **LOTUS sets the bar on our exact setting**: GPT-2 at 44.1 against its own CODI re-run at 42.0. Our CODI eval is 206/500 = 41.2% on validation. A looped-CODI that gains +2 points would roughly tie LOTUS. PCCoT's 49.5 is far ahead and probably out of reach for a short fine-tune.
- The project memory records LOTUS GPT-2 work as paused. That does not stop a reviewer from requiring LOTUS's reported numbers as a comparison row.

### Gaps
- I did not read Penelope's or LoopUS's full text. Either could contain a CODI- or Coconut-based experiment with inner recurrence.
- I did not read LOTUS's appendix (characters 100k–197k). It may contain the Coconut and PCCoT comparisons, which layers loop, and whether it initializes from pretrained weights.
- OpenReview and ICLR/ICML 2026 submissions were not searched directly. NeurIPS 2026 submissions are not public.

---

## Q3. Implementation details that matter, and a concrete recipe for CODI GPT-2

### Takeaway
The literature converges on these choices:
1. Loop a **middle block of about 25–50% of the layers**, not the whole model. ETD and T2MLR found this better than whole-model loops at matched FLOPs.
2. **Re-inject the loop-entry state** every iteration, with an adapter or by addition.
3. Use **iteration-specific LoRA**, active only on extra iterations, so iteration 1 stays exactly the base model.
4. Control **norm growth**: sandwich or outer norm, or a damped / decayed update.
5. Train with **randomized loop counts**, moderate randomness rather than heavy.
6. Backprop through only the **last few iterations**.
7. Write **one KV entry per layer per step**, from either the final iteration (Ouro's last-step-only decode was near-lossless) or the first iteration (MoR / LPT first-recursion sharing). Ouro's first-step-only decode collapsed, but that model was trained with all per-step caches. In a retrofit, first-iteration KV is the only option that keeps k>1 exactly equal to CODI at init.
8. Add **adaptive halting**.

For CODI GPT-2 in transformers 4.46.2 all of this is about 60 lines. GPT-2 blocks take a tuple `layer_past` and return a new concatenated `present` without mutating the cache, so "overwrite" just means keeping one chosen iteration's `present` and discarding the rest.

### Cited Findings
**Which layers**
- ETD picks the block by the knee in angular distance between layer inputs. At matched FLOPs, middle 4 of 16 beats looping all 16 and beats a broad middle 12 of 16. Starting the block later (7) beats earlier (1). — [ETD](https://arxiv.org/html/2510.07358v1)
- McLeish: early layers become the prelude, later layers the recurrent block and coda. Dropping layers then healing beat ShortGPT pruning. — [McLeish](https://arxiv.org/html/2511.07384v1)
- T2MLR: "recurrence on as little as 20% of the network often beats full-layer recurrence." — [T2MLR](https://arxiv.org/abs/2607.15178)
- Huginn loops 4 of 8 layers. — [Huginn](https://arxiv.org/html/2502.05171)
- TaH loops the whole backbone, but only on ~7% of tokens. — [TaH](https://arxiv.org/html/2511.08577)
- Project evidence:
  - The `<eot>` attention dilution sits mostly in layers 0–6 (memory note simcot-past-depth).
  - Plain logit lens reads CODI's latent values mainly in late layers. **[prior notes]** — [LessWrong](https://www.lesswrong.com/posts/yetmqjCXEav9iaJGw/what-s-in-codi-s-latent-recurrent-thoughts)

**Input injection / recall**
- Huginn: concat [s, e] then adapter 2h→h. Concat is best at scale; addition is fine for small models. — [Huginn](https://arxiv.org/html/2502.05171)
- McLeish: concat + linear adapter. "Add" had higher training loss but similar eval. — [McLeish](https://arxiv.org/html/2511.07384v1)
- LOTUS: additive re-injection, h⁽ᵗ⁾ = f(E + h⁽ᵗ⁻¹⁾). — [LOTUS](https://arxiv.org/html/2606.31779)
- Kuo / Fan: H^(t) = Block(H^(t−1) + H^(0)). **[prior notes]**
- Blayney: without injection only pre-norm converges, and every layer reaches the same fixed point. Labovich: recall plus *outer* normalization gives input-smooth fixed points. **[prior notes]** — [Blayney](https://arxiv.org/abs/2604.11791); [Labovich](https://arxiv.org/abs/2604.15259)
- ETD reports none (plain re-application) and still gained. — [ETD](https://arxiv.org/html/2510.07358v1)

**Per-iteration parameters**
- TaH: base weights at depth 1, θ+LoRA at depth >1. — [TaH](https://arxiv.org/html/2511.08577)
- Relaxed Recursive: layer-wise LoRA with SVD init; full rank recovers the full model. — [RRT](https://arxiv.org/pdf/2410.20672)
- ITT: "thinking step encoding" and residual thinking connections. — [ITT](https://arxiv.org/html/2502.13842v2)
- LoopFormer: conditions on time and step size. **[prior notes]**
- Bansal (progressive loss) and Kuo (randomized counts) both aim to stop the net "internally counting iterations." Iteration-index embeddings work against extrapolation. **[prior notes]** — [Bansal](https://arxiv.org/abs/2202.05826)

**Normalization and stability**
- Huginn's failed runs (Q1): collapse without sandwich norm, or the state ignored. Its main run also lowered the LR to 4e-5. — [Huginn](https://arxiv.org/html/2502.05171)
- McLeish: Muon avoided the NaN divergence seen with AdamW. — [McLeish](https://arxiv.org/html/2511.07384v1)
- InfiLoop's damped form h_t = h_{t−1} + γ_t(z_t − h_{t−1}). Guo et al. find a quarter step fixes 72–83% of overshoot failures. FPRM uses pre-norm plus residual scaling. **[prior notes]** — [InfiLoop](https://arxiv.org/html/2610.11570); [Guo](https://arxiv.org/abs/2609.16665); [FPRM](https://arxiv.org/abs/2606.18206)
- LoopUS uses an "input-dependent selective gate to mitigate hidden-state drift." — [LoopUS](https://arxiv.org/abs/2605.11011)

**Randomized vs fixed counts**
- Fixed-depth Ouro (T=4) degrades past training. **[prior notes]**
- Huginn and McLeish use a log-normal Poisson, mean 32, with a curriculum. — [Huginn](https://arxiv.org/html/2502.05171); [McLeish](https://arxiv.org/html/2511.07384v1)
- RecurTrace: mean about 3.9, clamped at 8. **[prior notes]**
- Kuo: ±5 beats both no randomness and ±10. **[prior notes]**
- ETD trains a separate model per k. — [ETD](https://arxiv.org/html/2510.07358v1)

**Truncated backprop**
- Huginn and McLeish: last 8 iterations. — [Huginn](https://arxiv.org/html/2502.05171); [McLeish](https://arxiv.org/html/2511.07384v1)
- Bansal's progressive loss: n iterations without gradient, then k with gradient. **[prior notes]**
- LoopUS: random deep supervision. LOTUS: full backprop through R=6. — [LOTUS](https://arxiv.org/html/2606.31779)

**KV-cache semantics across iterations**
- Huginn caches each recurrence. With early exit, a token attends to the deepest available entry of each earlier token, which works because all recurrences share K/V projections. Zero-shot sharing (slot i mod k) cost little on MTBench. — [Huginn](https://arxiv.org/html/2502.05171)
- Ouro decode with **last-step KV only: GSM8K 78.85 vs 78.92 full; first-step only 18.73**. Prefill needs all caches. **[prior notes]** — [Ouro](https://arxiv.org/abs/2510.25741)
- MoR recursive KV sharing (first recursion only) cost 41.9 vs 42.6 few-shot. **[prior notes]**
- LPT shared memory: the first recursion writes KV; later loops read it plus a short local window. **[prior notes]** — [LPT](https://arxiv.org/abs/2610.02383)
- TaH duo-causal: each depth keeps its own cache; deeper iterations see all shallower ones; position = original index. — [TaH](https://arxiv.org/html/2511.08577)
- LOTUS encodes the question once and reuses the cache across iterations. — [LOTUS](https://arxiv.org/html/2606.31779)
- CoTFormer: later passes attend to earlier passes, which is append-style. — [CoTFormer](https://arxiv.org/abs/2310.10845)

**Adaptive halting**
- ETD: cumulative sigmoid router, ε=0.01, max 10 iterations. — [ETD](https://arxiv.org/html/2510.07358v1)
- TaH: MLP decider (threshold 0.9) imitating an oracle, about 83% accurate. — [TaH](https://arxiv.org/html/2511.08577)
- Huginn: KL < 5e-4 between successive steps. MTBench 5.63 → 5.56. — [Huginn](https://arxiv.org/html/2502.05171)
- Kuo: RL-Halt hazard head. LoopUS: confidence head. FPRM: fixed-point convergence. **[prior notes]**

**Code fit (local).**
- `dsa/bench/models/codi.py` merges CODI's LoRA (r=128, α=32) into FP32 GPT-2 weights. Each latent step calls `self.model(inputs_embeds=latent, past_key_values=cache, ...)` and then `projection(hidden_states[-1])`.
- `scripts/latent_gate/stage1_eval.py` uses explicit `position_ids = question + step` and tuple caches (`cache[0][0].shape[2]`).
- In the pinned transformers 4.46.2, `GPT2Attention` does `key = torch.cat((past_key, key))` and returns `present = (key, value)` without mutating `layer_past`. Checked in `.venv/.../transformers/models/gpt2/modeling_gpt2.py` lines 319–398.

### Inferences

#### Recipe 0: training-free probe (do first; costs only eval)
**"Overwrite-step" CODI.** At each latent position, run the full CODI step (12 layers + projection) r times. Feed `projection(h_last)` back **at the same position id**, and keep only the final iteration's `present` for every layer.
- It needs no new parameters and reuses CODI's trained feedback map. It gives a 6-entry cache at any compute.
- Variants:
  - (a) overwrite at every step, r ∈ {2, 3, 4};
  - (b) overwrite only step 6, i.e. steps 7..N replace entry 6.

  Variant (b) is the KV-neutral counterpart of plain depth 12/24, and of Stage 0's `keep_first` (205 at 24).
- Prediction: about 200–206. Like dedup, it should *preserve* the score; a gain would be surprising. It cheaply tells you whether depth at a position is usable by the frozen readout.
- Implementation: in `stage1_eval.run`, replace the per-step `model(...)` call with a loop that passes the *pre-step* `cache` each time and keeps the last `output.past_key_values`. Works with 4.46.2 tuple caches.

#### Recipe 1: inner block loop, fine-tuned (main proposal)
- **Layout.** Prelude = layers 0–3, loop block B = layers 4–7, coda = 8–11 (4/4/4, ETD- and Huginn-like). Or choose the block by an ETD-style angular-distance knee on CODI's *latent* positions. Ablate 2-layer (T2MLR ~20%) and 6-layer blocks.
- **Step forward** (one latent position, position id p):
  1. `h = latent_emb + wpe[p]`. Run the prelude once, writing KV. Set `e = h`.
  2. Iteration 1: `x_1 = B(e)`, which is exactly CODI.
  3. Iterations r = 2..k: `x_r = x_{r−1} + g ⊙ (B_r(x_{r−1} + W_inj·e) − x_{r−1})`. Here `g` is a learnable per-channel (or scalar) gate initialized to 0 (InfiLoop / ReZero-style damping). `W_inj` is zero-initialized. `B_r` is B with a **depth-aware LoRA** (rank 16–64 on c_attn/c_proj/c_fc, B-matrix zero-initialized) active only for r ≥ 2 (TaH).
  4. Each block call in every iteration passes `layer_past = cache[l]` (the question, earlier latents and nothing from this position). **One `present` per looped layer is kept**, so the step adds exactly 1 KV entry per layer. Which iteration's K/V to keep is a real design choice:
     - **Iteration-1 K/V** (MoR recursive sharing / LPT "first recursion writes"). Default for the first run, because with g=0 it makes every k *exactly* CODI.
     - **Final-iteration K/V** (Ouro last-step). This is *not* identity at init: the final pass feeds layer 4 with x_{r−1} (a layer-7 output), so the stored keys change even when g=0. An exact-init alternative is to blend, K = K_1 + g_kv·(K_k − K_1) with g_kv initialized to 0.
     - Either way, the **coda's** K/V (layers 8–11) are computed from the deepened state, so depth reaches the cache and the next latent through the coda and the projection.
  5. Run the coda once, then `ln_f`, then CODI's projection to make the next latent.

  At init (g=0, LoRA-B=0, W_inj=0, iteration-1 or blended K/V) **every k reproduces the published CODI exactly**. The fine-tune can therefore only move away from the 206 baseline if the loop is useful. This avoids TaH's AlwaysThink drop at the start.
- **Normalization.** GPT-2 is pre-LN, so each pass adds to the residual norm. Either rely on the damped gate, or rescale `x` to ‖h‖ before the coda (outer norm, Labovich). Watch the coda input norm vs k.
- **Positions.** Use the same position id for all iterations (TaH). GPT-2 adds `wpe` once at input, so the block sees no position change.
- **Counts.** Sample k per batch from {1,2,3,4} with P(k=1) ≈ 0.25, a moderate spread per Kuo. Evaluate at k = 1..8. No iteration embedding by default; ablate one (ITT / LoopFormer style).
- **Backprop.** Full BPTT is fine for k ≤ 4 with a 4-layer block on GPT-2. For k up to 8, detach all but the last 3–4 iterations (Huginn / McLeish).
- **Objective.** Answer-only CE (Stage 1) is the cheapest. But Stage 1 already showed answer-only fine-tuning "finds no use" for extra steps, and Huginn's Bad Run 2 shows models learn to ignore recurrence. So add a **per-step signal**:
  - CODI's distillation at the answer colon (needs the teacher pass), and/or
  - SIM-CoT's auxiliary step decoder. Memory notes say the SIM-CoT GPT-2 checkpoint ships one that we currently drop.

  A per-latent step target is the direct pressure for "a deeper latent decodes its step better."
- **KV ablation.** Iteration-1 KV (default) vs final-iteration KV (Ouro) vs blended vs Huginn-style "attend to the deepest available" vs TaH duo-causal. Duo-causal appends k−1 extra entries per latent per looped layer. Near-converged iterations are near-duplicates, so it should **reintroduce dilution**; a useful control.
- **Halting (post hoc).** Per step, stop when ‖x_r − x_{r−1}‖/‖x_{r−1}‖ < τ or the logit-lens KL < 5e-4 (Huginn). Report accuracy vs average k.
- **Batching.** In training, question padding needs the 4-D additive mask passed to each `GPT2Block` call. stage1_train already left-pads questions.

#### Recipe 2: cheaper alternative
Whole-model overwrite loop (Recipe 0) plus depth-aware LoRA and the zero-init gate, trained with random r. This reuses the projection as the injection map. It costs 2× ft6 per extra iteration (Q5), but it is the most direct "steps that don't append" comparison against ft12.

### Gaps
- I found no published ablation of KV policies across iterations (final-only vs per-iteration vs duo-causal) *at fine-tune time* in a retrofit. Ouro's numbers are for decode with a model trained with all caches.
- I found no evidence on whether CODI's *question* KV, computed once without looping, is a good key set for deeper iterations. Huginn's "deepest available" remark and MoR's first-recursion sharing (−0.7 few-shot points) are the only indirect evidence.
- None of the retrofits reports results on 124M-scale models. Whether GPT-2's 12 layers have a usable "middle" for looping is untested.

---

## Q4. Does looping beyond the trained count extrapolate, or does it drift? What enables extrapolation?

### Takeaway
At LM scale, **gains saturate at or slightly past the trained depth, and fixed-depth training degrades past it**:
- LOTUS: 70.0 → 69.3 at R+1.
- Ouro: AIME24 64.7 → 39.0 at 2× T.
- McLeish: levels off between 16 and 32; nothing tested beyond.
- PCCoT: fluctuates past T≈6.
- TaH: "latent overthinking" flips correct tokens.

Extrapolation that keeps *gaining* exists only on algorithmic or puzzle tasks: Bansal (prefix sums, mazes), InfiLoop (Sudoku to about 25k layers), Loop-Think-Generalize (5 → 10 hops). It needs input injection, iteration-agnostic training (randomized counts or progressive loss), and a stability device (fixed-point loss, spectral regularization, decay or damping).

For idea A, the realistic goal is **"accuracy rises with k up to the trained maximum and stays flat after."** "More loops at test time gives more GSM8K" is not a realistic goal.

### Cited Findings
- LOTUS test-time R for a model trained at R=6: 22.7, 40.0, 55.0, 63.5, 68.7, **70.0**, 69.3 (R = 1–7). — [LOTUS](https://arxiv.org/html/2606.31779)
- McLeish: GSM8K rises from 1 to 32 recurrences (training mean 32), "largely level off between 16 and 32," and nothing past 32 was tested. Deeper extrapolation is listed as open. — [McLeish](https://arxiv.org/html/2511.07384v1)
- ETD: each k trained separately; MATH peaks at k=3, then falls below baseline at k=4–5. — [ETD](https://arxiv.org/html/2510.07358v1)
- TaH: a second pass fixes 8.7% of first-pass predictions and breaks 2.1%. Overthinking is the dominant error (coefficient −2.73 vs −1.41 for underthinking). — [TaH](https://arxiv.org/html/2511.08577)
- Ouro (trained at T=4) degrades at T=8: MMLU 67.45 → 64.49; AIME24 64.70 → 39.00. **[prior notes]** — [Ouro](https://arxiv.org/abs/2510.25741)
- Huginn saturates by task: about 8 iterations for HellaSwag, 8–32 for ARC-C. EMA at r=64 gives 47.23 GSM8K. Think Shallow finds collapse on procedural probes (carry 0.69 at r=8 → 0.00 at r≥32). **[prior notes]** — [Huginn](https://arxiv.org/html/2502.05171); [Think Shallow](https://arxiv.org/html/2608.18222)
- PCCoT: "Increasing the number of iterations does not necessarily improve the performance"; latents "stay fluctuating." **[prior notes]** — [PCCoT](https://arxiv.org/html/2506.18582)
- Loop-Think-Generalize (COLM 2026): more inference recurrence enables 5 → 10-hop extrapolation, but "overthinking … limits generalization to very deep compositions." — [arXiv abs](https://arxiv.org/abs/2604.07822)
- Tricks with evidence **[prior notes]**:
  - Recall + progressive loss: Bansal, 32 → 512-bit prefix sums at >97%. — [arXiv](https://arxiv.org/abs/2202.05826)
  - Randomized counts: Kuo, OOD 22.1 → 30.9, std 9.9 → 5.9. — [arXiv](https://arxiv.org/abs/2606.29983)
  - Terminal fixed-point loss: Think Shallow, Sudoku 0.19 → 0.34 past the horizon vs 0.01 → 0.03 without the loss. — [arXiv](https://arxiv.org/html/2608.18222)
  - Jacobian spectral-radius regularization plus random loops: STARS, "substantially mitigates" math degradation. — [arXiv](https://arxiv.org/abs/2605.26733)
  - Learned decay over past loop states: InfiLoop, 82–90% vs 38–40% without decay. — [arXiv](https://arxiv.org/html/2610.11570)
  - Shortcut-consistency: LoopFormer. Readout averaging over the invariant measure: ImR. — [LoopFormer](https://arxiv.org/abs/2602.11451); [ImR](https://arxiv.org/abs/2610.10996)
  - A finite safe depth window exists: Yang et al. 2610.14456. — [arXiv](https://arxiv.org/abs/2610.14456)
- Even successful stabilization at LM scale rarely beats the peak: "No treated model exceeds the base's peak competence at any depth" (Think Shallow on Huginn). **[prior notes]** — [arXiv](https://arxiv.org/html/2608.18222)

### Inferences
- With a KV-neutral inner loop, the project's dilution mechanism is out by construction. What remains is the *state* overthinking that the looped-model literature describes. Inner loops therefore face the classic regime, where the settled/drift theory (Think Shallow, Blayney) *does* apply, because the readout sees one entry per step.
- So inner loops can reuse that literature's stabilizers:
  - a damped gate;
  - randomized k;
  - an optional terminal fixed-point penalty ‖x_k − x_{k−1}‖ at the last iteration.
- Design for "flat past k_train" and claim gains only within the trained range.

### Gaps
- I found no study that runs a looped *retrofit* of a small model (≤ 200M) past its trained count on GSM8K.
- I found no measurement of extrapolation for TaH-style selective loops beyond 2 iterations (ablations unread).

---

## Q5. Rough cost of a short fine-tune of CODI GPT-2 with k loops per step (single Colab A100 or L4)

### Takeaway
The project's own Stage 1 timings anchor this. Answer-only CE, batch 32, bf16, 32,768 examples (1,024 updates):
- **ft6: 436 s**
- **ft12: 683 s**

That makes each extra 12-layer latent forward and backward cost about **41 s per 32k examples**, roughly 40 ms per update. Loops scale this by layer evaluations per step. A 4-layer block at k=4 costs the same as ft12 (about 11 min per 32k examples). The whole model at k=4 is about 20 min per 32k.

A full GSM8K-Aug epoch (385k examples) is about **1.4 h (ft6), 2.2 h (middle-4 × k=4, or whole × k=2), and 3.9 h (whole × k=4)** on the same GPU. Adding CODI's teacher and distillation pass costs more. A short fine-tune of 32k–100k examples per arm is therefore minutes to tens of minutes. A 3-seed × 4-arm study fits in a few A100-hours.

### Cited Findings
- Stage 1 training logs, last line of each `metrics.jsonl`:
  - CODI ft6 436.0 s, ft12 683.4 s.
  - SIM-CoT ft6 439.2 s, ft12 678.3 s.
  - All with 1,024 updates, batch 32, lr 2e-5, bf16, 32,768 records.
  - Local files: `results/latent_gate/stage1/{codi,simcot}/training/{ft6,ft12}/{args.json,metrics.jsonl}`.
  - The GPU model is not recorded in these files. The earlier CODI-recipe run note says A100.
- CODI's own full training: 40 epochs, effective batch 128, lr 3e-3, LoRA r=128 / α=32, bf16, **about 36 h on one A100 80GB**. CODI GPT-2 scores 43.7 after 40 epochs and 38.4 after 20 epochs. — [CODI HTML](https://arxiv.org/html/2502.21074)
- Retrofit token budgets for comparison: about 50B tokens (McLeish, ETD), 15B (Relaxed Recursive Gemma 1B), and 100K long examples × 5 epochs (TaH). — [McLeish](https://arxiv.org/html/2511.07384v1); [ETD](https://arxiv.org/html/2510.07358v1); [RRT snippet](https://arxiv.org/pdf/2410.20672); [TaH](https://arxiv.org/html/2511.08577)

### Inferences
**Cost model.** From the logs, the fixed part is about 436 − 6 × 41.2 ≈ 189 s per 32k examples (question, answer, optimizer). With L_b looped layers run k times, each step costs (12 + (k−1)·L_b)/12 latent-forward units:

T(k, L_b) ≈ 189 + 6 · 41.2 · (12 + (k−1)·L_b)/12 seconds per 32,768 examples

| Arm | Layer-evals per step | s per 32k | h per 385k epoch |
|---|---|---|---|
| ft6 (k=1) | 12 | 436 | 1.4 |
| middle-4, k=2 | 16 | ≈520 | 1.7 |
| middle-4, k=4 | 24 | ≈683 (= ft12) | 2.2 |
| middle-4, k=8 | 40 | ≈1,013 | 3.3 |
| whole-12, k=2 | 24 | ≈683 | 2.2 |
| whole-12, k=4 | 48 | ≈1,178 | 3.9 |

Caveats on these estimates:
- The model assumes latency is linear in layer evaluations, which is plausible because small-batch GPT-2 is launch-bound.
- Random k ∈ {1..4} costs about the mean k.
- CODI's full loss adds a parallel teacher forward/backward over about 100+ CoT tokens. My guess is +30–60%; this is not measured.
- CODI's 36 h / 40 epochs, about 0.9 h per epoch with the teacher at batch 128, suggests **larger batches cut wall-clock substantially**. Prefer batch 128 if memory allows.

**Memory.** Full BPTT through k iterations multiplies activations of the looped layers by k. For GPT-2 at batch 32–128 this is well within an A100 40/80 GB, and probably within an L4's 24 GB.

**L4 vs A100.** Not measured. My rough expectation is 1.5–3× slower on L4 for this launch-bound workload. Measure one 100-update run before budgeting.

**Eval cost.** Inner loops touch only 6 latent positions out of ~30–60 answer-decode forwards per question, so evaluation time grows only modestly. A 500-question eval stays in minutes.

**Matched-compute controls cost the same as the treatment.** ft12 has the same layer-evals as middle-4 k=4 and whole-12 k=2. ft6x2 matches on examples processed.

### Gaps
- The GPU model for the Stage 1 timings is not logged.
- No measured L4 numbers.
- No measured cost for the CODI teacher/distillation pass in this codebase.

---

## Q6. What result would be most convincing, and how could it fail?

### Takeaway
The convincing result is a **matched-compute win of depth over steps**. Looped-CODI at 6 latent steps (k=2–4) should beat **both** of these on GSM8K validation (500) *and* test (1,319), across ≥3 seeds with paired tests:
- ft6, the same fine-tune at k=1;
- ft12, the same layer-evals spent on 6 more appended steps.

It should also replicate on SIM-CoT GPT-2 and on a LLaMA-3.2-1B CODI or SIM-CoT checkpoint. The effect needs to clear about **+3 points (≥15/500)**, because single-seed noise in this project is about ±10/500.

The likeliest outcomes are:
- **null**: the model ignores the loop, as answer-only fine-tuning ignored steps 7–12;
- **small**: +1–2 points, inside the noise;
- **negative**: TaH's AlwaysThink lost about 6 points.

### Cited Findings
- **Noise floor (local).** CODI ft6 scored 209 and SIM-CoT ft6 215, against SIM-CoT ft6x2 206 at 6 steps. The memory note gives single-seed noise of about 10. — `results/latent_gate/stage1/codi/report.txt`; memory note dsa-pivot-latent-eviction
- **Depth headroom in the current model is small (local).** "Correct at any depth of 6, 12 or 24" adds only 9–15 questions over the best single depth (memory note). ft12 at 12 steps (210) ≈ ft6 at 6 (209). — `results/latent_gate/stage1/codi/report.txt`
- **Naive iterate-everything hurts after fine-tuning.** TaH AlwaysThink: 58.2 → 52.6 (0.6B), 80.2 → 74.2 (1.7B). — [TaH](https://arxiv.org/html/2511.08577)
- **Models can learn to ignore recurrence** (Huginn Bad Run 2: same perplexity at 1 and 32 recurrences). — [Huginn](https://arxiv.org/html/2502.05171)
- **Latent tokens are often unneeded and readouts attend to the question.** Dilgren & Wiegreffe: Coconut/CODI rarely need their latents. Cui et al.: answer tokens' top-10 attention targets are question tokens (snippet). **[prior notes]** — [arXiv 2604.04902](https://arxiv.org/pdf/2604.04902); [arXiv 2602.22441](https://arxiv.org/html/2602.22441)
- **Outcome-only supervision gives latents no measurable benefit over direct answers on GSM8K-Aug; process supervision does.** **[prior notes]** — [arXiv 2606.20075](https://arxiv.org/pdf/2606.20075)
- **External bars on GPT-2 / GSM8K:**
  - LOTUS 44.1 (vs its own CODI re-run 42.0) — [LOTUS](https://arxiv.org/html/2606.31779)
  - PCCoT 49.48; Continuous CoT 48.24 — [PCCoT](https://arxiv.org/html/2506.18582)
  - PMPS 44.35 vs CODI 40.86 **[prior notes]** — [arXiv 2609.09928](https://arxiv.org/pdf/2609.09928)

### Inferences
**Most convincing package**, in order of importance:
1. **Primary.** Looped-CODI (4/4/4 block, k ~ U{1..4} in training, evaluated at k=3) at 6 steps vs ft6 and vs ft12. Same examples, same order, same LR, ≥3 seeds. McNemar on val-500 and test-1319, preregistered as in Stage 1.
2. **Dose-response.** Accuracy vs k = 1..8 at 6 steps: rising to k_train and flat after. Also the per-step logit-lens / SIM-CoT-decoder **intermediate-result accuracy rising with k**, the mechanistic evidence that depth is used.
3. **The dilution contrast.** The same extra compute spent as appended steps (ft12) vs overwrite (Recipe 2) vs inner block (Recipe 1). The cache-size story is supported if overwrite and inner loops ≥ appended steps.
4. **Generality.** The SIM-CoT GPT-2 checkpoint, then `zen-E/CODI-llama3.2-1b-Instruct` or `internlm/SIM_COT-LLaMA3-CODI-1B`. The project's attention patch hooks `GPT2Attention._attn` only, so the loop wrapper needs a LLaMA port. That port is simpler than the bias patch, since it only re-calls decoder layers.
5. **Controls a reviewer will ask for:**
   - parameter-matched k=1 with the same added LoRA (rules out "more parameters");
   - TaH-style duo-causal KV (expected to dilute);
   - LOTUS and PCCoT reported numbers as rows;
   - a pause-token / extra-steps arm.

**Main ways it could fail:**
1. **The model ignores the loop** (gate g stays ≈0, LoRA-B ≈0). Answer-only CE gives no reason to use depth. Mitigation: per-step SIM-CoT decoder loss or CODI distillation; monitor ‖g‖.
2. **Overthinking / drift inside the loop** (TaH, Ouro). Mitigation: damping, random k, terminal fixed-point penalty.
3. **The coda and readout see shifted residual norms**, a distribution shift like McLeish's "healing" issue. Mitigation: outer norm, zero-init gate.
4. **Question-KV mismatch.** Deeper iterations query keys computed at "depth 1." Huginn says this is workable; it is unmeasured here.
5. **Compute-matched null.** A gain appears vs ft6 but not vs ft12 or ft6x2. That makes it "more compute," not "depth beats steps."
6. **Budget too small.** Retrofits that gained 6–12 points used about 10³–10⁴× more tokens. A 32k–100k-example fine-tune may only reach a +1–2 point regime, invisible under the ±10/500 noise. Mitigation: use test-1319 plus seeds, or one full epoch (≈2.2 h).
7. **Capacity rather than depth limits GPT-2.** CODI's errors may be arithmetic or knowledge errors that deeper latents cannot fix. The any-depth oracle's +9–15 is an upper-bound hint from depth variation.
8. **Novelty objection.** "LOTUS / TaH applied to CODI." The defence is the matched steps-vs-depth result and the dilution link, not the mechanism.

**My expected gain (speculative).** With a short fine-tune and a per-step loss: 0 to +2 points at 6 steps (≈0–10/500), hard to separate from noise. With one or more full epochs: maybe +2–4. Reaching PCCoT's 49.5 looks unlikely without much more training. The most likely *publishable* result is the controlled comparison (3), whatever its sign.

### Gaps
- No prior work reports a loop retrofit on a CODI or SIM-CoT checkpoint, so no direct effect-size prior exists. The estimates above are extrapolations from TaH, ETD, McLeish and LOTUS at other scales and with other recipes.
- I do not know whether the SIM-CoT GPT-2 auxiliary-decoder weights are usable for a step loss after looping. The memory note says the checkpoint contains them; I did not inspect them.
