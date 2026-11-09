# Literature brief: latent CoT (CODI / SIM-CoT) with a Mixture-of-Experts student

Compiled 2026-11-08 for the MoE idea round (literature, then design, then review). Scope: GPT-2-small CODI and SIM-CoT, with an MoE student meant to cut active parameters at equal GSM8K accuracy.

**Tags.** **[verified]**: I opened the arXiv abstract, HTML or PDF (or the official page) this session, and the stated numbers come from it. **[unverified]**: from memory, a search-engine snippet or a secondary site. Treat these as leads, not citations. **[Proj]**: this project's own code or measurements. **[Inf]**: my inference, not a paper's claim.

**Search coverage.** Ten or more arXiv full-text and abstract queries (sorted newest first, through Nov 2026): "latent reasoning mixture-of-experts", "latent tokens reasoning MoE", "implicit reasoning mixture of experts", "continuous chain-of-thought experts", "latent reasoning MoE", "implicit chain-of-thought mixture-of-experts", "latent chain-of-thought distillation student compression", "soft thinking Qwen3-30B-A3B", and "Coconut MoE latent". Plus about 25 web searches pairing Coconut, CODI, SIM-CoT, Soft Thinking, OLMoE, DeepSeek-V2-Lite, Qwen3-30B-A3B, gpt-oss and GSM8K-Aug with MoE, upcycling, routing and distillation terms.

---

## Bottom line

- **The exact combination looks new.** I found no paper that trains a Coconut/CODI/SIM-CoT-style continuous-CoT model (latents fed back as input embeddings) on an MoE backbone. I also found none that distills latent or implicit (ICoT-style) reasoning into an MoE student. Every trained latent-CoT method I checked uses a dense backbone: GPT-2, LLaMA-3.2-1B/3B/8B or Qwen2.5. The only latent-thought runs on an MoE backbone are **training-free**: ThinkRouter (2602.11683) ran Soft Thinking on gpt-oss-20b as a baseline, with no MoE analysis.
- **Main threat: "latent reasoning + MoE", framed as depth recurrence.** Dreamer (2601.21582; ~2B, 200B tokens, from scratch) combines depth-recurrent latent reasoning with sparse expert attention and depth-aware routing, and reports GSM8K. It uses no continuous thought tokens and no distillation, but a reviewer will cite it for "MoE helps latent reasoning".
- **Step-specific routing in a recurrence is already published as a mechanism.** LOOM (2610.01153) names "expert selection collapse" across loops and fixes it with per-loop routers. MoEUT, SUT, SMELT and LatentLoop study MoE in looped transformers. Think-at-Hard (2511.08577) uses iteration-depth-specific LoRA in latent iterations. Time-step-specific experts are standard in diffusion (eDiff-I, Denoising Task Routing). So step- or phase-indexed experts for CODI latents count as an *application* of a known mechanism, not a new one. Caveat: CODI's recurrence runs along the sequence (each latent is a new position through all 12 layers), not through depth. The looped-MoE results transfer only by analogy [Inf].
- **"Convert a dense GPT-2 into an MoE with fewer active params and distill" is not new by itself.** MoEfication (2110.01786), MoEBERT (2204.07675: BERT to 4 experts, top-1, 66M effective params, layer-wise hidden-state distillation, beats DistilBERT/TinyBERT), D2DMoE (2310.04361, done on GPT-2) and G-MoEfication (EMNLP 2024, GELU models) all do it. What is new is applying it to *latent* reasoning, and asking whether the 6 continuous thoughts survive sparsification or route differently from tokens.
- **Reasoning distillation into MoE students exists, but only in explicit form.** Examples: PADD (2606.10369, ICML 2026: dense 7B math teachers to Qwen3-30B-A3B / DeepSeek-V2-Lite students, with teacher-neuron clustering to initialise the experts), Qwen3-30B-A3B's strong-to-weak distillation, and MoDE-CoTD (LoRA experts). None of them has a latent student.
- **The literature predicts no accuracy gain and a real risk of loss.** Mixture of Parrots (2410.19034) and Optimal Sparsity (2508.18672, ICLR 2026 oral) find that math reasoning (GSM8K) tracks active parameters and width, not total parameters. MoEfication needs more active neurons for harder tasks and for smaller models: T5-Small needs 30–40% of FFN neurons for ~98%. ICoT finds that implicit reasoning depends on depth. The defensible claim is "equal accuracy at fewer active params", or a clear negative.
- **Efficiency claims at 124M must be about active params and FLOPs, not wall-clock.** MoEUT at 244M trains 1.7x *slower* than dense (772 vs 443 ms/batch). MoEfication with 25% of FFN gives only 1.2x on GPU (2x on CPU). MoEBERT runs slower than DistilBERT at batch 1. A 25%-FFN GPT-2 MoE has **~42.6M active non-embedding params, the same as DistilGPT2 (42.5M)**, so a reviewer will demand DistilGPT2 and a width-pruned dense baseline.

---

## 1. Direct prior work: latent CoT on MoE, and latent or implicit reasoning distilled into MoE

### 1a. Latent-CoT methods: all dense backbones

| Method | ID | Backbone(s) | Relevance | Tag |
|---|---|---|---|---|
| Coconut | 2412.06769 | GPT-2 (later revisions add LLaMA) | Last hidden state fed back raw. No MoE | [verified] abstract; backbone [Proj] |
| CODI | 2502.21074 (EMNLP 2025) | GPT-2, LLaMA-3.2-1B | Shared-weight self-distillation, L1 on hidden states at the token before the answer; GPT-2 43.7%, "3.1x compression". No MoE | [verified] abstract; [Proj] for the recipe (LoRA r=128, α=32 on c_attn/c_proj/c_fc, merged in `dsa/bench/models/codi.py`) |
| SIM-CoT | 2509.20317 | GPT-2, LLaMA 1B/3B/8B | Step decoder supervises latent k as step k. No MoE | [Proj] |
| CoLaR, PCCoT, LightThinker, CCoT, HRPO, Token Assorted | 2505.16552, 2506.18582, 2502.15589, 2412.13171, 2505.18454, 2502.03275 | LLaMA-3.2-1B, GPT-2, Qwen2.5-7B / Llama-3.1-8B, Llama-2-7B, Qwen2.5 1.5B/3B, Llama | All dense as far as I know | [unverified] (memory and project notes) |
| Soft Thinking | 2505.15778 (NeurIPS 2025) | QwQ-32B, R1-distill 32B/70B (dense) | Training-free; up to +2.48 pass@1, −22.4% tokens | [verified] abstract numbers; backbones [unverified] |
| **ThinkRouter** | 2602.11683 | Qwen3-1.7B/8B/32B and **gpt-oss-20b (MoE)** | Training-free switching between soft and discrete thinking. On gpt-oss-20b STEM: CoT 71.95, Soft Thinking 70.08, ThinkRouter 83.88 avg Pass@1. **Says nothing about MoE routing of soft inputs** | [verified] HTML |
| System-1.5 Reasoning | 2505.18962 | not named in abstract | Latent CoT distilled from language CoT, plus *depth shortcuts* (non-critical tokens exit early via adapter branches) and step shortcuts. GSM8K: comparable to CoT fine-tuning, >20x faster inference, −92.31% tokens. The closest "fewer active compute in latent reasoning" precedent, but it uses early exit, not MoE | [verified] abstract |
| Think-at-Hard (TaH) | 2511.08577 (ICLR 2026) | Qwen3 | Extra latent iterations only on hard tokens (94% skip the second pass); **depth-aware LoRA used only on latent iterations**; +8.1–11.3 over always-iterate-twice | [unverified] (arXiv listing via search snippet) |
| LOTUS | 2606.31779 | up to 3B | Looped padded transformer, parallel latent blocks. No MoE | [verified] abstract |

### 1b. Papers that pair "latent" with "experts" (near misses)

- **Dreamer: Depth-Recurrent Attention Mixture**, 2601.21582 (Jan 2026, rev. Nov 2026, NeurIPS 2026 submission) [verified abstract + HTML]
  - One recurring layer applied up to depth 32, with depth attention and *expert attention*: attention over a pool of MLP experts, top-8 per token per depth. Experts are shared across depths, and routing is made depth-aware by depth-RoPE on the queries.
  - Sizes 0.45–1.8B non-vocab, ~100B reasoning tokens plus ~100B general, ~34.5k H100-hours. Benchmarks include GSM8K and MATH.
  - Results: 2–8x data efficiency, reasoning average 51.4 vs 39.5 for a matched Transformer at depth 32, 2–11x more expert-selection diversity than standard MoEs, ~13% inference slowdown unoptimised.
  - **No continuous thought tokens, no distillation, from scratch.** It threatens the general "MoE + latent reasoning" framing only.
- **MoLE: Mixture of Latent Experts for Complementary Visual Reasoning**, 2610.01917 (Oct 2026, cs.CV) [verified abstract]
  - VLM latent visual reasoning in which latent tokens act as "latent visual experts" plus "latent summary experts". The abstract describes no standard routed MoE.
  - 78.6 average on 5 visual benchmarks, +3.6 over the best latent baseline at equal latent budget. Shares the name, not the mechanism.
- **MoLEM: Dynamic Mixture of Latent Memories**, 2605.21951 [verified abstract]
  - A dynamic MoE of adapters generates latent memory that is injected into a *frozen* reasoner; aimed at continual learning, +10.40% average. Not latent CoT and not an MoE student.
- **Mixture of Thoughts**, 2509.21164 [unverified, secondary]
  - Latent-level collaboration among separate expert LLMs under a router, through cross-attention in a shared latent space. A multi-model ensemble, not an MoE student.
- **CoWorld-VLA**, 2605.10426; **Beyond the Trace**, 2608.17638 (reads reasoning state from native MoE routing); **MORES**, 2607.08116 (the MoE is a scheduler) [verified arXiv listing abstracts]. Peripheral.

### 1c. Implicit-CoT distillation (all dense)

- **ICoT-KD**, Deng et al., 2311.01460 [verified PDF]
  - Teacher, emulator and student "vertical" reasoning on GPT-2. GSM8K-Aug 22% with no CoT, against 17% for the best no-CoT GPT-2.
  - GPT-2 Small gets 97% on 4x4 multiplication but **10% on 5x5, while GPT-2 Medium gets 96%**; the authors read this as implicit CoT "might depend on having a sufficient number of layers." Relevant to depth-reduced baselines (DistilGPT2).
- **Stepwise Internalization (ICoT-SI)**, 2405.14838 [verified abstract]
  - GPT-2 Small 99% on 9x9 multiplication; Mistral-7B over 50% on GSM8K with no steps. Dense.
- I found no ICoT, ICoT-SI or CODI distillation into an MoE student.

---

## 2. CoT / reasoning distillation into MoE students; dense-to-MoE and MoE-to-dense KD

### Dense teacher to MoE student

- **PADD: Path-Aligned Decompression Distillation**, 2606.10369 (ICML 2026) [verified HTML]
  - Dense 7B math teachers (Qwen2.5-Math-7B, DeepSeek-Math-7B) to *pretrained* MoE students: Qwen3-30B-A3B (128 experts, top-8, 3.3B active) and DeepSeek-V2-Lite (2.4B active).
  - Stage I clusters teacher FFN neurons (rows of W1) with cardinality-constrained K-means, one cluster per student expert, and initialises the router from the centroids. Then logit KL plus an expert-activation alignment loss, advantage-scaled online KD, a routing-aware GRPO variant and a reward-augmented load-balancing term.
  - Average Pass@1 on 5 math benchmarks: Qwen family 80.2 vs teacher 77.7 vs online KD 73.6; dense-at-active-size GRPO 53.5. No GSM8K, explicit CoT only, logit-level (not hidden-state) KD.
  - **Closest "reasoning distillation into MoE" paper.** Its neuron-clustering-to-experts step is the same idea as MoEfication.
- **MoDE-CoTD**, LREC-COLING 2024, pp. 11475–11485 [verified ACL page]
  - Explicit CoT distillation into a *frozen* student plus a mixture of decoupled LoRA experts. +6.3% on seen and +7.8% on unseen datasets (14 datasets). Adds parameters rather than cutting active ones.
- **Qwen3 Technical Report**, 2505.09388 [unverified detail]
  - Strong-to-weak distillation (off-policy, then on-policy logit KD) trains the 0.6B–14B dense models and the **Qwen3-30B-A3B MoE** from larger teachers. Explicit-CoT distillation into an MoE student is routine at scale.
- **Instella-MoE**, 2609.00791; **Motif 3**, 2608.09119 [verified arXiv-listing abstracts]: MoE models (16B, 314B) post-trained with multi-teacher on-policy distillation. Explicit.
- **MoEBERT** (Section 3): a dense BERT teacher with layer-wise hidden-state distillation into an MoE student built from the same weights. The closest *structural* match to "CODI teacher hidden states into an MoE student".

### MoE to dense, and MoE to smaller MoE (for contrast)

- **Switch Transformer**, 2101.03961 [verified abstract; KD numbers unverified, snippet]: distilling sparse into dense keeps ~30% of the quality gain at ~1/20 of the params, and 28% at 99% compression.
- **Every Expert Matters**, 2502.12947 [verified HTML]
  - Llama-MoE-3.5B/3.0B teachers vs a dense Sheared-Llama-2.7B teacher, both into a dense Sheared-Llama-1.3B student.
  - With standard KD, **the dense teacher beat every MoE teacher** (ROUGE-L 21.95 vs 20.50–20.92). Their Knowledge Augmentation and Student-Aware Router recover 25.3–26.2.
- **DeepSpeed-MoE, Mixture-of-Students**, 2201.05596 [unverified, snippet]: staged KD of PR-MoE into a shallower MoE student keeps 99.5% / 99.1% of the 350M / 1.3B teachers with 12.5% fewer layers. **Distilling for the whole run hurt; KD had to stop early (~400K steps).**
- **Efficient Large-Scale LM with MoEs**, 2112.10684; **Pruning and Distilling MoE into Dense**, 2605.28207 [unverified, snippets]: both distill MoE into dense.

---

## 3. Sparse upcycling and dense-to-MoE conversion (fewer active params)

### Conversions that keep total params and cut active ones

- **MoEfication**, Zhang et al., 2110.01786 (Findings ACL 2022) [verified PDF]
  - Split each FFN into experts of 32 neurons each, giving 64–512 experts depending on T5 size. Best construction: co-activation graph partitioning. Best selection: a small 2-layer tanh MLP router trained with CE to predict which experts hold the activated neurons (Adam lr 1e-2, batch 512, 10 epochs, ~500k sampled inputs, "several minutes per FFN on one GPU"). No base-model gradient is needed.
  - 10–30% of FFN params keep >95% of performance.
  - **Smaller models need more:** T5-XLarge keeps ~98% with 10% of neurons, T5-Small needs 30–40%. Harder tasks need more: RACE drops most.
  - Speedup at 50% / 25% / 12.5% of FFN: FLOPs 1.5x / 2.0x / 2.4x, CPU 1.43x / 1.98x / 2.28x, **GPU 1.15x / 1.20x / 1.47x**.
  - **GELU models (BERT, GPT):** GELU is not sparse, so they first swapped BERT-Large to ReLU and continued pretraining for 400 steps; then 30–40% of neurons recover 97%. Random split plus random selection is far worse (SST-2 65.9 vs 96.2).
- **MoEBERT**, Zuo et al., 2204.07675 (NAACL 2022) [verified PDF]
  - BERT-base FFN (3072) into **4 experts of width 768, top-1**. The 512 most important neurons (first-order Taylor importance) are *shared* by all experts; the rest are split.
  - Effective params 66M vs 110M, matching DistilBERT. **Random-hash routing** (token id to expert, fixed) was the default and performed about as well as a learned sentence gate or balanced hash.
  - Layer-wise distillation: MSE on every layer's output plus prediction KL. "Last-layer-only" distillation was unsatisfactory; "every other layer" matched "all". Removing distillation costs 0.7 on RTE and 1.1 EM on SQuAD 2.0.
  - GLUE: beats DistilBERT, TinyBERT, MiniLM, PKD and Theseus on 6/7 tasks and matches BERT-base. SQuAD 2.0 F1 76.8 vs MiniLMv2 76.3 vs DistilBERT 69.5.
  - **The "1 expert" ablation (= a width-pruned dense model at the same active size) is clearly worse**, and random splitting is worse than importance splitting.
  - Inference on CPU at batch 1: **slower than DistilBERT** (6 layers vs 12), faster than BERT.
- **D2DMoE: Dense to Dynamic-k MoE**, Szatkowski et al., 2310.04361 [verified HTML; venue unverified, I believe NeurIPS 2024]
  - **Run on GPT-2 base (OpenWebText).** A sparsification phase (square-Hoyer activation penalty, 1000 fine-tune steps, ~1B tokens, under 1% of pretraining) is followed by expert-contribution routers (2-layer MLP, hidden 128, 2000 steps on 8–16M tokens) and dynamic-k selection (an expert runs if its predicted norm ≥ τ·max).
  - Beats MoEfication at every FLOP budget on GPT-2. **A non-sparsified GELU model collapses under conversion**; a sparsified GELU model behaves like ReLU. Up to 60% cost reduction (abstract); smaller experts are better.
  - No GPT-2 wall-clock numbers; a single-layer A100 Triton kernel is "almost 3x" faster.
- **G-MoEfication ("Breaking ReLU Barrier")**, Lee et al., EMNLP 2024 main [verified PDF]
  - For GELU / non-ReLU models: a deactivated expert's output is replaced by a static "representative value" rather than zero.
  - mBERT at 35% of FFN keeps 96% (NER) and 97.7% (POS) at 58% FLOPs. At 75% of FFN: G-MoEfication 86.46 vs MoEfication 82.79 (oracle 87.32).
  - Also run on SantaCoder-1.1B, Phi-2 and Falcon-7B.
- **LTE (Learn-To-be-Efficient)**, 2402.06126 (NeurIPS 2024) [verified abstract]: trains structured sparsity into non-ReLU LLMs; 25% lower LLaMA2-7B latency at 50% sparsity with a custom kernel.
- **CMoE**, 2502.04416 [unverified, snippet]: training-free carving into shared plus routed experts via balanced assignment; 25% active gives 1.5x end-to-end latency; optional LoRA recovery.
- **DS-MoE: Dense Training, Sparse Inference**, 2404.05567 [verified abstract]: dense training through all experts, sparse inference; 30–40% of params active; DS-MoE-6B up to 1.86x faster than Mistral-7B in vLLM.
- **LLaMA-MoE**, 2406.16554 [verified abstract]: split LLaMA-2-7B FFNs into experts, continual pretraining on 200B tokens; LLaMA-MoE-3.5B beats dense models with similar active params. **LLaMA-MoE v2**, 2411.15708 [verified abstract]: experts in attention and MLP, two-stage post-training to recover math, code and chat lost to sparsity.

### Upcycling (more total params, same or more active)

- **Sparse Upcycling**, Komatsuzaki et al., 2212.05055 [verified abstract]: copy the dense FFN into experts. On T5 Base/Large/XL and ViT B/L, about 50% of the original pretraining cost beats continued dense training and beats MoE from scratch. **It does not reduce active params.**
- **Drop-Upcycling**, 2502.19261 (ICLR 2025) [verified abstract]: partially re-initialises expert weights to break copy symmetry and promote specialisation. Naive upcycling "is slower than from scratch and suboptimal in the long run". 5.9B-active MoE matches a 13B dense at ~1/4 the FLOPs.
- **Upcycling LLMs into MoE** (NVIDIA), 2410.07524 [verified abstract]: "virtual group" initialisation plus weight scaling for *fine-grained* upcycling; softmax-then-top-K beats top-K-then-softmax; higher granularity helps; Nemotron-4 15B, 1T tokens: MMLU 67.6 vs 65.3 for continued dense.
- **Scaling Laws for Upcycling MoE**, 2502.03009 (ICML 2025) [verified abstract]: an interaction term between the dense and upcycling datasets limits upcycling efficiency at large budgets.
- **DeepSeekMoE**, 2401.06066 [verified abstract]: fine-grained segmentation (mN experts, mK active) plus K_s shared experts. DeepSeekMoE 2B ≈ GShard 2.9B (1.5x the expert compute) and nearly matches a dense model of equal *total* params.
- **Small-scale GPT-2 upcycling** (HF repo `sumitdotml/moe-emergence`, not peer reviewed) [verified README]
  - GPT-2 small, layers 8–11 turned into 8 experts (noisy MLP copies), top-1, 257M total, 6.6M tokens of code, math and prose, batch 8.
  - **Without load balancing, one expert took 73.6% of tokens by step 500**; z-loss alone did not prevent it. Top-2 vs top-1 gave 0.14% eval loss for 2x expert compute.
  - Caveat: the MoE trained 10k steps against 5k for dense, an unmatched comparison and a pitfall to avoid.

---

## 4. MoE in looped, recurrent and universal transformers; step-, phase- and token-type routing

### Looped and universal MoE

- **MoEUT**, Csordás, Irie, Schmidhuber, Potts, Manning, 2405.16039 (NeurIPS 2024) [verified HTML]
  - σ-MoE FFN plus SwitchHead attention experts, layer grouping G=2 (<300M), peri-layernorm. At **126M: 254 experts of size 128, K=12, 10 attention experts. C4 perplexity 14.76 vs 14.97** for a parameter-matched dense model; 244M: 13.24 vs 13.40.
  - "Halves the MAC requirements." **Wall-clock 772 vs 443 ms/batch (~1.7x slower)** at 244M on 8 V100s, blamed on kernels; memory 9.0 vs 9.2 GB.
  - Routing: first-layer selection is mostly driven by token identity, deeper layers by context; frequent tokens reuse experts across layers, rare tokens use layer-specific ones.
- **Sparse Universal Transformer**, Tan et al., 2310.07096 [verified abstract]: SMoE plus stick-breaking halting; matches WMT'14 baselines at ~half the compute and params; halting cuts inference by ~50% on formal tasks (logical inference, CFQ).
- **Mixture-of-Recursions**, Bae et al., 2507.10524 [verified abstract]: routers assign per-token recursion depth; attention and KV only over active tokens; at 135M–1.7B, a better Pareto frontier at equal training FLOPs (no numbers in the abstract).
- **Relaxed Recursive Transformers**, Bae et al., 2410.20672 (ICLR 2025) [verified abstract]: convert pretrained LLMs into looped models, initialised from the pretrained weights; **depth-wise (per-loop) LoRA** relaxes the tying; recursive Gemma 1B beats TinyLlama 1.1B and Pythia 1B.
- **LOOM: Looping Beyond Twice**, 2610.01153 (Oct 2026) [verified abstract]
  - Names **"expert selection collapse"**: routers pick the same experts on every loop, so extra loops add no diversity. Fix: per-loop routers, plus residual scaling, input re-injection and a looping residual.
  - 100M–1.7B. At 700M near-iso-FLOP, 5 loops: perplexity 18.36 to 16.54.
- **SMELT**, 2609.01343 [verified abstract]: compute-, param- and KV-matched looped MoE (middle half looped twice) saves 6.8–18.0% of training FLOPs, up to 54B.
- **LatentLoop**, 2610.19975 [verified abstract]: looped MoE in which MoE weights are shared across visits and attention is separate; the first visit tolerates a smaller residual scale.
- **Tying the Loop**, 2606.16825 [unverified, snippet]: experts tied across consecutive layers, routers kept per layer.
- **Chain-of-Experts**, 2506.18945 [verified abstract]: experts applied iteratively inside a layer with a *dedicated router per iteration*; on math, validation loss 1.20 to 1.12 at fixed compute; 2 iterations match 3x expert width with 17.6–42% less memory.
- **Mixture-of-Depths**, 2404.02258 [verified abstract]: top-k token routing per layer, with fixed capacity; matches the baseline at equal FLOPs; sampling up to 50% faster per step.
- **Think-at-Hard** (Section 1a): iteration-specific LoRA for latent passes [unverified].

### Step- or phase-indexed parameters and routing

- **Mixture-of-Transformers**, 2411.04996 (TMLR 2025) [verified abstract]
  - Decouples *all* non-embedding params (FFN, attention projections, norms) by modality, with global attention. Chameleon-7B quality at 55.8% of FLOPs; 760M MoT beats 1.4B dense on image metrics.
  - Note: per-token active params stay at the full stack size; the savings are "FLOPs to reach equal quality", not fewer active params.
- **Hash Layers**, Roller et al., 2106.04426 [verified abstract]: fixed hash routing by token, with no router or balancing loss, is competitive with Switch and BASE. MoEBERT's hash routing is the same idea. Routing CODI latents by *step index* is a deterministic hash of this kind [Inf].
- **eDiff-I**, 2211.01324 [verified abstract]: one model is trained, then split into denoisers specialised for different stages of the iterative process, at the same inference cost as one model.
- **Denoising Task Routing**, 2310.07138 (ICLR 2024) [verified abstract]: a fixed channel subset per timestep, with sliding-window overlap between adjacent steps and more task-specific channels at early steps. No extra params; with MTL methods, DiT-L matches DiT-XL in 2M vs 7M iterations. **The closest precedent for "step-indexed FFN neuron subsets in an iterative process."**
- **RICE: Two Experts Are All You Need for Steering Thinking**, 2505.14681 [verified abstract]: "cognitive experts" found by nPMI with `<think>` tokens in DeepSeek-R1 and Qwen3-235B. Reinforcing them at inference improves reasoning accuracy and efficiency. Phase-specific experts emerge in large MoEs.
- **SAEM: Stage-Aware Expert Management**, 2608.21614 [verified abstract]: consecutive CoT stages show "coherent and predictable expert activation patterns"; 1.33–1.54x throughput from stage-aware caching.
- **Phase-Aware MoE for agentic RL**, 2602.17038; **Does the Same Token Mean the Same State?**, 2606.22798 (routing at the answer boundary separates reasoning budgets) [unverified, snippets].
- **Depth-Aware Sensitivity Analysis of MoE**, 2608.13565 [abstract verified; phase claim unverified]. The abstract covers only layer-wise expert masking in Qwen3.6-35B-A3B. A search snippet says a probe in the body found reasoning-phase routing broader than answer-phase routing, with think-heavy and answer-heavy experts (Qwen3.5-35B-A3B). I did not see this in the abstract.
- **ST-MoE**, 2202.08906 [verified PDF]: "at each layer, at least one expert specializes in sentinel tokens". Special non-text tokens get their own expert. By analogy, `<bot>`/`<eot>` and off-manifold latent embeddings may attract dedicated experts [Inf].

---

## 5. Expert specialisation for arithmetic and math; small-scale MoE on GSM8K

- **ST-MoE**, 2202.08906 [verified PDF]: encoder experts specialise in punctuation, verbs, proper names, "counting, etc."; **decoder experts lack meaningful specialisation**, which the authors attribute to fewer routed tokens per group and the objective. GPT-2 is decoder-only.
- **Mixtral**, 2401.04088 [verified HTML]: "no obvious patterns" of experts by topic; **only DM Mathematics has a "marginally different distribution"** (synthetic data). Routing is syntactic (Python `self`, indentation). Consecutive tokens repeat the first-choice expert 23.6–28.4% at layer 15 vs a 12.5% chance level.
- **The Myth of Expert Specialization in MoEs**, 2604.09780 [verified HTML]: on 30 HMMT questions, two models' solutions to the *same* question share ~60% of top experts, about the same as one model on *different* questions. Routing follows hidden-state geometry, not domain. Deep layers converge to near-identical experts during prefill in reasoning models and diverge once generation starts. No digit- or operation-level analysis.
- **The Expert Strikes Back**, 2604.02178 [unverified, snippet]: experts are neither domain specialists nor simple token processors.
- **OLMoE**, 2409.02060 [verified abstract; details unverified]: 1B active / 7B total, 5T tokens; "high specialization" (domain and vocabulary specialisation, early router saturation, from memory).
- **Mixture of Parrots**, 2410.19034 [verified HTML]
  - 20-layer dense models (width 256–4096) vs MoEs (width 256–1024, 8–64 experts, top-2) on ~65B math tokens; SVAMP, GSM8K, GSM-Hard, MATH.
  - **On math, MoEs do not reach dense performance at equal total params**, and "increasing the dense model width is more effective than adding experts". The MoE train-test gap is larger above 159M, which the authors read as memorisation.
  - Theory: some graph problems are unsolvable by any number of experts of a given width but solvable by a slightly wider dense model.
- **Optimal Sparsity of MoE LMs for Reasoning**, 2508.18672 (ICLR 2026 oral) [verified HTML]
  - 16 layers, d ∈ {512, 1024, 2048}, E from 8 to 256, k from 2 to 16, 125B tokens.
  - GSM8K and GSM-Plus task loss is **U-shaped in total params** and can worsen while pretraining loss improves; the optimum moves with active params.
  - "At identical training loss, larger top-k consistently outperforms smaller top-k on reasoning." GRPO and self-consistency (128 samples) do not remove the effect. d=2048, E=16, k=2: GSM8K 31.8% at 125B tokens, 36.3% at 1T.
- **Chain-of-Experts** (Section 4): the math-loss gains come from iteration-specific routing.
- **Small-scale GSM8K with MoE:** I found **no GPT-2-scale (≤150M) MoE result on GSM8K or GSM8K-Aug**, latent or explicit. The smallest are Mixture of Parrots (18M–200M-active MoEs pretrained on math; numbers only in figures) and the HF GPT-2 upcycling repo (math loss on MathQA, not accuracy).
- **No paper analyses expert routing per digit, operation or reasoning step in an arithmetic setting** that I could find.

---

## 6. Efficiency accounting and known pitfalls

### How small-MoE papers report cost

- **Active parameters:**
  - MoEBERT counts "effective params" including embeddings (66M vs 110M).
  - MoEfication reports % of FFN params.
  - MoEUT is parameter-matched on total and reports MACs and memory.
  - σ-MoE (2310.10837, Findings EMNLP 2023) [verified abstract] argues for **parameter-equal** comparisons (not compute-equal) at small scale. Its MoEs are competitive with dense Transformer-XL on WikiText-103 and enwik8 at two scales.
- **FLOPs and wall-clock:**
  - MoEfication gives FLOPs, CPU and GPU speedups separately (GPU only 1.2x at 25%).
  - MoEUT trains ~1.7x slower in wall-clock.
  - MoEBERT measures CPU latency at batch 1.
  - D2DMoE measures single-layer A100 Triton kernels only.
  - TEAL (2408.14690, ICLR 2025) [verified abstract]: 40–50% training-free activation sparsity, 1.53x / 1.8x decode speedup *with custom kernels*.
  - Deja Vu (2310.17157, ICML 2023) [verified abstract]: contextual sparsity of heads and neurons via predictors, >2x latency on OPT-175B.
- **Deployment overheads:** Towards MoE Deployment (2303.06182) [verified abstract] reports throughput and memory inefficiencies; dynamic gating gives 6.21–11.23x throughput for LM.

### Pitfalls

1. **Load imbalance and router collapse.**
   - Within 500 steps without a balancing loss (HF GPT-2 repo).
   - Remedies: Switch auxiliary loss; ST-MoE router z-loss, which "stabilizes models without quality degradation" [verified PDF]; Aux-loss-free bias balancing, 2408.15664 [verified abstract], avoids interference gradients.
2. **Sparse models overfit in fine-tuning and want a different protocol** (ST-MoE [verified PDF]).
   - "Sparse models benefit from smaller batch sizes and a higher learning rate", unlike dense models. "Simply using the same fine-tuning hyperparameters that worked well for the dense model can mask any pre-training improvements."
   - Updating only the MoE params "drastically" hurts; updating only the non-MoE params works about as well as updating everything. Expert dropout did not help.
   - **This conflicts with the project's large-batch default (batch 128, sqrt LR) and should be flagged** [Inf].
3. **GELU is not sparse.** Naive MoEfication of GPT-2 loses much more than for ReLU models (MoEfication §4.3; D2DMoE: a non-sparsified GELU model collapses). Fixes: a sparsity fine-tune (Hoyer), a ReLU swap with continued training, or G-MoEfication's representative values.
4. **Copy symmetry in upcycling.** Identical experts get identical gradients unless noise, partial re-initialisation (Drop-Upcycling) or splitting (MoEfication) breaks the tie.
5. **Expert-choice routing is non-causal** (experts pick tokens across the batch), so it does not suit autoregressive latent or answer decoding without fixes [unverified; known property of 2202.09368].
6. **Expert selection collapse across recurrent steps** (LOOM). CODI latents in this project are known to drift and partly duplicate past step 6 [Proj], so latent steps may collapse onto one expert set [Inf].
7. **Small-batch and latent-step inefficiency** [Inf]. Each latent step processes one position per sequence: at batch 128 that is 128 tokens per layer, split over E experts, so per-expert matmuls are tiny and a Python loop over experts is overhead-bound. At batch-1 decoding, GPT-2 latency is dominated by kernel launches, not FLOPs, so top-k savings will barely show. The answer-token LM head (38.6M params, 77M FLOPs per token) is untouched by FFN-MoE.
8. **Distillation schedule.** DeepSpeed-MoS found KD for the whole run hurts [unverified]. MoEBERT found last-layer-only hidden-state KD insufficient [verified]. CODI distils only at one token position [Proj]; whether that is enough for an MoE student is untested [Inf].
9. **Mixed comparisons.** Unequal steps (HF repo), unmatched tokens, or different hyperparameter tuning effort between dense and MoE arms.

---

## Facts the designer needs

### GPT-2 small parameter breakdown (computed exactly; vocab 50,257, n_ctx 1024, d 768, 12 layers, d_ff 3072)

| Component | Params |
|---|---|
| Token embedding wte (tied LM head) | 38,597,376 |
| Position embedding wpe | 786,432 |
| Attention, per layer (c_attn 768×2304 + c_proj 768×768, with biases) | 2,362,368; **×12 = 28,348,416** |
| FFN, per layer (c_fc 768×3072 + c_proj 3072×768, with biases) | 4,722,432; **×12 = 56,669,184** |
| LayerNorms (2 per layer + ln_f) | 38,400 |
| **Non-embedding total** | **85,056,000** (FFN = 66.6%, attention = 33.3%) |
| **Total** | **124,439,808** |
| CODI additions | +3 token rows (2,304); projection MLP Linear-GELU-Linear + LN = 1,182,720, run once per latent step [Proj] |

- **Forward FLOPs per token** ≈ 2 × 85.06M = 170M non-embedding, plus 77.2M for the LM head on answer tokens, plus ~7.4M attention-score FLOPs at a context of 200 [Inf, standard 2N rule].
- **Ceiling on savings from FFN-only MoE:** the FFN is 2/3 of non-embedding compute, so FFN-only MoE can never cut more than ~67% of non-embedding active params. Attention (28.3M) stays dense unless SwitchHead- or MoA-style attention experts are added (MoEUT).

### What MoEfication-style splitting gives (no new params; total stays 124.4M; router = 768×E per layer)

| Experts E × width | top-k | FFN active | Active non-emb params | % of dense non-emb |
|---|---|---|---|---|
| 8 × 384 | 1 | 12.5% | 35.55M | 42% |
| 4 × 768 (MoEBERT-style) | 1 | 25% | 42.60M | 50% |
| 16 × 192 | 4 | 25% | 42.71M | 50% |
| 32 × 96 | 8 | 25% | 42.86M | 50% |
| 96 × 32 (MoEfication expert size) | 24 | 25% | 43.45M | 51% |
| 4 × 768 or 16 × 192 | 2 / 8 | 50% | 56.8M | 67% |

- **DistilGPT2:** 6 layers, 81.9M total, **42.53M non-embedding**, the same active size as the 25%-FFN MoE. Distilled from GPT-2 124M on OpenWebText; WikiText-103 perplexity 21.1 vs 16.3 for GPT-2 [verified HF card; layer count from config, unverified here]. It would need the full CODI or SIM-CoT recipe from scratch, which is likely far beyond 30 G4-minutes [Inf].
- **Literature priors for the active fraction:**
  - MoEfication: small models need 30–40% of FFN neurons for ~98% on easy tasks, more on hard ones.
  - G-MoEfication on GELU mBERT: 35% of FFN keeps 96–98%.
  - MoEBERT: 25% of FFN with distillation matches BERT-base on GLUE.
  - GSM8K latent reasoning is a "hard task", so expect 25% to be aggressive and 50% to be safer [Inf].
- **Conversion costs reported:** MoEfication routers take minutes per layer and use only the dense model's activations as labels. The MoEfication ReLU swap took 400 pretraining steps. D2DMoE's GPT-2 sparsification took 1000 steps (~1B tokens), about 60x this project's 98,304-record fine-tune (~15M tokens; [Proj] `scripts/inner_loop/train.py`, 768 updates at batch 128).

### Project facts that constrain the design [Proj, from MEMORY.md and code]

- SIM-CoT GPT-2: 519/1319 test, 204/500 validation; CODI GPT-2 published at 43.7%.
- Seed noise ≈ 9/500 on validation, so differences below ~2 points are not resolvable with 1–2 seeds.
- The fine-tune loop is answer CE plus SIM-CoT step loss on 6 latents with a frozen step decoder, batch 128, lr 2e-5·sqrt(batch/32).
- CODI's teacher and student share weights. An MoE student therefore either becomes its own teacher, or needs the dense SIM-CoT/CODI model as a separate frozen teacher: a cross-architecture CODI with no precedent found, and roughly double the memory [Inf].
- Latents drift and partly duplicate past 6 steps; training-free dedup holds the 6-step score.

### Baselines a reviewer would demand

1. The **dense parent** (SIM-CoT or CODI GPT-2) fine-tuned with the *same* extra updates and recipe (the project's `ft6` control), so MoE is separated from extra training.
2. A **dense model at matched active size by width**: FFN pruned to d_ff = 768 (importance-ranked neurons, as in MoEBERT's "1 expert" ablation, Minitron 2407.14679 [verified abstract]) and fine-tuned identically. This is the key "is routing worth anything" control.
3. A **dense model at matched active size by depth**: DistilGPT2 with the CODI recipe (the canonical one), or, more cheaply, the trained CODI with 6 layers dropped and then fine-tuned. ICoT's depth finding makes this informative, not redundant.
4. **Routing controls:** random/hash routing (Hash Layers, MoEBERT) and fixed step- or phase-indexed routing against learned routing.
5. **Training-free sparsity at the same fraction:** MoEfication or G-MoEfication with no fine-tuning, and TEAL-style top-magnitude neuron selection.
6. **Explicit-CoT control:** the same MoE conversion applied to explicit CoT (CODI's teacher mode), to show the effect is specific to latent reasoning.
7. **Reporting:** active non-embedding params (and with embeddings), total params, FLOPs per latent step and per answer token, measured wall-clock at batch 1 and batch 128 on the G4, load-balance statistics, ≥2 seeds, and test and validation splits as in the project's earlier stages.

---

## Open questions the literature does not answer

1. Do continuous latents, which are off-manifold input embeddings, route like tokens, like ST-MoE's sentinel tokens (one dedicated expert), or chaotically? Nobody has routed Coconut/CODI latents through an MoE.
2. Do the six latent steps use step-specific experts (an odd/even pattern matching CODI's known cycle, PMPS 2609.09928 [Proj]), or collapse to one set (LOOM's expert selection collapse)? Does step-indexed routing beat learned routing, given that hash routing was competitive in MoEBERT and Hash Layers?
3. Is latent reasoning more fragile under FFN sparsification than explicit CoT at the same active fraction? Mixture of Parrots and Optimal Sparsity concern pretraining, not post-hoc conversion of a distilled latent reasoner.
4. Which distillation target transfers latent reasoning to a sparse student: CODI's single-position L1, MoEBERT-style all-layer hidden-state MSE, or matching the latent vectors themselves? Can a dense CODI/SIM-CoT teacher supervise an MoE student across architectures?
5. Does SIM-CoT's step-decodability of latents survive sparsification? Do experts specialise by arithmetic operation or digit? No arithmetic-level routing analysis exists.
6. Does sparsity change the project's drift past 6 latent steps (better, worse, or unchanged)?
7. Is any real wall-clock gain possible at 124M for 6-step latent decoding on a G4? Every source suggests no without custom kernels, but none measured latent loops.
8. No GPT-2-scale MoE GSM8K number exists, so there is no prior for the accuracy cost at 25% or 50% FFN on GSM8K-Aug.

---

## Sources

- Coconut: https://arxiv.org/abs/2412.06769
- CODI: https://arxiv.org/abs/2502.21074 ; https://aclanthology.org/2025.emnlp-main.36/
- Soft Thinking: https://arxiv.org/abs/2505.15778
- ThinkRouter: https://arxiv.org/abs/2602.11683 ; https://arxiv.org/html/2602.11683v1
- System-1.5 Reasoning: https://arxiv.org/abs/2505.18962
- Think-at-Hard: https://arxiv.org/abs/2511.08577
- LOTUS: https://arxiv.org/abs/2606.31779
- Dreamer: https://arxiv.org/abs/2601.21582 ; https://arxiv.org/html/2601.21582
- MoLE: https://arxiv.org/abs/2610.01917
- MoLEM: https://arxiv.org/abs/2605.21951
- Mixture of Thoughts: https://arxiv.org/abs/2509.21164
- ICoT-KD: https://arxiv.org/abs/2311.01460
- Stepwise Internalization: https://arxiv.org/abs/2405.14838
- PADD: https://arxiv.org/abs/2606.10369 ; https://arxiv.org/html/2606.10369
- MoDE-CoTD: https://aclanthology.org/2024.lrec-main.1003/
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Every Expert Matters: https://arxiv.org/abs/2502.12947 ; https://arxiv.org/html/2502.12947v1
- DeepSpeed-MoE: https://arxiv.org/abs/2201.05596
- Switch Transformer: https://arxiv.org/abs/2101.03961
- Efficient Large Scale LM with MoEs: https://arxiv.org/abs/2112.10684
- Pruning and Distilling MoE into Dense: https://arxiv.org/abs/2605.28207
- MoEfication: https://arxiv.org/abs/2110.01786
- MoEBERT: https://arxiv.org/abs/2204.07675
- D2DMoE: https://arxiv.org/abs/2310.04361 ; https://arxiv.org/html/2310.04361v4
- G-MoEfication (EMNLP 2024): https://aclanthology.org/2024.emnlp-main.563.pdf
- LTE: https://arxiv.org/abs/2402.06126
- CMoE: https://arxiv.org/abs/2502.04416
- DS-MoE: https://arxiv.org/abs/2404.05567
- LLaMA-MoE: https://arxiv.org/abs/2406.16554
- LLaMA-MoE v2: https://arxiv.org/abs/2411.15708
- Sparse Upcycling: https://arxiv.org/abs/2212.05055
- Drop-Upcycling: https://arxiv.org/abs/2502.19261
- Upcycling LLMs into MoE (NVIDIA): https://arxiv.org/abs/2410.07524
- Scaling Laws for Upcycling: https://arxiv.org/abs/2502.03009
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- GPT-2 MoE upcycling repo: https://huggingface.co/sumitdotml/moe-emergence/blob/main/README.md
- MoEUT: https://arxiv.org/abs/2405.16039 ; https://arxiv.org/html/2405.16039
- Sparse Universal Transformer: https://arxiv.org/abs/2310.07096
- Mixture-of-Recursions: https://arxiv.org/abs/2507.10524
- Relaxed Recursive Transformers: https://arxiv.org/abs/2410.20672
- LOOM: https://arxiv.org/abs/2610.01153
- SMELT: https://arxiv.org/abs/2609.01343
- LatentLoop: https://arxiv.org/abs/2610.19975
- Tying the Loop: https://arxiv.org/abs/2606.16825
- Chain-of-Experts: https://arxiv.org/abs/2506.18945
- Mixture-of-Depths: https://arxiv.org/abs/2404.02258
- Mixture-of-Transformers: https://arxiv.org/abs/2411.04996
- Hash Layers: https://arxiv.org/abs/2106.04426
- eDiff-I: https://arxiv.org/abs/2211.01324
- Denoising Task Routing: https://arxiv.org/abs/2310.07138
- RICE (cognitive experts): https://arxiv.org/abs/2505.14681
- SAEM: https://arxiv.org/abs/2608.21614
- Phase-Aware MoE (agentic RL): https://arxiv.org/abs/2602.17038
- Does the Same Token Mean the Same State?: https://arxiv.org/abs/2606.22798
- Depth-Aware Sensitivity Analysis of MoE: https://arxiv.org/abs/2608.13565
- Beyond the Trace: https://arxiv.org/abs/2608.17638
- ST-MoE: https://arxiv.org/abs/2202.08906
- Mixtral: https://arxiv.org/abs/2401.04088 ; https://arxiv.org/html/2401.04088
- OLMoE: https://arxiv.org/abs/2409.02060
- Myth of Expert Specialization: https://arxiv.org/abs/2604.09780 ; https://arxiv.org/html/2604.09780v1
- The Expert Strikes Back: https://arxiv.org/abs/2604.02178
- Mixture of Parrots: https://arxiv.org/abs/2410.19034 ; https://arxiv.org/html/2410.19034
- Optimal Sparsity of MoE for Reasoning: https://arxiv.org/abs/2508.18672 ; https://arxiv.org/html/2508.18672
- σ-MoE (Approximating Two-Layer FFNs): https://arxiv.org/abs/2310.10837
- Aux-loss-free load balancing: https://arxiv.org/abs/2408.15664
- Towards MoE Deployment: https://arxiv.org/abs/2303.06182
- TEAL: https://arxiv.org/abs/2408.14690
- Deja Vu: https://arxiv.org/abs/2310.17157
- Minitron: https://arxiv.org/abs/2407.14679
- DistilGPT2 model card: https://huggingface.co/distilbert/distilgpt2
- Expert Choice Routing: https://arxiv.org/abs/2202.09368
- Latent-reasoning survey (context): https://arxiv.org/abs/2507.06203
