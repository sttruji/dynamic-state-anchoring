# Idea D: decode latent steps to discrete tokens every k steps and re-inject them (hybrid latent + discrete reasoning) for CODI / SIM-CoT

Notes compiled 2026-11-05. Each source is tagged with how much of it I read:
- [full] = full text read through a summarizing fetch (for long pages, the first ~100k characters).
- [abs] = abstract or landing page only.
- [snippet] = search-result snippet or third-party summary only. Verify these before citing.
- [prior] = taken from this project's earlier notes (`research_notes/Latent thought attention gating/*.md`) and not re-fetched today.
- [code] = I read the released code or checkpoint header myself today (HTTP range reads only; no model code was run).

Relation to idea D:
- *threatens novelty*: the paper does part of D already.
- *competes*: an alternative way to reach the same goal.
- *supports*: evidence that D's mechanism could work.
- *warns*: evidence that it could fail.

Project numbers (500-question GSM8K validation split, from this project's memory):
- CODI: 206 / 192 / 173 at 6 / 12 / 24 steps.
- SIM-CoT: 204 / 185 / 170.
- gpt2-sft: 42.4% (212/500).
- Our CODI-recipe explicit route: 41.0%.
- On the GSM8K test set, SIM-CoT GPT-2 scores 39.3% (the paper reports 42.6%), CODI-1B 55.5% and SIM-CoT-1B 55.7%.

Sources: [codi-recipe-static-run memory](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/codi-recipe-static-run.md) and [heldout-scale-results memory](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/heldout-scale-results.md).

## Q1. Which papers mix continuous latent thoughts with discrete tokens, and what GSM8K gains did they get?

### Takeaway
Hybrid latent and text reasoning is a crowded area. The designs split into four groups:
- **Latent prefix, then text:** Coconut stage-k models, Token Assorted.
- **Per-step soft mixtures:** HRPO, Soft Thinking.
- **Switching between explicit and latent mode:** SwiReasoning, Hybrid Reasoning Agents, ALAR.
- **Alternating latent blocks with model-written text steps:** SpiralThinker (Findings ACL 2026), HybridCoT (NeurIPS 2025 workshop).

The largest accuracy gains come from *trained* hybrids at 1B+ scale:
- Token Assorted: +6.0 over CoT on Llama-3.2-1B.
- SpiralThinker: +3.2 over the best retrained latent baseline, but the paper reports no explicit-CoT baseline.
- HRPO: +0.9 to +1.1 over GRPO.

Training-free hybrids on large models move GSM8K by about 0.1–0.2 points (Soft Thinking) or 1.5–2.8 points averaged over math/STEM (SwiReasoning). None of these papers runs a CODI-family model past its trained latent depth, and none uses a calculator inside a latent loop.

### Cited Findings
**Summary table (GSM8K or GSM8K-Aug accuracy, %; numbers as reported, each with its own baselines)**

| Method (id, venue) | Hybrid form | Model | Reported GSM8K | Baseline(s) in same table |
|---|---|---|---|---|
| Coconut (2412.06769) [full] | curriculum: stage k = k·c latents + remaining steps in text | GPT-2 | 34.1±1.5 | CoT 42.9; w/o curriculum 14.4; pause-as-thought 24.1; w/o thought 21.6; Pause 16.4; No-CoT 16.5 |
| Token Assorted (2502.03275) [full] | VQ-VAE latent codes replace the leftmost m CoT tokens (random m), then text | Llama-3.2-1B / 3B / 3.1-8B (MetaMathQA) | 48.7 / 73.8 / 84.1 | CoT 42.7 / 69.7 / 80.1; iCoT 10.5 / 17.3 / 29.6 |
| HRPO (2505.18454) [full] | each reasoning step's input = gated mix of the sampled token embedding and the hidden state; RL with outcome reward | Qwen2.5-1.5B / 3B | 72.0 / 84.5 | GRPO 71.1 / 83.4; PPO 69.4 / 81.9; CODI (1.5B) 65.8; Coconut (1.5B) 31.5 |
| Soft Thinking (2505.15778) [full, table] | training-free probability-weighted top-n embedding inputs + entropy "Cold Stop" | QwQ-32B / R1-Qwen-32B / R1-Llama-70B | 96.81 / 95.83 / 94.90 | CoT 96.67 / 95.61 / 94.82 |
| SwiReasoning (2510.05069; ICLR 2026 poster) [abs] | training-free block-wise switch between explicit and latent by entropy trend; capped switch count | "reasoning LLMs of different families" | not given per benchmark | +1.5 to +2.8 average accuracy on math/STEM; +56–79% token efficiency |
| System-1.5 (2505.18962; NeurIPS 2025 poster) [full, table] | language-to-latent distillation + depth/step shortcuts; fully latent at inference | GPT-2 and Llama-3.2-1B (which backbone this row uses was not confirmed) | 46.66 | CoT 46.94; CODI 43.78; Coconut 36.75; iCoT 32.14 |
| CoLaR (2505.16552) [full] | compressed latent embeddings with a random compression factor; latent-only at inference | Llama-3.2-1B-Instruct | CoLaR-2 40.1; CoLaR-5 26.8 | CoT 49.4; Coconut 23.1; Distill (CODI repro) 13.3; iCoT 19.8 |
| LightThinker (2502.15589; EMNLP 2025) [full] | explicit CoT compressed into gist/cache tokens, raw thought KV dropped | Qwen2.5-7B (R1-distill init) | 90.14 (thought-level) | Vanilla 90.90; H2O 89.92; AnLLM 78.39 |
| SpiralThinker (Findings ACL 2026) [full, PDF] | every odd- or even-position text step replaced by N latents; K iterative latent refinements; at inference alternates latent blocks and generated text steps | Llama-3.2-1B, LoRA | 56.56 | Pause 53.37; CODI 51.02; Coconut 49.85; Token Assorted 48.7 (copied from its paper); iCoT-SI 29.72. No explicit-CoT row |
| HybridCoT (NeurIPS 2025 Efficient Reasoning workshop) [abs] | interleaves latent and text tokens; "critical text tokens like math operations" stay as text | not stated | no GSM8K | 94% of text-CoT performance on AIME/MATH at 1.97× less compute |
| Thinking States (2602.08332, Feb 2026) [full] | generates NL thought tokens every chunk of input, compresses them, adds them to the next chunk's embeddings (prefill only) | Qwen2.5-1.5B | 42.22 (GSM-style set) | CoT 60.50; No-CoT 34.11; Coconut 32.65; iCoT 34.00 |
| Hybrid Reasoning Agents (2610.18861, Oct 2026) [full, first 100k] | text mode ↔ `<latent>` spans via a HybridLink adapter; Python + search tools in text mode | Qwen3-8B / 14B | no GSM8K (MATH500 84.6 / 88.2 after RL) | ARPO 82.8 / 87.0; Coconut 51.4 / 61.2 |

Sources for the table rows:
[Coconut](https://arxiv.org/html/2412.06769); [Token Assorted](https://arxiv.org/html/2502.03275); [HRPO](https://arxiv.org/html/2505.18454); [Soft Thinking](https://arxiv.org/html/2505.15778v1); [SwiReasoning abs](https://arxiv.org/abs/2510.05069v2), [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10007015); [System-1.5](https://arxiv.org/html/2505.18962v3), [NeurIPS 2025 poster](https://nips.cc/virtual/2025/poster/118475); [CoLaR](https://arxiv.org/html/2505.16552); [LightThinker](https://arxiv.org/html/2502.15589); [SpiralThinker](https://aclanthology.org/2026.findings-acl.1605.pdf); [HybridCoT](https://neurips.cc/virtual/2025/126670); [Thinking States](https://arxiv.org/html/2602.08332); [Hybrid Reasoning Agents](https://arxiv.org/html/2610.18861v1).

**Per-paper details that matter for D**
- **Coconut** [full]. At stage k the first k language steps become k×c continuous thoughts and the rest stay as language, so the intermediate-stage models are latent-prefix + text hybrids. On ProsQA, inference with k ∈ {0..6} thoughts and the rest in language works with the same weights (moving `<eot>`). This mixed inference is reported only for ProsQA, not GSM8K. Continuous thoughts are never mapped back to words during reasoning; decoding is "for probing purposes" only. *Context; supports "latent then verbalize" being usable without retraining in Coconut.* — [Source](https://arxiv.org/html/2412.06769)
- **Token Assorted** [full].
  - Compression is r=16 (16 CoT tokens per latent code); the codebook has 1024 entries.
  - At inference the model "directly generates latent tokens without any use of VQVAE".
  - Latents only ever form a prefix: there is no mid-generation interleaving and no decoding of latents back to text, except post-hoc inspection.
  - Official code is not released (per SpiralThinker's footnote).
  - *Competes ("latent then verbalize" with a trained model).*
  - [Source](https://arxiv.org/html/2502.03275); [SpiralThinker footnote](https://aclanthology.org/2026.findings-acl.1605.pdf)
- **HRPO** [full].
  - The hidden state is projected to a probability-weighted embedding, h_{t+1} = W_eᵀ·p_{t+1}/‖p_{t+1}‖.
  - It is then gated with the sampled token embedding: e_{t+1} = a_t⊙ê_{t+1} + √(1−a_t²)⊙(i_t⊙h_{t+1}).
  - Initialization makes token embeddings dominate.
  - So HRPO decodes a discrete token at every step and re-injects it alongside the latent. That is a per-step, soft version of D, trained by RL.
  - *Threatens novelty of "decode + re-inject" in its general form; HRPO is not CODI-style and not past-depth.*
  - [Source](https://arxiv.org/html/2505.18454)
- **Soft Thinking** [full, table]. The method is training-free. Top-n (5–30) probability-weighted embeddings are used as the next input, and Cold Stop ends thinking after k consecutive low-entropy steps (τ ∈ {0.01..0.2}, k ∈ {128..1024}). GSM8K changes are within +0.08 to +0.22 points. *Warns: training-free soft/hard mixing barely moves GSM8K.* — [Source](https://arxiv.org/html/2505.15778v1)
- **SwiReasoning** [abs]. It argues that training-free pure latent reasoning "spreads probability across multiple implicit paths", which adds noise and hinders convergence. Switching to explicit tokens when confidence rises is its fix. This is the closest training-free precedent for "periodically commit to discrete tokens", but on large reasoning LLMs. — [Source](https://arxiv.org/abs/2510.05069v2)
- **SpiralThinker** [full PDF via pdftotext].
  - Training data replace every odd- or even-position text step with N `<latent>` tokens wrapped in `<bol>`/`<eol>`.
  - A residual-MLP + RMSNorm adapter rewrites latent positions over K iterations.
  - An alignment loss matches the `<eol>` hidden states to a frozen explicit model's `<eot>` states, weighted toward later iterations.
  - At inference: "we prepend N <latent> tokens to each input question… subsequently produces explicit textual reasoning steps conditioned on the updated latent representations… learns to autonomously insert N <latent> tokens after each textual step."
  - Best settings are N=5 and K=5 on GSM8K-Aug. Ablation (GSM8K-Aug):
    - neither component: 45.49
    - alignment only: 48.67
    - iteration only: 45.72
    - both: 56.56
  - The authors note iteration alone "may drift when left unconstrained".
  - Code: github.com/shengminp/SpiralThinker.
  - *Threatens novelty of "think latently, write a step, think again".* — [Source](https://aclanthology.org/2026.findings-acl.1605/); [PDF](https://aclanthology.org/2026.findings-acl.1605.pdf)
- **HybridCoT** [abs].
  - Authors: Zejiang Shen, Rulin Shao, … Yoon Kim, Jason Weston, David Sontag, Scott Yih.
  - Track: NeurIPS 2025 Efficient Reasoning workshop poster. OpenReview id NRGRrHmq1H (an earlier ICLR 2026 OpenReview id is 4mfGbMzTwu; that page was blocked by a bot check).
  - Math operations stay as text and "semantic reasoning" is compressed into latents, with "in-context text-to-token distillation" and "iterative parallelized latent rollout".
  - *Supports the design choice of keeping arithmetic discrete; competes.*
  - [Source](https://neurips.cc/virtual/2025/126670)
- **Pause / filler tokens.**
  - In Coconut's GPT-2 table, Pause scores 16.4 and Coconut's "pause as thought" 24.1, against 34.1 with recurrent thoughts [full]. — [Source](https://arxiv.org/html/2412.06769)
  - In SpiralThinker's Llama-1B retraining, Pause scores 53.37, *above* their retrained CODI (51.02) and Coconut (49.85) [full]. That retrained CODI is below the 55.6 that CODI and KaVa report for Llama-1B [prior]. — [SpiralThinker](https://aclanthology.org/2026.findings-acl.1605.pdf); [KaVa](https://arxiv.org/html/2510.02312v1)
  - Pause-token fine-tuning treats a changed pause count at inference as "a severe test-time distribution shift" [prior]. — [Source](https://arxiv.org/html/2310.02226)
- **Other 2026 hybrids.**
  - **HyLaR** (2604.20328, Tencent). Visual latent "canvas" spans inside text with vMF-based RL, evaluated on perception benchmarks only. Nothing is decoded and re-injected [abs]. — [Source](https://www.alphaxiv.org/abs/2604.20328)
  - **ALAR** (2606.02871). Per-turn latent/explicit switching [prior, snippet]. — [Source](https://www.alphaxiv.org/abs/2606.02871)
  - **Switch-GRPO** (2606.13106). Boundary tokens between latent and text for GRPO; MATH-500 79.3% [prior, snippet]. — [Source](https://arxiv.org/pdf/2606.13106)
  - **AdaAnchor** (2603.15051). Refines latent anchor vectors and emits only the answer; per the search summary it does not decode intermediate tokens [snippet]. — [Source](https://arxiv.org/html/2603.15051v1)
  - **SPOT** (2603.06222). "Span-level Pause-of-Thought for Efficient and Interpretable Latent Reasoning" [title only]. — [Source](https://arxiv.org/pdf/2603.06222)

### Inferences
- Every paper with a real GSM8K gain from hybridization *trains* for the hybrid format: Token Assorted, SpiralThinker, HRPO. Training-free mixing on large models (Soft Thinking, SwiReasoning, Chang et al. below) gives about 0–3 points.
- For a training-free D on CODI/SIM-CoT GPT-2, a prior of "small or negative effect" is the honest default. The burden is on a Stage-0 oracle test (Q6).
- SpiralThinker's 56.56 on Llama-1B is about CODI's own published Llama-1B number (55.6) plus 1 point. Its +3.2 headline is measured against retrained baselines that look weak (CODI 51.02, and Pause above CODI).
- CODI's paper puts Llama-1B CoT-SFT well above CODI: CODI reaches "about 90%" of CoT-SFT on Llama-1B (Q3). So trained interleaving may mostly recover explicit-CoT accuracy rather than beat it. This could not be checked because SpiralThinker has no CoT row.
- HybridCoT and our idea share a design principle: arithmetic in discrete tokens, everything else latent. That is a reason to inject expressions, not only results (Q4).

### Gaps
- SwiReasoning's per-benchmark numbers (and whether GSM8K is included) were not visible on the abstract page.
- HybridCoT's full paper, base models and interleaving schedule were not readable (OpenReview bot check).
- I could not confirm which backbone System-1.5's GSM8K row (46.66) belongs to.
- The original Pause-token GSM8K numbers (1B model) were not re-fetched.
- Soft Thinking's and Token Assorted's main-conference venues were not verified.
- The SpiralThinker GitHub code was not inspected.

## Q2. Has anyone decoded intermediate results from latent thoughts at inference time and fed them back as tokens? Closest competitors

### Takeaway
No paper was found that does D exactly: periodic decoding of a CODI-family latent into an arithmetic step, optional calculator correction, and re-injection as discrete tokens to continue the loop past the trained depth. The mechanism has close neighbours:
- **SpiralThinker**: latent block, then a model-generated text step, then another latent block, trained.
- **Thinking States**: generate NL thought tokens, re-encode them, inject them; prefill only, trained.
- **HRPO**: a sampled token mixed into every latent input.
- **SwiReasoning**: training-free switch to explicit tokens mid-reasoning.
- **Chang et al. 2606.01243**: soft re-embedding of decoded distributions into CODI/Coconut latents, plus a theory of decode-then-re-embed as nearest-neighbour projection, with no hard-token test.

The defensible open slice is narrow: training-free use of the *released SIM-CoT step decoder* as an inference-time verbalizer, combined with calculator verification, as a fix for the past-depth repetition attractor.

### Cited Findings
Closest competitors, ranked by overlap with D:
1. **SpiralThinker** (Piao and Park, Findings ACL 2026) [full]. At inference it alternates N latent tokens with *generated* text steps until the answer. This is the "think latently, write a step, think again" scheme. Differences from D:
   - The text step is produced by the main model autoregressively in context, not decoded from the latent by a side decoder.
   - It is trained for the alternating format.
   - It uses no calculator and does not go past the trained depth.
   - **Threatens novelty of the interleaving schedule.** — [Source](https://aclanthology.org/2026.findings-acl.1605.pdf)
2. **Thinking States** (Amos, Caciularu, Geva, Globerson, Herzig, Shani, Szpektor; arXiv 2602.08332, Feb 2026) [full].
   - A Thinking Block autoregressively generates a natural-language thought from deep-layer chunk representations.
   - A Compression Block pools that thought into a state S, which is added to the next chunk's input embeddings at a shallow layer.
   - The thought tokens "never enter the backbone's context window". Performance drops when the chunk size goes beyond c=8.
   - It works on prefill only; the authors leave decoding-time use to future work.
   - It does not analyse error propagation from wrong decoded thoughts.
   - **Threatens novelty of "decode a latent into text and re-inject it".** — [Source](https://arxiv.org/html/2602.08332)
3. **HRPO** (2505.18454) [full]. Every latent input mixes a sampled discrete token with the hidden state, trained with RL. **Partial overlap (soft, per-step, trained).** — [Source](https://arxiv.org/html/2505.18454)
4. **Chang et al., "Unlocking the Black Box of Latent Reasoning: An Interpretability-Guided Approach to Intervention"** (arXiv 2606.01243) [full].
   - Setting: CoConut and CODI on Qwen3-8B, Llama-3.1-8B and Llama-3.2-3B, with K=6.
   - Interventions include weight-tying projection C.1: a soft re-embedding Eᵀsoftmax(o/τ) blended into the hidden state, +0.6 on Qwen3-8B CoConut (50.4→51.0).
   - Other interventions gain +0.4 to +1.8, with no variance reported.
   - The theory section frames decode-then-re-embed as nearest-neighbour projection onto the embedding codebook, but "does not test that discrete version".
   - **Closest training-free precedent on CODI.** The gains are small. — [Source](https://arxiv.org/html/2606.01243v1)
5. **SwiReasoning** (2510.05069) [abs]. A training-free switch from latent to explicit decoding mid-reasoning when entropy falls, i.e. "commit to tokens and continue", but on native-CoT LLMs. **Competes (training-free).** — [Source](https://arxiv.org/abs/2510.05069v2)
6. **Coconut stage-k / ProsQA mixed inference** and **Token Assorted**. A latent prefix followed by text continuation from the same model ("latent then verbalize"). **Competes.** — [Coconut](https://arxiv.org/html/2412.06769); [Token Assorted](https://arxiv.org/html/2502.03275)
7. **HybridCoT** (NeurIPS 2025 workshop) [abs]. Math operations kept as text between latents. **Competes on the "arithmetic stays discrete" design.** — [Source](https://neurips.cc/virtual/2025/126670)
8. **Hybrid Reasoning Agents** (2610.18861) [full, partial]. Latent spans plus text-mode Python tool calls. Latents are "not decoded during generation", only mapped to nearest tokens for inspection. **Closest "latent + tool" system.** — [Source](https://arxiv.org/html/2610.18861v1)
9. **Dilgren and Wiegreffe, "Are Latent Reasoning Models Easily Interpretable?"** (arXiv 2604.04902, COLM 2026) [abs]. A method decodes a *verified* natural-language trace from latents without a gold trace. It finds one for most correct predictions and only a minority of incorrect ones, so the decode can signal correctness. This is offline and not fed back. **Competes on the "self-check" use of decoding.** — [Source](https://arxiv.org/abs/2604.04902)
10. **LSRL** (Findings EMNLP 2025) [prior, snippet]. Decodes each recurrent depth of Huginn into a partial solution and scores it with an LLM grader as a process reward. This is a training signal, not re-injection. — [Source](https://preview.aclanthology.org/author-page-yu-wang-polytechnic/2025.findings-emnlp.669.pdf)

Explicit non-competitors:
- **CODI, Coconut, SIM-CoT and CoLaR** all state that latents are decoded only for interpretation and never fed back as tokens.
  - CODI "directly connects the last hidden representation to the subsequent input" [full].
  - SIM-CoT: "The auxiliary decoder is removed at inference" [full].
  - CoLaR's latent-to-vocabulary similarity is post-hoc [full].
  - Sources: [CODI](https://arxiv.org/html/2502.21074); [SIM-CoT](https://arxiv.org/html/2509.20317); [CoLaR](https://arxiv.org/html/2505.16552)
- I found **no paper combining latent CoT with a calculator.** Two searches turned up only explicit-CoT calculator work (Calc-X/Calcformers, EMNLP 2023) and Hybrid Reasoning Agents' text-mode Python. — [Calc-X](https://arxiv.org/pdf/2305.15017); [Hybrid Reasoning Agents](https://arxiv.org/html/2610.18861v1)

### Inferences
- "Hybrid latent/text" and "decode then re-embed" are both claimed in general form, so a paper cannot sell D as a new reasoning paradigm. It can sell three things none of the above have:
  - (a) a *training-free* verbalizer built from SIM-CoT's released step decoder;
  - (b) symbolic verification and correction of the decoded `<<a op b = c>>` before re-injection;
  - (c) its use specifically to break the past-depth repetition attractor this project has measured.
- SpiralThinker is the reviewer's first comparison. The fair version on GPT-2 GSM8K-Aug requires training it (code exists). That costs Colab credits. A cheaper fallback is to compare against a CODI-teacher "explicit continuation" arm, which is the training-free analogue of a latent block followed by text steps (Q4, R2).

### Gaps
- SPOT (2603.06222), AdaAnchor (2603.15051) and HybridCoT were not read in full. One of them could contain an inference-time decode-and-continue step.
- No direct OpenReview search was done for ICLR 2027 submissions (not yet public) or NeurIPS 2026 workshop papers.

## Q3. How decodable are CODI and SIM-CoT latents? Is SIM-CoT's step decoder released, and is it usable at inference?

### Takeaway
- **CODI latents** are only partly decodable by logit lens:
  - The paper reports 97.1 / 83.9 / 75.0% top-5 matches for 1 / 2 / 3-result problems, measured only on *correctly answered* items and only on the *first token* of multi-token numbers.
  - This project measured about 36–56% of number-phase GPT-2 latents matching a gold result.
  - PMPS finds no one-to-one map to steps.
- **SIM-CoT's decoder** maps each projected latent to a full step string ("<<a*b=c>>"). It *is* released inside `internlm/SIM_COT-GPT2-CODI`: a separately fine-tuned full GPT-2 (163.0M params, BF16) that takes only the latent as prefix, with no question context.
- **No one has reported its per-step decode accuracy.** The paper shows only qualitative examples.
- It is mechanically usable at inference with about 30 lines of code. It was never trained on latents past slot 7 or on latents beyond a problem's gold step count, so past-depth decodes are out of distribution.

### Cited Findings
**CODI**
- [full] Sec. 5.1 and Table 3. Decoding projects the last hidden state onto the vocabulary; Table 3 uses the top-5 decoded words. Measured on correctly predicted answers only, decoded intermediate results match the reference at 97.1% (1 result), 83.9% (2) and 75.0% (3). "Each intermediate result is separated by a seemingly meaningless continuous token." The Limitations section notes that the probe decodes only the first token of multi-token numbers. — [Source](https://arxiv.org/html/2502.21074)
- [full] Appendix D, Table A3 (GPT-2). Models fine-tuned to emit the *decoded* intermediate results as CoT score:
  - Res: 34.0
  - Op-Res: 35.7
  - CODI: 43.7
  - No-CoT: 19.1
  - So the decoded text carries less than the latents. *Warns: replacing latents with decoded text loses about 8–10 points.* — [Source](https://arxiv.org/html/2502.21074)
- [full] Out-of-domain results (GPT-2): CODI scores SVAMP 42.9, GSM-Hard 9.9, MultiArith 92.8, against CoT-SFT 41.8 / 9.8 / 90.7. On Llama-1B, CODI 61.1 / 12.8 / 96.1 against CoT-SFT 66.7 / 15.6 / 99.3. In-domain, CODI is about 99% of CoT-SFT on GPT-2 and about 90% on Llama-1B (the exact in-domain values are in the paper's Fig. 3 only). CODI appears in the EMNLP 2025 Anthology. — [Source](https://arxiv.org/html/2502.21074); [Anthology PDF](https://aclanthology.org/2025.emnlp-main.36.pdf)
- [prior] This project's Stage 0. About 36–56% of number-phase GPT-2 CODI latents match a gold intermediate result, against about 4% by chance, and a gate trained on these decode labels failed. — [Source](/Users/stevotrujillo/Desktop/DSALT/dynamic-state-anchoring/research_notes/Latent%20thought%20attention%20gating/latent_cot_depth_supervision.md)
- [prior] PMPS (2609.09928): logit-lens readouts "do not establish a one-to-one mapping to explicit CoT steps". CODI shows odd-even collapse and SIM-CoT shows tail collapse. — [Source](https://arxiv.org/html/2609.09928)
- [prior] Cywiński et al. (LessWrong, Dec 2025), CODI Llama-3.2-1B. Logit lens shows the step-1 value in latent 3 and the step-2 value in latent 5 "with high top-1 probability", including on wrong answers. Patching those latents is causal; the last two latents are not. — [Source](https://www.lesswrong.com/posts/YGAimivLxycZcqRFR/can-we-interpret-latent-reasoning-using-current-mechanistic)
- [prior] LessWrong analyses: plain logit lens recovers CODI intermediate values "weakly but significantly", mainly in late layers. — [Source](https://www.lesswrong.com/posts/yetmqjCXEav9iaJGw/what-s-in-codi-s-latent-recurrent-thoughts)
- [full] Chang et al. 2606.01243 (3B–8B CODI/CoConut): linear CKA 0.72; mapped cosine 0.75; *lexical top-1 token accuracy 0.38* ("above random"). There is no per-step breakdown. — [Source](https://arxiv.org/html/2606.01243v1)
- [abs] Dilgren and Wiegreffe (COLM 2026). Where latents are necessary, gold reasoning traces can be decoded "65–93% of the time for correctly predicted instances". On logical-reasoning datasets the two latent models "can almost always reach the same final answers without using latent reasoning". — [Source](https://arxiv.org/abs/2604.04902)
- [prior] "Observable Patterns Are Not Explanations" (2606.12689) argues that decodable intermediate values are over-read as a compressed scratchpad. — [Source](https://arxiv.org/pdf/2606.12689)

**SIM-CoT paper**
- [full] The decoder is "architecturally identical to the LLM". Its input is [z_k; e(y_k,1)…e(y_k,L_k)], with loss λ_step·L_step + λ_lm·L_ans; the λ values are not stated. "The auxiliary decoder is removed at inference", yet Sec. I.1 reuses it "at inference time" to visualise latents (Fig. 4: 0.3×120=36, 120−36=84, …).
- **No per-step exact-match decode accuracy is reported.**
- App. K: trailing latents "no longer introduce new steps but simply repeat the final prediction".
- GPT-2 Table 1: CODI 42.0 → CODI+SIM-CoT 42.6; Coconut 36.6 → 44.8; SFT-CoT 42.7.
- [Source](https://arxiv.org/html/2509.20317)

**Released decoder (checkpoint header and code)**
- [code] Header of `internlm/SIM_COT-GPT2-CODI@8685b42`, read by HTTP range request (no weights loaded):
  - The file has three groups: `codi.*` (181.9M params), `prj.*` (1.18M) and `decoder.*` (163.0M).
  - `decoder.*` is 149 tensors: a full GPT-2 (12 blocks + wte + wpe + ln_f) plus a separate `decoder.lm_head.weight` [50257, 768]. All BF16, with no LoRA tensors.
  - Small decoder tensors differ from base GPT-2 well beyond BF16 rounding (h.0.ln_1.weight mean |Δ| 0.078; h.11.ln_2.bias mean |Δ| 0.122). So the decoder is a *trained*, full-parameter copy, not frozen GPT-2.
  - [Checkpoint](https://huggingface.co/internlm/SIM_COT-GPT2-CODI); local loader that currently drops `decoder.*`: [fetch_simcot.py](/Users/stevotrujillo/Desktop/DSALT/dynamic-state-anchoring/scripts/latent_gate/fetch_simcot.py)
- [code] How the decoder is trained, from SIM-CoT `CODI/src/model.py` (GPT-2 branch):
  - **Targets:** `get_steps(ref_input_ids, num_latent+1, start_ids=(16791, 9959), end_id=4211, eot_id=eos)` pulls each `<<…>>` span (GPT-2 ids 16791='<<', 9959=' <<', 4211='>>') and appends EOS (50256).
  - **Slots:** there are num_latent+1 = 7 step slots. Steps beyond slot 7 are *merged* into the last slot. Problems with fewer steps get pad rows, whose labels are −100, so **latents past the gold step count receive no decoder loss**.
  - **Decoder input:** `concat([latent_embd, codi_wte(step_ids)])`. `latent_embd` is the post-`prj` vector (the same vector that is fed as the next latent input). The step tokens are embedded with the **CODI model's** input embedding, not the decoder's own wte.
  - **Construction:** without `decoder_path`, the decoder is `from_pretrained(model_name)`, i.e. a fresh GPT-2 copy.
  - **Training flag:** the GPT-2 script trains with `--use_decoder True --num_latent 6`, LoRA r=128 on the CODI model, bf16, 40 epochs, lr 3e-3.
  - [model.py](https://raw.githubusercontent.com/InternLM/SIM-CoT/main/CODI/src/model.py); [train script](https://raw.githubusercontent.com/InternLM/SIM-CoT/main/CODI/scripts/train_gpt2_gsm8k-aug-decoder-2.sh)
- [code] SIM-CoT's `CODI/test.py` never references the decoder, and its README documents no latent-to-text inference script. `probe_latent_token.py` applies the logit lens (top-5 of `codi.lm_head`) to the *pre-projection* hidden state. — [test.py](https://raw.githubusercontent.com/InternLM/SIM-CoT/main/CODI/test.py); [probe script](https://raw.githubusercontent.com/InternLM/SIM-CoT/main/CODI/probe_latent_token.py); [README](https://github.com/InternLM/SIM-CoT)

### Inferences
- **The decoder conditions on the latent vector alone.** It sees no question or context, so any number it emits must be stored in that one 768-d vector. Two consequences:
  - (i) It is a stronger readout than the logit lens: it produces multi-token, full expressions.
  - (ii) Changing the CODI context by injecting tokens does not change the decoder's input format, only the latent distribution.
- **Slot alignment** (inferred from the code, not traced line by line): slot 0 = prj(h_<bot>), slot t = prj(h after latent step t), for t = 1..6. Slot 6 is the projected latent that plain 6-step inference computes and then discards.
- **Which latents the decoder can be trusted on:** about slots 0–6, and only those whose index is below the problem's step count. Trailing latents and every latent past step 6 were unsupervised. Decodes there are a measurement of what the attractor contains (App. K says it is the final answer), not a reliable verbalization.
- **Usability.** It is usable at inference: load `decoder.*` into a GPT2LMHeadModel (untied lm_head), upcast to FP32, feed `[latent; codi_wte(generated ids)]` as `inputs_embeds`, and greedy-decode until EOS or about 24 tokens. The cost is about 7 × 10 decoder forwards per question, which is negligible.
- **Reliability is unknown and must be measured first.** The cheap metrics are per-slot parse rate, expression and result exact match against the gold k-th step, and arithmetic self-consistency (eval(expr)==res).
- **For CODI (zen-E/CODI-gpt2) there is no decoder.** The logit lens yields only the first token of a number and cannot produce expressions. D on CODI would need a post-hoc decoder trained on frozen CODI latents. That is cheap, since only the decoder trains and the latents can be cached, but the odd/even alternation makes per-slot labels messy.

### Gaps
- No published per-step decode accuracy for the SIM-CoT decoder (GPT-2 or Llama). This is an open measurement, and it is itself a small, useful result.
- I did not confirm whether SIM-CoT's Llama-1B/3B checkpoints in the HF collection (Wiselnn / internlm) also ship `decoder.*`. The project's SIM-CoT-1B port exists, so its header can be checked the same way.
- How the decoder handles multi-step merges in slot 7, and what it emits for unsupervised trailing latents, is not documented.

## Q4. Implementation design: what to decode, where to insert, training-free or trained, calculator, and risks

### Takeaway
Start with **SIM-CoT GPT-2 plus its released decoder**, training-free, in this order:
1. **R0.** Decode-only diagnostics, plus a *gold-step oracle injection* test.
2. **R1.** Decode-verify-append into the KV cache.
3. **R2.** Latent-then-explicit continuation in CODI's own teacher mode, with a calculator.
4. **R5.** Decode-verify-*select* over depths, with no re-injection.

Decode the full `<<a op b=c>>` expression, not only the result. That is the decoder's and the CODI teacher's native format, and the calculator needs the expression.

Training (a SpiralThinker-style alternating LoRA fine-tune) should wait until the oracle test shows the model can use mid-loop tokens.

The main risks are:
- error hardening when a wrong decode is injected (CODI App. D: decoded text < latents);
- distribution shift, since the student never saw tokens between `<bot>` and `<eot>`;
- on 1B models, the extra positions the injected tokens add, because this project found RoPE answer distance is the dominant depth cost there.

### Cited Findings
- **The teacher format is in-distribution.** CODI trains teacher and student in one model with shared weights. The teacher reads question + explicit CoT (`<<…>>` steps) + answer. So step tokens in GSM8K-Aug format are text the model knows; only their placement after latents is new [prior/full]. This project's CODI-recipe explicit route scores 41.0% on validation. — [CODI](https://arxiv.org/html/2502.21074); [project memory](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/codi-recipe-static-run.md)
- **The calculator format is native to GSM8K.** GSM8K's `<<48/2=24>>` annotations were introduced for calculator use at sampling time (Cobbe et al. 2021) [not re-fetched this session]. Calc-X/Calcformers (EMNLP 2023) trains models to emit calculator calls whose outputs are inserted into context, and reports roughly double the accuracy of correct results over baselines on its suite [snippet]. No latent-CoT calculator work was found. — [GSM8K](https://arxiv.org/abs/2110.14168); [Calc-X](https://arxiv.org/pdf/2305.15017); [Calc-X Anthology](https://aclanthology.org/2023.emnlp-main.742)
- **HybridCoT keeps math operations as text** to reduce latent "compression errors". — [Source](https://neurips.cc/virtual/2025/126670)
- **Decoded text loses information.** CODI App. D: decoded results (or operators+results) used as CoT score 34.0 / 35.7, against 43.7 for latents. — [Source](https://arxiv.org/html/2502.21074)
- **Latents can carry several candidates at once.** Coconut reads continuous thoughts as encoding several frontier nodes (BFS-like values). Hard decoding collapses that. — [Source](https://arxiv.org/html/2412.06769)
- **Distribution shift from changing the inserted-token pattern is real.** Pause tokens show it [prior]. SpiralThinker's iteration-only arm drifts without alignment (45.49 → 45.72 only) [full]. — [Pause](https://arxiv.org/html/2310.02226); [SpiralThinker](https://aclanthology.org/2026.findings-acl.1605.pdf)
- **This project's earlier findings** [prior]:
  - Answer-only fine-tuning at 12 steps finds no use for steps 7–12.
  - Soft dedup holds the 6-step score at 24 steps but never raises it.
  - At 1B, RoPE answer position causes most of the depth loss: −2.2 / −2.4 points, and the shift alone halves the answer's attention on latents 1–6.
  - `dedup_anchor`@24 holds depth on all four models with no gain.
  - [heldout memory](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/heldout-scale-results.md); [SIM-CoT past depth memory](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/simcot-past-depth.md)
- **Code fit** [code, local, read-only]:
  - `CODIGPT2._answer` runs `model(inputs_embeds=latent, past_key_values=cache)` and then `latent = projection(hidden)`. The projected `latent` is exactly the SIM-CoT decoder's prefix input.
  - `stage1_eval.py` passes explicit `position_ids` (question + step).
  - Its bias policies assume latent entries are contiguous at cache indices `question+1 … question+len(fed)`. Injected tokens would break that indexing, so position and index bookkeeping must change for any combined dedup + inject arm.
  - `fetch_simcot.py` discards `decoder.*`; a variant must save it.
  - [codi.py](/Users/stevotrujillo/Desktop/DSALT/dynamic-state-anchoring/dsa/bench/models/codi.py); [stage1_eval.py](/Users/stevotrujillo/Desktop/DSALT/dynamic-state-anchoring/scripts/latent_gate/stage1_eval.py); [fetch_simcot.py](/Users/stevotrujillo/Desktop/DSALT/dynamic-state-anchoring/scripts/latent_gate/fetch_simcot.py)

### Inferences (concrete recipes; all are my design proposals, not results)
**R0: decode-only diagnostics and oracle injection.** Training-free, about 1 Colab T4-hour or less.
- (a) For SIM-CoT GPT-2 at 6 / 12 / 24 steps on the 500-question validation split, decode every slot.
- Report per slot:
  - parse rate of `<<expr=res>>`;
  - exact match of expr and res to gold step t (positional) and to any gold step (order-free);
  - arithmetic consistency, eval(expr)==res;
  - the fraction of slots past the gold step count whose decode equals an earlier decode or the final answer (quantifying the attractor).
- (b) On SIM-CoT's wrong answers, split the errors into three kinds:
  - "decoded chain right, readout wrong";
  - "chain has an arithmetic slip that a calculator fixes";
  - "wrong plan".
  - **Go/no-go:** the calculator-fixable share bounds the gain of the calculator arms. If it is below about 15 questions out of 500 (3 points, about 1.4 SE), drop the calculator angle.
- (c) **Oracle test, decisive and cheap.** Inject the *gold* first m steps (m = 1, 2, 3) as tokens after latent 2m and continue the latent loop to 12 / 24 steps. Compare with plain@6 and dedup@24.
  - If even gold steps do not raise accuracy above plain@6, the model cannot use mid-loop tokens training-free. In that case D needs training (R4) or should be abandoned.
  - Also inject *shuffled* or *wrong-number* steps as a control for "any tokens break the attractor".

**R1: decode-verify-append into the cache.** Training-free.
- Every k latent steps (k=1 for SIM-CoT, whose slots map one step per latent; k=2 for CODI's odd/even alternation), decode the step from the projected latent.
- If it parses and is *new* (not a string-duplicate of an earlier decoded step), then:
  1. optionally recompute `res` with a safe arithmetic evaluator (calculator arm);
  2. feed " <<expr=res>>" through CODI as ordinary token ids, with continuing position ids, appending about 6–12 KV entries;
  3. take the hidden state at the last injected token, project it, and continue the latent loop.
- If the decode is a duplicate or the final answer, stop the loop and read out. This **decode-based halting** is a free baseline that should at least match dedup@24.
- Arms: no-calc / calc / result-only (" 12") / expression-only.

**R2: latent then explicit, in teacher mode with a calculator.** Training-free.
1. Run 6 latents.
2. Decode slots 0–6, verify each, and keep the longest prefix of consistent, non-duplicate steps.
3. Write them after the question in teacher format.
4. Let CODI's explicit (teacher) mode generate the remaining steps and the answer, with a Calc-X-style override whenever it emits `=` inside `<<…>>`.
- Everything after the decoded prefix is in-distribution, so this arm tests "latents as a fast first half of explicit CoT".
- Its natural ceiling is explicit CoT + calculator. It must be compared with **gpt2-sft + the same calculator**, or a reviewer will attribute the gain to the calculator alone.

**R3: restart the latent loop.** Training-free, likely out of distribution.
- Build question + decoded steps (text) + `<bot>`, then run 6 fresh latents.
- This gives the latent loop genuinely fresh input and bounded depth per segment (it sidesteps dilution, because each segment is 6 latents). The student never saw text before `<bot>`, though.
- Cheap to try within R0's harness.

**R4: trained hybrid.** Only if R0(c) shows partial uptake.
- LoRA fine-tune SIM-CoT on a mixed format: q `<bot>` z1..zk " <<gold step>>" z_{k+1}… `<eot>` answer. Keep CODI distillation and SIM-CoT step losses on the latents, and teacher-force gold steps at random positions (Token Assorted-style random mixing).
- This is close to SpiralThinker, so the novelty must come from the decoder-verifier and the depth extension.
- This project's earlier 256-update Colab runs were too short to move anything, so budget for thousands of updates [prior].

**R5: decode-verify-select across depths.** Training-free, with no injection.
- Run depths 6, 8, …, 24 (one pass gives every prefix depth). Decode each depth's chain, and score its arithmetic consistency and agreement with the readout answer.
- Choose the answer from the best-verifying depth, or majority-vote over verified depths.
- This uses Dilgren-and-Wiegreffe-style "verified decode ⇒ likely correct" as a self-check with zero distribution shift. It is the most likely way to get *any* gain above plain@6, because it can only pick among existing answers.
- Competes with Best-of-N latent rerankers (e.g. LatentRM [snippet, unverified attribution]). — [2510.07745](https://arxiv.org/html/2510.07745)

**What to decode.** Use the full expression `<<a op b=c>>`. It is what the decoder was trained to emit, what CODI's teacher reads, and what a calculator can check, and it carries operands and operator, which result-only tokens do not. Run result-only as an ablation.

**Where to insert.** Append (R1) keeps the latents' extra information, which CODI App. D says matters, but it is out of distribution. Teacher continuation (R2) is in distribution but becomes explicit CoT. Restart (R3) is in between. Measure all three in R0 with gold steps before using decoded ones.

**Risks to pre-register:**
1. **Error hardening.** A wrong decoded step injected as text turns a soft, possibly multi-candidate latent into a confident wrong token. Mitigate by injecting only arithmetic-consistent, non-duplicate decodes.
2. **Distribution shift.** Student latents after injected tokens may be far from training; check latent norms and cosine to the 6-step trail.
3. **The model ignores the tokens.** The prior answer-only fine-tune found no use for steps 7–12.
4. **The decoder is unreliable on slots ≥ 7 and on trailing latents**, because those got no decoder loss.
5. **The calculator rarely fires**, because most GPT-2 errors are plan errors (unknown; R0b measures it).
6. **Position cost at 1B.** Each injected step adds about 8 positions between the question and the answer. This project found RoPE answer distance dominates the 1B depth loss, so injection may *hurt* LLaMA-1B even if it helps GPT-2. Consider re-using or compacting position ids, as the `shift`/`keep_first` analysis did.
7. **BF16/FP32 mismatch.** The decoder is stored in BF16 and our CODI port runs FP32; upcast the decoder and check that ties and argmax are stable (cf. this project's earlier bf16 tie bug [prior]).

### Gaps
- No source measures how often GPT-2 latent-CoT errors are arithmetic slips versus plan errors. R0b would be the first measurement.
- Whether CODI's teacher mode in the *released* zen-E/CODI-gpt2 and SIM-CoT checkpoints generates well-formed `<<…>>` CoT has not been checked here. The 41.0% figure is from this project's own CODI-recipe model, not the released checkpoints.
- There is no published evidence on whether a CODI student tolerates discrete tokens mid-loop. R0c is untested territory.

## Q5. Do iterative "think latently, write a step, think again" schemes beat a single latent block of the same total compute?

### Takeaway
No paper found runs a compute-matched comparison of an interleaved schedule (k latents → text step → k latents) against one 2k+L latent block for CODI-family models. The available evidence is indirect:
- Trained interleaving with alignment beats single-pass latents with the *same* interleaved data scheme (SpiralThinker, +11 points with alignment and iteration, +0.2 with iteration alone).
- Trained latent-prefix-then-text beats full-text CoT (Token Assorted, +6 on Llama-1B).
- Training-free switching gives +1.5–2.8 points on large models (SwiReasoning).
- Simply adding latent steps or iterations past the trained count hurts (CODI, PMPS, PCCoT, KaVa; this project's 206 → 173).

So "interleaving beats more latents" is plausible past depth 6 (more latents is negative there), but "interleaving beats explicit CoT at equal compute" is unproven.

### Cited Findings
- SpiralThinker ablation, GSM8K-Aug, Llama-1B. Every arm uses the alternating text–latent data, so the comparison is about iteration and alignment, not about interleaving versus no interleaving:
  - neither: 45.49
  - alignment only: 48.67
  - iteration only: 45.72
  - both: 56.56
  - Extra iterations past K=5 lead "to saturation or slight degradation". — [Source](https://aclanthology.org/2026.findings-acl.1605.pdf)
- Coconut: recurrent thoughts (34.1) beat the same number of pause positions (24.1), and the curriculum matters (14.4 without it). No interleaved inference on GSM8K. — [Source](https://arxiv.org/html/2412.06769)
- Token Assorted: a latent prefix plus a text suffix beats full-text CoT at lower token cost (Llama-1B 48.7 vs 42.7). — [Source](https://arxiv.org/html/2502.03275)
- Thinking States, with decoded thoughts re-injected every chunk, stays far below CoT (42.22 vs 60.50) at a 2.66× speedup. — [Source](https://arxiv.org/html/2602.08332)
- HybridCoT: 94% of text-CoT performance at 1.97× less compute (efficiency, not accuracy). — [Source](https://neurips.cc/virtual/2025/126670)
- More latents or iterations past the trained count hurt [prior]:
  - CODI Fig. 5 (training count);
  - PMPS ("CODI drops sharply at 8");
  - PCCoT and KaVa iteration sweeps;
  - this project's inference sweep: CODI 206/192/173, SIM-CoT 204/185/170.
  - [latent_cot_depth_supervision.md](/Users/stevotrujillo/Desktop/DSALT/dynamic-state-anchoring/research_notes/Latent%20thought%20attention%20gating/latent_cot_depth_supervision.md)
- Huginn (recurrent depth, random depth in training) keeps improving GSM8K with more recurrence (47.23% at r=64) [prior]. So depth helps when training randomizes it. — [Source](https://arxiv.org/html/2502.05171v2)

### Inferences
- For D the relevant comparison is not "interleaved vs one long latent block". Past depth 6 the long block is known to lose, so beating it is easy and not publishable alone. Beating **plain@6** and **gpt2-sft** at matched or lower forward passes is what counts.
- Count injected tokens as forward passes. R1 at 24 latents + 3 injected steps × about 8 tokens is about 48 passes, which is more than explicit CoT's about 25 generated tokens (Coconut's GSM8K table lists CoT at 25.0 tokens). Report accuracy against forward passes.

### Gaps
- No compute-matched interleaving study exists for CODI, SIM-CoT or Coconut at GPT-2 scale. This is a measurable gap that R0c/R1 would partly fill.

## Q6. Most convincing result, novelty, expected gain, and main ways it could fail

### Takeaway
**The most convincing paper result** is a *training-free* decode → verify → inject (or select) loop on SIM-CoT that:
- raises accuracy at depth > 6 *above* the 6-step score on held-out GSM8K test (1,319) and the 500-question validation split;
- does so with paired significance (McNemar or bootstrap), concentrated on ≥4-step problems;
- survives four controls: calculator-only applied to gpt2-sft and to the SIM-CoT readout, shuffled-step injection, compute-matched dedup@24, and a gold-step oracle upper bound;
- replicates on SIM-CoT-1B.

**Expected gain**, my estimate, not a measured or cited figure:
- R1 without calculator: −3 to +1 points.
- R1 or R2 with calculator: 0 to +3.
- R5 (select): 0 to +2.

**Novelty** is partial. Hybrid interleaving (SpiralThinker, HybridCoT, Token Assorted, HRPO) and decode-then-re-embed (Thinking States, Chang et al.) are taken. The defensible novelty is the released-decoder verbalizer, symbolic verification and correction inside a latent loop, and the past-depth framing.

### Cited Findings
- **Noise floor.** For a 500-question accuracy near 41%, the binomial SE is √(0.41·0.59/500) ≈ 2.2 points, about 11 questions. On the 1,319-question test it is about 1.35 points. (Computed; standard binomial formula.)
- **Training-free gain sizes in the literature:**
  - Soft Thinking on GSM8K: +0.08 to +0.22 points.
  - Chang et al. interventions on CODI/CoConut: +0.4 to +1.8 (no variance).
  - SwiReasoning: +1.5 to +2.8 average.
  - [Soft Thinking](https://arxiv.org/html/2505.15778v1); [Chang et al.](https://arxiv.org/html/2606.01243v1); [SwiReasoning](https://arxiv.org/abs/2510.05069v2)
- **GPT-2 ceiling context.** CODI already reaches about 99% of CoT-SFT on GPT-2 (paper). On this project's validation split, gpt2-sft scores 212 against CODI 206 and SIM-CoT 204. — [CODI](https://arxiv.org/html/2502.21074); [project memory](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/codi-recipe-static-run.md)
- **This project's prior repairs** hold depth but never add accuracy: soft dedup, dedup_anchor, and the gate (G2 fails, no depth beats 6). — [heldout memory](/Users/stevotrujillo/.claude/projects/-Users-stevotrujillo-Desktop-DSALT/memory/heldout-scale-results.md)
- **Verified decodes signal correctness.** A verified trace is found for most correct and only a minority of incorrect predictions (Dilgren and Wiegreffe), which supports R5. — [Source](https://arxiv.org/abs/2604.04902)

### Inferences
**Ways it could fail, in rough order of likelihood:**
1. **The oracle fails.** Gold-step injection does not lift accuracy, because the CODI/SIM-CoT student ignores or is confused by mid-loop tokens. This is consistent with "answer-only FT finds no use for steps 7–12". It would kill training-free D and push it to R4, which costs credits and overlaps SpiralThinker.
2. **The gain is only the calculator.** If gpt2-sft + calculator gains as much, the contribution collapses to "use a calculator".
3. **The decoder is unreliable** where it is needed: slots ≥ 7 are unsupervised, and early slots on wrong answers are unknown. Injected errors then cost more than the calculator saves.
4. **Gains are within noise**, under ±2.2 points on 500 questions.
5. **The 1B port is cancelled out by RoPE position cost** from the added tokens.
6. **Reviewers call it SpiralThinker / Thinking States** with a calculator.

**Decision rule** (proposal):
- Run R0 first. It is all training-free and about an hour of T4 time.
- Proceed to R1/R2 only if (a) gold-step oracle injection beats plain@6 by ≥ 2 points with McNemar p < 0.1 on 500 questions, *or* (b) the calculator-fixable share of SIM-CoT errors is at least 15 questions.
- Otherwise, report the decoder-reliability measurement and the decode-based halting / R5 selection as small, honest results, and drop D as an accuracy method.
- R5 is the low-risk fallback, because it cannot do worse than its best constituent depth once a held-out split chooses the selection rule.

### Gaps
- There are no published numbers to anchor the expected-gain ranges for D on CODI-family GPT-2. They are judgement calls built from the training-free hybrid literature and this project's own results.
- Whether SIM-CoT's Llama-1B checkpoint ships a decoder (needed for the port) was not verified.
