# Self-stated limitations and future directions: latent-reasoning hybrids and analyses of latent CoT models (9 papers)

Group: LOTUS, Penelope, AGCLR, Pause tokens (Goyal et al.), Liang & Pan, Chang et al., Li et al., Duraipandian et al., Dilgren & Wiegreffe.

## Per-paper records: sections used, stated limitations (L), stated future directions (F), theme tags

### Takeaway
I read the full text of all nine papers, using the latest arXiv HTML version of each. All nine arXiv IDs match the expected papers. Counting every tallied statement, task breadth (T2) is the most common theme, in 7 of 9 papers. Next come model scale (T1) and fixed-vs-adaptive compute (T5), 6 each, then extrapolation past trained depth (T7), architecture generality (T16) and evaluation rigor (T10), 5 each. No paper states test-time scaling or verification (T11), and none states hybrid or tool integration (T15). Counting only statements about each paper's own study, T2 is still first (7), and T1, T7, T10 and T16 tie at 5. Only two papers have a dedicated Limitations section (Chang et al. and Li et al.). The Pause paper has a combined "Conclusion, Limitations and Future Work" section, and two papers (Duraipandian et al., Dilgren & Wiegreffe) have no limitations section at all.

### Cited Findings

#### Tagging conventions used (for merging with other groups)
- There is one row per statement. When a statement is both a limitation and a direction, it is listed once as L and once as F.
- **Phrasing rule:**
  - "X remains future work" is recorded as F only.
  - "X remains unexplored", "remains to be validated" or "is limited to" is recorded as L.
  - When the authors state both, both rows are recorded.
- **Scope rule (this group has five analysis papers):**
  - Rows not marked otherwise are about the paper's own study or method.
  - Rows marked "scope: field" are limitations or directions that the authors state about latent CoT methods in general (for example CODI or Coconut). They count only when they appear in the paper's Limitations, Discussion, Conclusion or a results or appendix paragraph that frames them as a limitation or open problem.
  - Critiques of prior work that only motivate the paper in its Introduction or Related Work are **not** tallied. They are listed under Inferences.
  - The second tally below excludes the field rows, for merging with groups that counted only self-limitations.
- **Weak fits:** a tag marked "(weak fit, not tallied)" is my closest guess when no tag really fits. It is shown for transparency but excluded from both tallies.
- Generic statements with no theme are omitted. Examples: Pause §1, "Our work raises a range of conceptual and practical future research questions…"; Chang §5, "offering a robust foundation for future latent reasoning paradigms".
- **Rendering of quotes:**
  - Quotes are verbatim from the arXiv HTML. "..." marks an elision.
  - Math is rendered as plain letters, for example "K, c, and R" for {K}, {c}, {R}.
  - The arXiv HTML renders the pause token as "¡pause¿"; I write it as `<pause>`, as in the PDF.
  - I checked every quote segment against the downloaded full text with a script. All match, apart from whitespace around math.
- **NEW tags defined in this group:**
  - **NEW: latent-underuse** — latent steps go unused: the model fixes its answer before or without latent reasoning, or a no-CoT model matches the latent model (Dilgren, field-level only).
  - **NEW: latent-intervention** — editing or steering latent states at inference time to raise accuracy (Duraipandian F). The closest existing tag is T11. Chang et al.'s method is exactly this, but they do not state it as a limitation or direction.
  - **NEW: latent-token-design** — varying the form or number of distinct latent or pause token types (Pause F).
- **Rules shared with the "latent CoT training methods" group file:**
  - "Other pretraining mixtures and objectives" (Pause) is tagged T12, because it concerns objectives, not scaling to pretraining.
  - Liang & Pan's "hybrid latent-text approaches" names an architecture to analyze next, so it is tagged T16, not T15. A merger may add T15.

---
#### 1. LOTUS — "Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers" (Fan, Svete, Lee)
- arXiv: 2606.31779. I verified the title; the arXiv title has no "LOTUS:" prefix, but the method is named LOTUS throughout.
- Venue/date: arXiv preprint, no venue in the comments field. v1 is dated 30 Jun 2026 and v2 13 Jul 2026.
- Version read: **v2**, the latest — [arXiv HTML v2](https://arxiv.org/html/2606.31779v2).
- Sections used:
  - **No standalone Limitations section.** There is a "Limitations." paragraph inside **§6 Conclusion**, which is the main source.
  - Self-stated caveats also come from §3.4 (LOTUS-aux), §4.2 (main results and latency) and §4.3.2 (inference-time R ablation).
  - I scanned Appendices A–F and found no further limitation statements.

| quote | section | L/F | tags |
|---|---|---|---|
| "We follow prior latent reasoning work (Hao et al., 2025; Shen et al., 2025; Wei et al., 2025) and evaluate on math benchmarks." | §6 Conclusion, "Limitations." ¶ | L | T2 |
| "Whether the recipe transfers to other domains remains an open direction for future work." | §6 Conclusion, "Limitations." ¶ | F | T2 |
| "The block budget K, per-block width c, and loop depth R are fixed hyperparameters that must be set to cover the expected step count" | §6 Conclusion, "Limitations." ¶ | L | T5 |
| "so chains longer than K steps fall back to autoregressive completion of the tail." | §6 Conclusion, "Limitations." ¶ | L | T5, T7 |
| "Making K, c, and R adaptive is a natural and promising direction for future work." | §6 Conclusion, "Limitations." ¶ | F | T5 |
| "Running fewer iterations leaves the model less opportunity to refine, while running more than at training time brings no further gain." | §4.3.2 ablation, "Inference-time R" | L | T7 |
| "This parity holds only at 3B: at smaller scales LOTUS-aux degrades, whereas direct LM-head supervision (LOTUS) stays robust across scale." | §4.2 Main Results | L | T1 (weak fit, not tallied) — variant; here the smaller scale is worse |
| "CODI is faster than LOTUS in the thought phase ... LOTUS is nonetheless only modestly slower (133 vs. 88 ms)" | §4.2 Main Results (latency) | L | T9 |
| "teacher forcing introduces an extra train/inference mismatch." | §3.4 LOTUS-aux (design trade-off) | L | T6 (weak fit, not tallied) — variant; loose fit |

LOTUS tags (tallied): L = T2, T5, T7, T9; F = T2, T5.

---

#### 2. Penelope — "Penelope: Localized Latent Recurrence for Efficient Structured Reasoning" (Yutong Chen, Shouqian Shi, Xinran Liu, Haochen Wang, Jiaying Wang, Tianxing Xu, Yuanxi Wang, Zirui Ding)
- arXiv: 2607.25915. Title verified.
- Venue/date: arXiv preprint, v1 dated 28 Jul 2026. The comments field says "8 pages, 2 figures"; no venue is given.
- Version read: **v1**, the only version — [arXiv HTML v1](https://arxiv.org/html/2607.25915v1).
- Sections used:
  - **No Limitations section.** The main source is **§5 "Discussion and Future Work"**.
  - Scope caveats also come from §6 Conclusion, §4.1 (comparison protocol) and §3.6 (execution accounting).
  - The "supplement" that the text cites (parameter audit, schedule sensitivity) is not part of the arXiv HTML, and I did not read it.
- Setup, for context: Llama-3.2-1B and Qwen3.5-0.8B-Base backbones; Deep ListOps, ProsQA and PrOntoQA tasks.

| quote | section | L/F | tags |
|---|---|---|---|
| "The evaluation covers three structured-reasoning settings with deterministic answer metrics; detailed depth and placement analyses use compositional expression reduction." | §5 Discussion and Future Work | L | T2 |
| "validation-selected recurrence provides a small, consistent gain over the strong inference-only K=0 boundary rather than monotonic improvement with depth." | §5 Discussion and Future Work | L | T7 — non-monotonic depth scaling |
| "The system selects a fixed validation depth per run" | §5 Discussion and Future Work | L | T5 |
| "online per-input halting, larger backbones, and more heterogeneous state transitions remain future work." | §5 Discussion and Future Work | F | T5, T1, T2 (weak fit, not tallied) — T2 for "heterogeneous state transitions" is a loose reading |
| "All main comparisons use fixed validation-selected depths rather than online input-adaptive halting." | §4.1 Experimental Setup | L | T5 |
| "It does not parameter-match the methods or equalize training FLOPs, wall-clock time, or target presentations." | §4.1 Experimental Setup | L | T10, T9 |
| "Our efficiency claim concerns inference topology and measured latency, not parameter efficiency." | §4.1 Experimental Setup | L | T9 |
| "It is a serial decoder-depth proxy, not a FLOP count or hardware-independent complexity measure." | §3.6 Pre-Answer Execution Accounting | L | T9 |
| "This is efficient latent-computation allocation, not parameter efficiency or monotonic depth scaling." | §6 Conclusion | L | T9, T7 |

Penelope tags (tallied): L = T2, T5, T7, T9, T10; F = T1, T5.

---

#### 3. AGCLR — "Why Limit the Residual Stream to Layers and Not Tokens? Persistent Memory for Continuous Latent Reasoning" (Mujtaba Farhan, Maheep Chaudhary)
- arXiv: 2606.07720. I verified the title; the arXiv title has no "AGCLR:" prefix. The method is "AGCLR (Adaptive Gated Continuous Latent Reasoning)", built on Coconut.
- Venue/date: arXiv preprint, v1 dated 5 Jun 2026; no venue given.
- Version read: **v1**, the only version — [arXiv HTML v1](https://arxiv.org/html/2606.07720v1).
- Sections used: **§6 Conclusion**, which has a limitations sentence and a future-work sentence, and the dedicated appendix **App. C "Additional Limitations"**.

| quote | section | L/F | tags |
|---|---|---|---|
| "Our evaluation uses single-seed GPT-2 124M on three benchmarks" | §6 Conclusion | L | T10, T1 |
| "scalability to larger models and broader task distributions remains to be validated." | §6 Conclusion | L | T1, T2 |
| "Future work includes scaling to larger base models and multi-seed evaluation." | §6 Conclusion | F | T1, T10 |
| "Single-seed evaluation. All reported results are from single training runs." | App. C Additional Limitations | L | T10 |
| "variance estimates from multiple seeds would strengthen claims." | App. C Additional Limitations | F | T10 |
| "Scalability to larger models (1B+ parameters) remains unexplored, though the architectural modifications are parameter-agnostic." | App. C Additional Limitations | L | T1 |
| "Dataset coverage. While we evaluate on arithmetic (GSM8K), multi-hop QA (HotpotQA), and planning (ProsQA), additional benchmarks (MuSiQue, 2WikiMultihopQA, StrategyQA) would further validate generalization." | App. C Additional Limitations | L | T2, T10 |
| "additional benchmarks (MuSiQue, 2WikiMultihopQA, StrategyQA) would further validate generalization." | App. C Additional Limitations | F | T2, T10 |
| "Comparison to retrieval-augmented systems or significantly larger models (e.g., GPT-3.5, GPT-4) is beyond our scope but would provide additional context." | App. C Additional Limitations | L | T10 |
| "Comparison to retrieval-augmented systems or significantly larger models (e.g., GPT-3.5, GPT-4) ... would provide additional context." | App. C Additional Limitations | F | T10 |
| "Our concept-stream analysis follows a logit-lens-style probing; finer-grained feature attribution (Chaudhary & Geiger, 2024; Golechha et al., 2025) is left to future work." | App. C Additional Limitations | L | T4 |
| "finer-grained feature attribution ... is left to future work." | App. C Additional Limitations | F | T4 |

AGCLR tags (tallied): L = T1, T2, T4, T10; F = T1, T2, T4, T10.

---

#### 4. Pause tokens — "Think before you speak: Training Language Models With Pause Tokens" (Goyal, Ji, Rawat, Menon, Kumar, Nagarajan)
- arXiv: 2310.02226. Title verified.
- Venue/date: ICLR 2024; the arXiv comment says "Published at ICLR 2024".
  - Versions: v1 3 Oct 2023, v2 13 Mar 2024, v3 21 Apr 2024.
- Version read: **v3**, the latest and post-acceptance — [arXiv HTML v3](https://arxiv.org/html/2310.02226v3). I did not compare it against the OpenReview PDF.
- Sections used:
  - **§8 "Conclusion, Limitations and Future Work"**, a combined section whose heading names limitations.
  - §6 "Discussion and key open questions".
  - §4.3 Results (Fig. 3 caption).
  - §5 Ablations ("Robustness to a varying number of inference-time pauses").

| quote | section | L/F | tags |
|---|---|---|---|
| "we do not claim that pause-training should benefit every downstream task. Some tasks may simply be better off with zero `<pause>` tokens." | §8 Conclusion, Limitations and Future Work | L | T5 |
| "The most important limitation though is that the expense of pause-pretraining comes in the way of making this idea more widely accessible." | §8 Conclusion, Limitations and Future Work | L | T6 |
| "we do not study how the gains generalize across more model sizes (beyond 1 B and 130 M), or to encoder-decoder architectures, or to other pretraining mixtures and objectives." | §8 Conclusion, Limitations and Future Work | L | T1, T16, T12 |
| "while we have laid out some preliminary theory for why `<pause>` tokens may be beneficial, we leave a rigorous understanding for future study." | §8 Conclusion, Limitations and Future Work | L | T13 |
| "we leave a rigorous understanding for future study." | §8 Conclusion, Limitations and Future Work | F | T13 |
| "pause-training with multiple different `<pause>` tokens" | §8 Conclusion, Limitations and Future Work | F | NEW: latent-token-design |
| "better determining the number of `<pause>` tokens (perhaps using model confidence)" | §8 Conclusion, Limitations and Future Work | F | T5 |
| "inducing robustness to shifts in delays" | §8 Conclusion, Limitations and Future Work | F | T7 |
| "But the most pressing next step would be to find ways to make delays helpful directly on a standard pretrained model." | §8 Conclusion, Limitations and Future Work | F | T6 |
| "In contrast, introducing delays only in the finetuning stage provides only lukewarm gains, and even hurts in GSM8k." | §4.3 Results (Fig. 3 caption) | L | T6 |
| "for PausePT_PauseFT, providing zero delay during inference breaks its performance spectacularly" | §5 Ablations, "Robustness to a varying number of inference-time pauses" | L | T7 |
| "The design of zero-delay-robust pause-trained models is thus an important question for future work." | §5 Ablations, "Robustness to a varying number of inference-time pauses" | F | T7 |
| "An immediate hypothesis would be that for smaller models, delays become more beneficial ... But a preliminary comparison between our two model sizes surprisingly suggests the opposite." | §6 Discussion and key open questions | L | T1 — smaller scale gains less |
| "CoT has a computational depth that is larger by a multiplicative factor of M, compared to pause-inference." | §6 Discussion and key open questions | L | T3 (weak fit, not tallied) — capacity argument, not an accuracy gap |
| "one major advantange of CoT is that it does not seem to require any special modifications in the pretraining objective." | §6 Discussion and key open questions | L | T3 (weak fit, not tallied), T6 — T3 weak: a capacity/pretraining contrast with CoT, not an accuracy gap |
| "We hope this preliminary result can inspire further discussions on how to formalize the Transformer's implementation capacity and differentiate it from the raw parameter count." | §6 Discussion and key open questions | F | T13 |

Pause tags (tallied): L = T1, T5, T6, T7, T12, T13, T16; F = T5, T6, T7, T13, NEW: latent-token-design.

---

#### 5. Liang & Pan — "Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks" (Jia Liang, Liangming Pan)
- arXiv: 2602.00449. Title verified; it is the CODI study on polynomial-iteration and finite-state sequential tasks.
- Venue/date: v1 31 Jan 2026, v2 27 Sep 2026. The comment says "32 pages, 20 figures"; no venue is given.
- Version read: **v2**, the latest — [arXiv HTML v2](https://arxiv.org/html/2602.00449v2).
- Sections used:
  - **No standalone Limitations section.** There is a "Limitations & future directions." paragraph inside **§8 Conclusion**.
  - App. L (CODI vs CoT and non-CoT), App. M.4 (scope of FSC), App. N (transition points) and App. F (three-hop analysis).

| quote | section | L/F | tags |
|---|---|---|---|
| "Our study focuses on CODI-style latent CoT, where latent reasoning is learned through teacher–student distillation into a fixed number of continuous thought tokens." | §8 Conclusion, "Limitations & future directions." ¶ | L | T16 |
| "further work is needed to determine how broadly the observed mechanisms extend to other latent-reasoning architectures, such as looped transformers, recurrent latent-state models, or hybrid latent-text approaches." | §8 Conclusion, "Limitations & future directions." ¶ | F | T16 |
| "our polynomial and finite-state sequential tasks provide ground-truth intermediate states ... but they remain synthetic abstractions of reasoning." | §8 Conclusion, "Limitations & future directions." ¶ | L | T2 |
| "An important direction for future work is to test whether the compressibility principle predicts latent-reasoning behavior in more naturalistic datasets" | §8 Conclusion, "Limitations & future directions." ¶ | F | T2 |
| "FSC is not intended to capture all sequential reasoning. Its rigorous scope is deterministic finite-state, operation-indexed sequential computation." | App. M.4 Generality and Scope of FSC | L | T2 |
| "It is important to note that this step-by-step reasoning pattern does not emerge uniformly across all polynomial-task runs." | App. F Three-hop analysis | L | T10 — run-to-run variability |
| "Formalizing this capacity–difficulty tradeoff remains an important direction for future work." | App. N Transition points depend on task difficulty | F | T13 |
| "they highlight a key failure mode on sequential tasks that cannot be stably compressed." | §8 Conclusion | L | T5 — scope: field (CODI) |
| "These results expose a key limitation of CODI-style latent CoT on strictly sequential tasks: with a fixed latent-compute budget, CODI can struggle when the underlying computation is effectively incompressible." | App. L Comparing CODI with CoT and Non-CoT | L | T5 — scope: field (CODI) |
| "CoT training substantially outperforms both CODI and a standard non-CoT transformer when the modulus m is prime." | App. L Comparing CODI with CoT and Non-CoT | L | T3 — scope: field (CODI) |
| "latent-CoT models inherit a similar constraint, now jointly bounded by architectural depth and the number of latent-thought steps." | App. L Comparing CODI with CoT and Non-CoT | L | T5 — scope: field (latent CoT) |

Liang-Pan tags (tallied): L = T2, T3, T5, T10, T16; F = T2, T13, T16.
Liang-Pan own-study only: L = T2, T10, T16; F = T2, T13, T16.

---

#### 6. Chang et al. — "Unlocking the Black Box of Latent Reasoning: An Interpretability-Guided Approach to Intervention" (Shuochen Chang, Tong Bai, Xiaofeng Zhang, Qianli Ma, Qingyang Liu, Zhaohe Liao, et al.; 8 authors)
- arXiv: 2606.01243. Title verified; it is the paper on training-free interventions, including causal-hub editing of early latents, on CODI and Coconut-style models.
- Venue/date: arXiv preprint, v1 dated 31 May 2026; no venue given. It uses an ACL-style unnumbered Limitations section.
- Version read: **v1**, the only version — [arXiv HTML v1](https://arxiv.org/html/2606.01243v1).
- Sections used:
  - The **dedicated "Limitations" section**, one paragraph after §5.
  - §5 Conclusion contains no limitation statements.
- Setup, for context: Qwen3-8B, Llama-3.1-8B and Llama-3.2-3B; GSM8K, GSM-Hard, SVAMP and StrategyQA.

| quote | section | L/F | tags |
|---|---|---|---|
| "they introduce a marginal computational overhead during inference due to the additional calculation of gradients and projections." | Limitations (dedicated, unnumbered) | L | T9 |
| "Future work aims to eliminate this latency by incorporating these interpretability priors as supervisory signals, thereby developing improved training algorithms for latent reasoning." | Limitations (dedicated, unnumbered) | F | T12, T9 |

Chang tags (tallied): L = T9; F = T9, T12.

---

#### 7. Li et al. — "Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure" (Zirui Li, Xuefeng Bai, Kehai Chen, Yizhi Li, Jian Yang, Chenghua Lin, et al.; 7 authors)
- arXiv: 2602.08783. Title verified; it is the do-intervention study on Coconut and CODI latents.
- Venue/date: ICML 2026; the comment says "Accepted to ICML 2026".
  - Versions: v1 9 Feb 2026, v2 17 Mar 2026, v3 28 May 2026.
- Version read: **v3**, the latest — [arXiv HTML v3](https://arxiv.org/html/2602.08783v3).
- Sections used:
  - The **dedicated "Limitations" section** (unnumbered, after §8).
  - **§7 Discussion**, the "Implications for latent-reasoning design" paragraph, which is field-level design advice.
  - §8 Conclusion and App. E (model size).

| quote | section | L/F | tags |
|---|---|---|---|
| "Our conclusions are tied to specific methodological choices, including the step-level causal interface, hidden-state overwrite interventions, and teacher-forced readout." | Limitations (dedicated, unnumbered) | L | T10, T4 |
| "this study is limited to single-step edits, a fixed latent budget (T=6), and a controlled set of paradigms, backbones, and CoT-supervised benchmarks." | Limitations (dedicated, unnumbered) | L | T5, T16, T2, T10 |
| "evaluation across more paradigms, longer horizons, adaptive latent budgets, and varied intervention types is still needed to assess the generalizability of the observed patterns." | Limitations (dedicated, unnumbered) | F | T16, T7, T5, T10 |
| "the main influence-structure observation is not confined to GPT-2/Llama3-1B/Qwen3-4B-Instruct settings, although a systematic scaling study over matched model families remains future work." | App. E Additional Model Size Analysis | F | T1 |
| "Latent budgets should not be treated as homogeneous extra depth: computation, supervision, and regularization should be allocated selectively according to the functional role of each step." | §7 Discussion, "Implications for latent-reasoning design" | F | T5, T12 — scope: field (design advice) |
| "Since latent states lack natural supervision targets" | §7 Discussion, "Implications for latent-reasoning design" | L | T8 — scope: field |
| "future objectives could move beyond uniform CoT imitation and instead impose functional constraints on high-leverage steps or high-influence routes" | §7 Discussion, "Implications for latent-reasoning design" | F | T12 — scope: field (design advice) |
| "early states should preserve uncertainty or multiple candidate modes, while later states consolidate them into a perturbation-stable decision." | §7 Discussion, "Implications for latent-reasoning design" | F | T12 — scope: field (design advice) |
| "stopping rules should consider not only when an answer first becomes decodable, but also when the supporting latent representation becomes stable." | §7 Discussion, "Implications for latent-reasoning design" | F | T5 — scope: field (design advice) |
| "future latent-reasoning models can be improved by explicitly shaping where information is written and how it is consolidated across steps ... rather than only scaling the number of latent steps." | §8 Conclusion | F | T14, T12 — scope: field (design advice) |

Li-causal tags (tallied): L = T2, T4, T5, T8, T10, T16; F = T1, T5, T7, T10, T12, T14, T16.
Li-causal own-study only: L = T2, T4, T5, T10, T16; F = T1, T5, T7, T10, T16.

---

#### 8. Duraipandian et al. — "Interpreting Latent CoT Reasoning as Dynamical Systems" (Sabari Iyyappan Duraipandian, Shreya Sanjay Boyane, Manju Nagesh, Jerome Francis, Archana Vaidheeswaran, Kevin Zhu)
- arXiv: 2607.09698. Title verified; it is the GPT-2 CODI/Coconut dynamical-systems paper that describes CODI's "two lobe pattern".
- Venue/date: ICML 2026 FoGen Workshop; the comment says "15 pages, ICML 2026 FoGen Workshop". v1 is dated 20 Jun 2026.
- Version read: **v1**, the only version — [arXiv HTML v1](https://arxiv.org/html/2607.09698v1).
- Sections used:
  - **No Limitations section.** The scope and future-work paragraph sits in **§7 Conclusion**.
  - §6 Discussion lists findings only, with no limitations.
  - One metric caveat comes from §4.3.2.

| quote | section | L/F | tags |
|---|---|---|---|
| "Our experiments mainly focus on the GSM8k dataset and the GPT-2 Small model." | §7 Conclusion | L | T2, T1 |
| "We also focus mainly on the CODI and COCONUT latent CoT reasoning methods, and additionally use the SIM-CoT training paradigm to show how are analysis is paradigm agnostic" | §7 Conclusion | L | T16 — hedged: authors also claim paradigm-agnostic |
| "Future works should investigate on the stability types for CODI and COCONUT on long CoT chains" | §7 Conclusion | F | T7 |
| "and whether interventions in latent space can improve downstream performance." | §7 Conclusion | F | NEW: latent-intervention |
| "Other directions include extending this framework to larger reasoning models like Llama, Deepseek-distill-Llama3-8b, etc." | §7 Conclusion | F | T1 |
| "and other datasets like MATH and Strategy-QA." | §7 Conclusion | F | T2 |
| "Although decreasing displacement is consistent with convergence toward a stable region of representation space, it does not necessarily imply convergence toward a correct solution." | §4.3.2 Quantitative Analysis (metric caveat) | L | T4 |

Duraipandian tags (tallied): L = T1, T2, T4, T16; F = T1, T2, T7, NEW: latent-intervention.

---

#### 9. Dilgren & Wiegreffe — "Are Latent Reasoning Models Easily Interpretable?" (Connor Dilgren, Sarah Wiegreffe)
- arXiv: 2604.04902. Title verified; it is the paper finding latent tokens often unnecessary on ProsQA and PrOntoQA.
- Venue/date: COLM 2026; the comment says "Published as a conference paper at COLM 2026". v1 6 Apr 2026, v2 10 Aug 2026.
- Version read: **v2**, the latest and the camera-ready per its comment — [arXiv HTML v2](https://arxiv.org/html/2604.04902v2).
- Sections used:
  - **No dedicated Limitations section.** Statements come from §7 Conclusion, §4.3, §6, §6.1, and App. A.3, F, I.3, I.4 and J.3.
  - Many statements are about the limits of the paper's own interpretability method (vocabulary projection).

| quote | section | L/F | tags |
|---|---|---|---|
| "First, LRMs do not fully utilize their latent reasoning tokens. On logical reasoning datasets, LRMs often determine their final answer without latent reasoning at all." | §7 Conclusion | L | NEW: latent-underuse, T5 — scope: field (Coconut, CODI) |
| "When controlling for training regimen benefits, no-CoT models match LRM performance on logical reasoning datasets." | §7 Conclusion | L | NEW: latent-underuse — scope: field (Coconut, CODI) |
| "The underutilization of latent reasoning tokens across all three datasets may partially explain why LRMs do not consistently surpass ERM performance (Table 1)." | §4.3 Results | L | T3, NEW: latent-underuse — scope: field (Coconut, CODI) |
| "Future work could address underutilization by training models to better use their reasoning budget or by improving efficiency through introducing early stopping mechanisms" | §4.3 Results | F | T12, T5 — scope: field |
| "in the subsequent sections, we present results only on the GSM8k-Aug dataset" | §4.3 Results | L | T2 |
| "We encourage future work to investigate on which tasks latent reasoning holds a comparative advantage" | §7 Conclusion | F | T2 — scope: field |
| "Our findings indicate that LRMs are more interpretable than previously assumed, though this may not hold for other classes of LRMs that have a weaker natural language prior." | §7 Conclusion | L | T16, T4 |
| "However, this encouraging result may be an artifact of training Coconut and CODI on gold reasoning traces." | §6.1 Results | L | T8, T4 |
| "Investigating the interpretability of such models is a promising direction for future work." | §6.1 Results | F | T4, T16 |
| "we do not claim that the reasoning traces found via the forward chaining method are causally used by the model." | §6.1 Results | L | T4 |
| "we hope that the reasoning traces decoded by our forward chaining method ... can be operationalized in future work to make improvements to model behavior and monitoring." | §6.1 Results | F | T4 |
| "We analyze a 460-instance subset of GSM8k-Aug's test set, filtered for unique, single-token numbers ... single-token numbers are a limitation of vocabulary projection." | §6 (Assembling verified reasoning steps) | L | T4, T10 |
| "Vocabulary projection, used in Section 5 and Section 6, only reveals single-token concepts; it omits multi-token concepts and latent space directions not well-aligned with vocabulary space." | App. F Vocabulary projection details | L | T4 |
| "We encourage future work to develop core mechanistic interpretability tools that can address these limitations, which would make LRMs more interpretable." | App. F Vocabulary projection details | F | T4 |
| "our backtracking search results provide a lower bound on LRM interpretability, with failures stemming from methodological limitations rather than fundamentally uninterpretable model behavior." | App. I.3 Backtracking error analysis | L | T4 |
| "This degradation reflects a limitation in the models' capacity to maintain longer reasoning traces." | App. I.4 Results by solution length | L | T14 (weak fit, not tallied) — scope: field (Coconut, CODI; loose fit, alt. T7) |
| "Requirement 6 makes this method difficult to use for tasks that have a small set of possible operands or results" | App. J.3 Dataset requirements | L | T4, T2 |
| "Requirement 3 can be removed if future work finds a way to detect the operator used from the model's representations directly." | App. J.3 Dataset requirements | F | T4 |
| "future work could use a method other than vocabulary projection for detecting concepts encoded in a latent token" | App. J.3 Dataset requirements | F | T4 |
| "The extent to which the conclusion of training from scratch holds beyond 2-layer models is unclear." | App. A.3 Comparison with Prior Work | L | T1 — scope: field (about Zhu et al. 2025a / Rizvi-Martel et al. 2026) |

Dilgren tags (tallied): L = T1, T2, T3, T4, T5, T8, T10, T16, NEW: latent-underuse; F = T2, T4, T5, T12, T16.
Dilgren own-study only: L = T2, T4, T8, T10, T16; F = T4, T16.

---

#### Tally for this group, all tallied statements: own-study rows plus field rows (9 papers; each paper counted at most once per tag per column)

| tag | papers stating it as L | papers stating it as F | papers stating it at all | paper short names |
|---|---|---|---|---|
| T1 model-scale | 4 | 4 | 6 | L: AGCLR, Pause, Duraipandian, Dilgren. F: Penelope, AGCLR, Li-causal, Duraipandian (weak fit, not counted: LOTUS) |
| T2 task-breadth | 7 | 5 | 7 | L: LOTUS, Penelope, AGCLR, Liang-Pan, Li-causal, Duraipandian, Dilgren. F: LOTUS, AGCLR, Liang-Pan, Duraipandian, Dilgren |
| T3 gap-to-explicit-CoT | 2 | 0 | 2 | L: Liang-Pan, Dilgren (weak fit, not counted: Pause) |
| T4 interpretability | 4 | 2 | 4 | L: AGCLR, Li-causal, Duraipandian, Dilgren. F: AGCLR, Dilgren |
| T5 fixed-vs-adaptive-compute | 6 | 5 | 6 | L: LOTUS, Penelope, Pause, Liang-Pan, Li-causal, Dilgren. F: LOTUS, Penelope, Pause, Li-causal, Dilgren |
| T6 training-cost-or-stability | 1 | 1 | 1 | L: Pause. F: Pause (weak fit, not counted: LOTUS) |
| T7 extrapolation-past-trained-depth | 3 | 3 | 5 | L: LOTUS, Penelope, Pause. F: Pause, Li-causal, Duraipandian |
| T8 supervision-dependence | 2 | 0 | 2 | L: Li-causal, Dilgren |
| T9 efficiency-claims | 3 | 1 | 3 | L: LOTUS, Penelope, Chang. F: Chang |
| T10 evaluation-rigor | 5 | 2 | 5 | L: Penelope, AGCLR, Liang-Pan, Li-causal, Dilgren. F: AGCLR, Li-causal |
| T11 test-time-scaling-and-verification | 0 | 0 | 0 | none |
| T12 RL-or-new-objectives | 1 | 3 | 4 | L: Pause. F: Chang, Li-causal, Dilgren |
| T13 theory-gap | 1 | 2 | 2 | L: Pause. F: Pause, Liang-Pan |
| T14 memory-or-attention-over-steps | 0 | 1 | 1 | F: Li-causal (weak fit, not counted: Dilgren) |
| T15 hybrid-or-tool-integration | 0 | 0 | 0 | none |
| T16 architecture-generality | 5 | 3 | 5 | L: Pause, Liang-Pan, Li-causal, Duraipandian, Dilgren. F: Liang-Pan, Li-causal, Dilgren |
| NEW: latent-token-design | 0 | 1 | 1 | F: Pause |
| NEW: latent-intervention | 0 | 1 | 1 | F: Duraipandian |
| NEW: latent-underuse | 1 | 0 | 1 | L: Dilgren |

Ranked by "papers stating it at all":
1. T2 (7)
2. T1, T5 (6 each)
3. T7, T10, T16 (5 each)
4. T4, T12 (4 each)
5. T9 (3)
6. T3, T8, T13 (2 each)
7. T6, T14, NEW-latent-token-design, NEW-latent-intervention, NEW-latent-underuse (1 each)
8. T11, T15 (0 each)

#### Tally for this group, own-study rows only (field rows excluded; for merging with groups that counted only self-limitations)

| tag | papers stating it as L | papers stating it as F | papers stating it at all | paper short names |
|---|---|---|---|---|
| T1 model-scale | 3 | 4 | 5 | L: AGCLR, Pause, Duraipandian. F: Penelope, AGCLR, Li-causal, Duraipandian (weak fit, not counted: LOTUS) |
| T2 task-breadth | 7 | 4 | 7 | L: LOTUS, Penelope, AGCLR, Liang-Pan, Li-causal, Duraipandian, Dilgren. F: LOTUS, AGCLR, Liang-Pan, Duraipandian |
| T3 gap-to-explicit-CoT | 0 | 0 | 0 | none (weak fit, not counted: Pause) |
| T4 interpretability | 4 | 2 | 4 | L: AGCLR, Li-causal, Duraipandian, Dilgren. F: AGCLR, Dilgren |
| T5 fixed-vs-adaptive-compute | 4 | 4 | 4 | L: LOTUS, Penelope, Pause, Li-causal. F: LOTUS, Penelope, Pause, Li-causal |
| T6 training-cost-or-stability | 1 | 1 | 1 | L: Pause. F: Pause (weak fit, not counted: LOTUS) |
| T7 extrapolation-past-trained-depth | 3 | 3 | 5 | L: LOTUS, Penelope, Pause. F: Pause, Li-causal, Duraipandian |
| T8 supervision-dependence | 1 | 0 | 1 | L: Dilgren |
| T9 efficiency-claims | 3 | 1 | 3 | L: LOTUS, Penelope, Chang. F: Chang |
| T10 evaluation-rigor | 5 | 2 | 5 | L: Penelope, AGCLR, Liang-Pan, Li-causal, Dilgren. F: AGCLR, Li-causal |
| T11 test-time-scaling-and-verification | 0 | 0 | 0 | none |
| T12 RL-or-new-objectives | 1 | 1 | 2 | L: Pause. F: Chang |
| T13 theory-gap | 1 | 2 | 2 | L: Pause. F: Pause, Liang-Pan |
| T14 memory-or-attention-over-steps | 0 | 0 | 0 | none (weak fit, not counted: Dilgren) |
| T15 hybrid-or-tool-integration | 0 | 0 | 0 | none |
| T16 architecture-generality | 5 | 3 | 5 | L: Pause, Liang-Pan, Li-causal, Duraipandian, Dilgren. F: Liang-Pan, Li-causal, Dilgren |
| NEW: latent-token-design | 0 | 1 | 1 | F: Pause |
| NEW: latent-intervention | 0 | 1 | 1 | F: Duraipandian |
| NEW: latent-underuse | 0 | 0 | 0 | none |

Ranked by "papers stating it at all":
1. T2 (7)
2. T1, T7, T10, T16 (5 each)
3. T4, T5 (4 each)
4. T9 (3)
5. T12, T13 (2 each)
6. T6, T8, NEW-latent-token-design, NEW-latent-intervention (1 each)
7. T3, T11, T14, T15, NEW-latent-underuse (0 each)

**Sensitivity notes for the merger:**
- **Field rows** come from Liang-Pan, Li-causal and Dilgren only. The table below gives the full count with field rows, the own-only count without them, and the papers that drive the difference.

  | tag | full count | own-only count | papers that drop out |
  |---|---|---|---|
  | T1 | 6 | 5 | Dilgren (its only T1 row is about other papers' from-scratch models) |
  | T3 | 2 | 0 | Liang-Pan, Dilgren |
  | T5 | 6 | 4 | Liang-Pan, Dilgren |
  | T8 | 2 | 1 | Li-causal |
  | T12 | 4 | 2 | Li-causal, Dilgren |
  | T14 | 1 | 0 | Li-causal |
  | NEW: latent-underuse | 1 | 0 | Dilgren |

- **Including the weak fits** would add:
  - LOTUS to T1, from the LOTUS-aux scale sensitivity, where the smaller scale is worse.
  - LOTUS to T6, from the teacher-forcing train/inference mismatch.
  - Pause to T3, from the computational-depth contrast with CoT.
  - Dilgren to T14, from "capacity to maintain longer reasoning traces"; T7 would be an alternative tag.
  - Penelope already has T2 from another row, so its weak T2 changes nothing.
- **T7 here is broader than "failure past trained depth":**
  - Two papers report measured results for running more steps than trained:
    - LOTUS: running more loops gives "no further gain" (69.3% at R=7 vs 70.0% at the trained R=6).
    - Pause: performance "degrades gracefully" for M_inf ∈ [5,25] with M_ft=10 (§5, Fig. 4).
  - Pause also reports running fewer delays than trained: zero delay "breaks its performance spectacularly".
  - Penelope's T7 is non-monotonic depth scaling.
  - Li-causal ("longer horizons") and Duraipandian ("long CoT chains") state T7 only as future work.
- **Folding NEW: latent-intervention into T11** would give T11 one paper (Duraipandian F).

**Section types:**
- **Dedicated Limitations section:** Chang, Li-causal (2 of 9).
- **Combined section with "Limitations" in its heading:** Pause (§8 "Conclusion, Limitations and Future Work").
- **Labeled "Limitations" paragraph inside the Conclusion:** LOTUS, Liang-Pan.
- **Limitations appendix plus sentences in the Conclusion:** AGCLR (App. C "Additional Limitations").
- **"Discussion and Future Work" section with no limitations heading:** Penelope.
- **Nothing labeled as limitations:** Duraipandian (scope sentences in the Conclusion) and Dilgren (statements scattered across the Results, Conclusion and appendices).

### Inferences
This section is **Observations, not counted**: my own observations, not author statements, and not in any tally.
- **As far as I found, none of the four CODI analysis papers runs CODI at inference with more latent steps than it was trained with.** The four are Liang-Pan, Li-causal, Duraipandian and Dilgren.
  - Duraipandian says "Each model generates a fixed sequence of T=6 latent reasoning steps per input" ([§4.2](https://arxiv.org/html/2607.09698v1)). Its claim that CODI is a "stable attractor" whose representations "converge as timestep approaches infinity" (§6) is inferred from metrics over those 6 steps.
  - Li-causal and Dilgren vary the step count downward only (early-stop decoding).
  - Liang-Pan's §7 ablation trains separate models with p ∈ {1,2,3,6,9,12,20} latent steps. It finds "increasing p does not reliably induce a longer internal rollout" ([Liang-Pan §7, App. K](https://arxiv.org/html/2602.00449v2)). That is a training-time budget sweep, framed as robustness, not a stated limitation.
  - So the depth-drift result in this project (CODI losing answers past 6 steps) is not something these authors report or acknowledge. Li-causal's "longer horizons" and Duraipandian's "long CoT chains" are the nearest stated directions.
- **Two papers explicitly flag the fixed 6-step latent budget** as a limit on their conclusions: Li-causal ("a fixed latent budget (T=6)") and Liang-Pan ("a fixed number of continuous thought tokens"). Liang-Pan also says CODI is "jointly bounded by architectural depth and the number of latent-thought steps" (App. L).
- **Excluded critiques of prior work in Introductions and Related Work:**
  - **Chang §1** says "its interpretability remains largely unverified" and "existing studies are typically limited to smaller-scale models" (field-level T4 and T1) ([Chang §1](https://arxiv.org/html/2606.01243v1)).
  - **AGCLR (Abstract and §5.2)** says "we identify a limitation we term the concept bottleneck. At each reasoning pass, intermediate hidden states are overwritten". This is a T14-type limitation of Coconut, and it motivates the method ([AGCLR](https://arxiv.org/html/2606.07720v1)).
  - **Duraipandian §2.1 (Related Work)** says COCONUT and CODI "suffer from the latent instability problem … as the number of implicit reasoning tokens scale up" (field T6 and T7, citing SIM-CoT).
  - **Duraipandian §1** says latent-CoT faithfulness "remains underexplored and lacks rigorous verification" (field T4) ([Duraipandian](https://arxiv.org/html/2607.09698v1)).
- **Unflagged facts:** the papers report these but never frame them as limitations.
  - Liang-Pan trains small GPT-2-style models, 3-layer and 2-head by default and at most 7-layer and 8-head (§7, App. L, App. O), but never states model scale as a limitation, so no T1 is recorded for it.
  - Duraipandian says "All experiments use a fixed random seed of 42" (§4.2) but never states this as a limitation, so no T10.
  - Chang's gains are small, for example "+1.3%" on GSM-Hard and "up to +1.5%" on StrategyQA (§4.5). They are not framed as a limitation.
  - LOTUS stays "within ∼1.5 points of explicit CoT in-domain at each scale" (§4.2). It reports this as success, so no T3.
- **Dilgren's App. H quantifies where latents matter.** Replacing latents with mean latents changes accuracy by only "-0.1 to +0.2 percentage points" on PrOntoQA and ProsQA. On GSM8k-Aug, accuracy "dropped 5.8 to 31.5 percentage points".
  - In App. G, their Coconut + Llama-3.2-1B result is "9.6 percentage points worse" than the published one. They attribute this to hyperparameters.
- **The two hybrid or recurrence method papers (LOTUS, Penelope) both state T5 (fixed loop or depth budget, adaptive halting needed).** This matches the T5 statements in the analysis papers, so adaptive budgets are the most consistent cross-paper theme after task breadth and scale.
- **Pause tokens (2023) predates CODI and Coconut**, and its limitations differ in kind.
  - It is the only paper in this group whose main stated limitation is training cost (pause-pretraining).
  - Its finding that pause tokens added only at finetuning give "lukewarm gains" is the closest analogue to latent methods bolted onto pretrained models.
  - Its tasks span QA and reasoning, so it does not state T2.

### Gaps
This section covers **full-text access, plus unresolved items**.
- **Full-text access:** I accessed the full text of all nine papers through the arXiv HTML pages, at the versions listed above. No paper was unavailable, and no substitute source was needed.
- **Not read:**
  - Penelope's "supplement" (parameter audit, schedule sensitivity, hardware details) is cited in its text but not included in the arXiv HTML. Any limitations stated only there are missing.
  - For Pause tokens I did not compare arXiv v3 against the OpenReview camera-ready PDF.
- **Venues not stated on arXiv** for LOTUS, Penelope, AGCLR, Liang & Pan, and Chang et al. I did not search for acceptances elsewhere.
- **Tag judgement calls** a merger may want to revisit:
  - Penelope's "more heterogeneous state transitions": the meaning is unclear, so it is weak T2.
  - Pause's "we do not claim that pause-training should benefit every downstream task": tagged T5, because the text ties it to each task's "optimal number of `<pause>` tokens", not T2.
  - Li-causal's "CoT-supervised benchmarks": tagged T2.
