# Self-stated limitations and future work: looped / recurrent-depth LMs at scale (9 papers)

Group: looped and recurrent-depth language models trained or retrofitted at scale. Read 2026-11-06.

## Takeaway

All nine arXiv IDs check out against their titles, and the full text of all nine was read. Within this group, the limitations stated most often are: scale untested or worse (T1, 8/9 papers), looping past the trained depth or adding loops degrades results (T7, 7/9), efficiency or runtime caveats (T9, 6/9), weak evaluation rigor (T10, 6/9), and backbone or architecture generality (T16, 6/9 counting L or F). Two tags appear in no paper in this group: supervision dependence (T8) and test-time verification (T11). One theme the taxonomy has no tag for came up in 4 papers: looping adds compute but not parametric knowledge storage (tagged NEW: knowledge-capacity). Two papers state almost nothing. InfiLoop has no limitations, future-work or impact text at all. Training-Free Looped Transformers has no limitations section; its caveats come only from a failed-configurations appendix.

## Coding rules and NEW tags

- I counted only statements the authors make about their own method, study, or the paradigm they work in. Mechanisms a paper proposes as fixes for earlier work are not counted. Neither are generic hopes that name no direction.
- A statement that is both a limitation and a future direction has two rows: one L, one F.
- Two stretched uses of existing tags, flagged so the merge can stay consistent:
  - T7 also covers "more loops give diminishing or negative returns", which is overthinking even when the loops were trained.
  - T16 also covers "other architectural variants untested" (linear attention, multiple recurrences, "more complex recurrent mechanisms").
  - T3 is used once (Huginn) for "a FLOP-matched comparison to verbal CoT is still missing", not for "below CoT accuracy".
- NEW tags (defined here so other groups can reuse them):
  - **NEW: knowledge-capacity**: looping or recurrence adds compute but not parametric storage. Knowledge- or memorization-heavy tasks gain little, so memory banks, MoE or retrieval are proposed alongside it. (Not T14: this is about knowledge storage, not memory across loop steps.)
  - **NEW: loop-placement**: which layers to loop or keep when retrofitting a pretrained model (layer choice, window position or width), and the fragility of that choice.
  - **NEW: broader-impact-risk**: misuse, bias, hallucination or energy risks stated in an impact statement. Oversight and opacity risks stay under T4.
- Papers marked "borderline" have very few explicit caveats. I included their one or two caveat-like sentences and flagged them.

---

## 1. Huginn — Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach

- arXiv: 2502.05171 (title verified). Authors: Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein. — [arXiv abs](https://arxiv.org/abs/2502.05171)
- Venue: NeurIPS 2025 main conference. — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html)
- Version read: NeurIPS 2025 camera-ready PDF (primary) — [camera-ready PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Paper-Conference.pdf). Cross-checked against arXiv v2 (17 Feb 2025) — [arXiv HTML v2](https://arxiv.org/html/2502.05171v2).
- Sections used: Sec. 8 "Limitations and Conclusions" (a combined section; the camera-ready has no standalone Limitations section), App. B "Future Work", App. A "Broader Impact", Sec. 4.1–4.2, App. C.3, App. D.3. In arXiv v2 the same future-work text is Sec. 9, and the conclusion is Sec. 10 "Conclusions".

| quote | section | L/F | tags |
|---|---|---|---|
| "the models described are ultimately still a proof-of-concept." | Sec. 8 Limitations and Conclusions | L | T1 |
| "future work with additional compute is still required to allow for precise comparisons to the other forms of scaling" | Sec. 8 | L | T10 |
| (same) "future work with additional compute is still required to allow for precise comparisons to the other forms of scaling" | Sec. 8 | F | T10 |
| "or training verbal CoT models targeting the same FLOP count at test time." | Sec. 8 | L | T3 |
| (same) "or training verbal CoT models targeting the same FLOP count at test time." | Sec. 8 | F | T3 |
| "the model is trained on only 800B tokens, while large in comparison to older fully open-source models ... tiny in comparison to the datasets used to train industrial open-weight models." | Sec. 4.1 | L | T1 |
| "lags behind the later OLMo models trained larger, more carefully curated datasets." | Sec. 4.1 | L | T1 |
| "considering the limitations of the training run, we find these results promising" | Sec. 4.1 | L | T6 |
| "it does not outperform dedicated code models, such as StarCoder2 ..., trained for several trillion tokens." | Sec. 4.2 | L | T1 |
| "At small scales all normalization strategies worked, and we observed only tiny differences between initializations. The same was not true at scale." | App. C.3 | L | T6 |
| "the recurrent models has less capacity to memorize facts but more capacity to reason about its context." | App. D.3 | L | NEW: knowledge-capacity |
| "there is concern that this comes with costs in model oversight in comparison to verbalized chains of thought, that are currently still human-readable." | App. A Broader Impact | L | T4 |
| "Aside from work extending and analyzing the scaling behaviors of recurrent depth models, there are many questions that remain unanswered." | App. B Future Work | F | T1 |
| "novel post-training schemes ... such as fine-tuning to compress the recurrence or reinforcement learning with data with different hardness levels" | App. B | F | T12 |
| "or to internalize reasoning from CoT data into the recurrence" | App. B | F | T12 |
| "Another aspect not covered in this work is the relationship to other modern architecture improvements." | App. B | L | T16 |
| (same) "Another aspect not covered in this work is the relationship to other modern architecture improvements." | App. B | F | T16 |
| "For simplicity, we also focus on a single recurrence, where prior work has considered multiple successive recurrent stages" | App. B | L | T16 |
| "Their complementarity supports the hypothesis that a future architecture would contain both modifications." (recurrence + MoE; "also hold for other memory mechanisms") | App. B | F | T16, NEW: knowledge-capacity |

## 2. Ouro — Scaling Latent Reasoning via Looped Language Models

- arXiv: 2510.25741 (title verified). Authors: Zhu, Wang, Hua, Zhang, ... Bengio, Eshraghian (33 authors; ByteDance Seed and others). — [arXiv abs](https://arxiv.org/abs/2510.25741)
- Venue: none stated in the arXiv metadata.
- Version read: arXiv v6 (11 Oct 2026), from the PDF ([PDF v6](https://arxiv.org/pdf/2510.25741v6)). The HTML for v6 is a stub, so section numbers come from PDF text extraction and may be off by a subsection.
- Sections used: no Limitations or Future Work section. Used Sec. 8 Conclusion, Sec. 7.3 "More Discussion", Sec. 4.3 (training stability), Sec. 4.5 (RL attempts), Sec. 5.3 (depth and extrapolation), Sec. 5.4.2 (KV cache), Sec. 6.2–6.3, App. D, and a footnote on Fig. 12.

| quote | section | L/F | tags |
|---|---|---|---|
| "Our initial experiments with 8 recurrent steps in Stage 1a (Stable Training I) led to loss spikes and gradient oscillations." | Sec. 4.3 | L | T6 |
| "recurrent architectures require smaller learning rates than parameter-matched Transformers. Given compute constraints, we did not run exhaustive LR sweeps" | Sec. 4.3 | L | T6, T10 |
| "These attempts did not yield significant performance gains over the final SFT checkpoint. The primary issue stemmed from the model's dynamic early-exit mechanism." | Sec. 4.5 RL Attempts | L | T12 |
| "vLLM/SGLang provide fast rollouts via a fixed execution path, which breaks under LoopLM's variable-depth computation." | Sec. 4.5 | L | T9 |
| "these smaller models may have limited headroom for RL gains." | Sec. 4.5 | L | T1, T12 |
| "The mechanism behind this generalization remains unclear." | Sec. 4.5 | L | T13 |
| "We will further explore RL alignment for this architecture as we continue to develop infrastructure that can fully support LoopLM's dynamic computation." | Sec. 4.5 | F | T12, T9 |
| "benchmark performance sees a moderate degradation when extrapolating, with a noticeable drop compared to the peak at T = 4." | Sec. 5.3 (base models) | L | T7 |
| "performance degrades as they extrapolate to deeper, unseen recurrent steps (T = 6 − 8), reinforcing that performance is optimized for the depth seen during training." | Sec. 5.3 (SFT models) | L | T7 |
| "naively, each recurrent step requires maintaining its own KV cache, leading to 4× memory overhead for our 4-step model." | Sec. 5.4.2 | L | T9, T14 |
| "Attempting to reuse KV caches during prefilling leads to performance degradation (>10 points on GSM8K)." | Sec. 5.4.2 | L | T9, T14 |
| "adding more recurrent steps significantly improves the performance on more reasoning-heavy categories, while the improvements on more knowledge-heavy tasks is limited." | Sec. 6.2 | L | NEW: knowledge-capacity |
| "Extending this construction to the causal attention and standard normalization used in our language models remains outside the theorem. ... does not by itself explain the empirical performance gap." | Sec. 6.3 | L | T13 |
| "Neither objective guarantees that next-token prediction loss decreases at every additional loop; the benefit of deeper computation must be evaluated for the chosen task and depth." | Sec. 7.3 | L | T7, T5 |
| "Future research should focus on enhancing performance extrapolation at greater depths" | Sec. 8 Conclusion | F | T7 |
| "and exploring more complex recurrent mechanisms" | Sec. 8 Conclusion | F | T16 |
| "when the model size is insufficient, the shallow step-wise loss increases with the growing amount of training data." / "a larger model size may be more effective for LoopLM." | App. D (scaling-law experiments) | L | T1 |
| "We conjecture that it could be because of the randomness, or insufficient hyperparameter tuning. We believe further follow-up experiments should be necessary to further validate this conclusion" | Appendix footnote 5 (Fig. 12) | L | T10 |
| (same) "We believe further follow-up experiments should be necessary to further validate this conclusion" | Appendix footnote 5 | F | T10 |

## 3. Retrofitted recurrence — Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence

- arXiv: 2511.07384. Title verified as "Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence". Authors: McLeish, Li, Kirchenbauer, Kalra, Bartoldson, Kailkhura, Schwarzschild, Geiping, Goldblum, Goldstein. — [arXiv abs](https://arxiv.org/abs/2511.07384)
- Venue: listed on the NeurIPS 2025 virtual site; the page references the Workshop on Efficient Reasoning. — [neurips.cc listing](https://neurips.cc/virtual/2025/126665)
- Version read: arXiv v1 (10 Nov 2025), the only version. — [arXiv HTML v1](https://arxiv.org/html/2511.07384v1)
- Sections used: no Limitations section. Used Sec. 5 "Discussion" (states future work explicitly), Sec. 4.3.2 and Sec. 4.4.

| quote | section | L/F | tags |
|---|---|---|---|
| "investing more FLOPs per parameter makes the training process for depth-recurrent models more expensive." | Sec. 5 Discussion | L | T6 |
| "One unsolved problem is how to most effectively build depth-recurrent models that can recur deeper at test time to solve harder problems than were seen during training." | Sec. 5 | L | T7 |
| (same) "One unsolved problem is how to most effectively build depth-recurrent models that can recur deeper at test time ..." | Sec. 5 | F | T7 |
| "how to imbue recurrent models with native adaptivity that automatically assigns the right amount of compute (recurrence) to a given problem" | Sec. 5 | F | T5 |
| "future work could identify a more optimal method for layer choice during model surgery." | Sec. 5 | F | NEW: loop-placement |
| "While our experiments are at the 1B parameter and 50B token scales, more experimentation is required to verify that our method generalizes to much larger model and data scales." | Sec. 5 | L | T1 |
| (same) "more experimentation is required to verify that our method generalizes to much larger model and data scales." | Sec. 5 | F | T1 |
| "we primarily focus on strengthening a model's mathematical capabilities via depth-recurrence, and future work should extend this to other reasoning-intensive domains." | Sec. 5 | L | T2 |
| (same) "future work should extend this to other reasoning-intensive domains." | Sec. 5 | F | T2 |
| "training strictly on math data slightly degrades performance on non-reasoning based evaluations such as Hellaswag, Arc and OpenbookQA" (they address this with a data mix) | Sec. 4.4 | L | T2 |
| "larger training recurrences lead to a larger performance improvement on a per FLOP basis for OLMo and Llama, however this trend does not hold for TinyLlama" | Sec. 4.3.2 | L | T16 |

## 4. Dreamer — Depth-Recurrent Attention Mixture (Dreamer): Giving Latent Reasoning the Attention it Deserves

- arXiv: 2601.21582. Exact title: "Depth-Recurrent Attention Mixture (Dreamer): Giving Latent Reasoning the Attention it Deserves". Authors: Knupp, Metzen, Bohn, Groh, + 1. — [arXiv abs](https://arxiv.org/abs/2601.21582)
- Venue: NeurIPS 2026, per the arXiv comments.
- Version read: arXiv v2 (5 Nov 2026). — [arXiv HTML v2](https://arxiv.org/html/2601.21582v2)
- Sections used: Sec. 5 "Discussion and Future Work", which opens with an explicit "Limitations:" paragraph. Also Sec. 6 Conclusion, App. A.1 Broader Impact Statement, and the Sec. 1 footnote.

| quote | section | L/F | tags |
|---|---|---|---|
| "Experiments are limited to fixed depth; dynamic depth is deliberately out-of-scope for reasons stated in Section 1." | Sec. 5 (Limitations paragraph) | L | T5 |
| "The tested scale is moderate; results at larger scales and rigorous scaling laws remain to be investigated." | Sec. 5 (Limitations) | L | T1 |
| (same) "results at larger scales and rigorous scaling laws remain to be investigated." | Sec. 5 (Limitations) | F | T1 |
| "Our work focuses on reasoning; the broader-domain evaluation in Section 4.2 is limited in scope." | Sec. 5 (Limitations) | L | T2 |
| "The depth attention pattern analysis (Section 4.3) is descriptive; establishing causal links between attention patterns and model behavior requires further investigation." | Sec. 5 (Limitations) | L | T4 |
| (same) "establishing causal links between attention patterns and model behavior requires further investigation." | Sec. 5 (Limitations) | F | T4 |
| "its memory footprint remains constant wrt. sequence length (Figure 3), though not wrt. depth." | Sec. 5 (Depth Attention) | L | T14, T9 |
| "the compute overhead of DA is small, though not entirely negligible." | Sec. 5 (Depth Attention) | L | T9 |
| "depth attention may continue being beneficial at larger depths, but reliable scaling laws are still to be established." | Sec. 5 (Depth Attention) | L | T1 |
| "There remains a memory movement overhead, slowing down inference by ∼13% with an unoptimized DA implementation." | Sec. 5 (Depth Attention) | L | T9 |
| "We expect this to improve by future kernel optimizations and explorations of DA variations, e.g. sliding windows, dilation, linear attention variants, and modern RNNs/SSMs along the depth dimension." | Sec. 5 (Depth Attention) | F | T9, T14 |
| "Nonetheless, dynamic depth remains a natural extension for future work." | Sec. 5 (Dynamic Depth) | F | T5 |
| "one may explore dynamic RoPE scaling in this context. Alternatives to RoPE may also be worth considering, possibly improving depth generalization." | Sec. 5 (Dynamic Depth) | F | T7 |
| "We believe that reliably generalizing DR beyond the trained depth regime is a critical and still underexplored problem." | Sec. 5 (Dynamic Depth) | L | T7 |
| (same) | Sec. 5 (Dynamic Depth) | F | T7 |
| "additional knowledge (learned latents or text embeddings) may be plugged in as separate attention dimension, resembling “in-depth” latent RAG." | Sec. 5 (Attention Mixture) | F | T15, NEW: knowledge-capacity |
| "it remains an open question whether such approaches can be beneficial in practice, given sparsity-averse GPUs/TPUs/etc." | Sec. 5 (Attention Mixture) | L | T9 |
| "at the cost of ∼13% inference slowdown in an unoptimized implementation." | Sec. 6 Conclusion | L | T9 |
| "treating depth as first-class sequential dimension, going beyond the prevalent static layering and residual connections, exploring alternatives to vanilla depth attention, and adding other attention dimensions." | Sec. 6 Conclusion | F | T14, T16 |
| "it may become harder for humans to interpret the exact path the model took to reach a conclusion, potentially masking underlying flaws in model behavior" | App. A.1 Broader Impact | L | T4 |
| "it is not immune to hallucinations and other errors. If the model scales its reasoning on top of incorrect premises ..., it may produce more convincing but ultimately flawed arguments." | App. A.1 Broader Impact | L | NEW: broader-impact-risk |
| "We do not mechanistically establish which internal computations constitute a particular form of reasoning." | Sec. 1 footnote | L | T4 |

## 5. InfiLoop — Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals

- arXiv: 2610.11570 (title verified). Authors: Li, Muhtar, He, Su, + 2. — [arXiv abs](https://arxiv.org/abs/2610.11570)
- Venue: none stated in the arXiv metadata. The HTML build prints a warning that the "page layout violates the ICML style", which suggests an ICML-format submission.
- Version read: arXiv v2 (16 Oct 2026), HTML ([arXiv HTML v2](https://arxiv.org/html/2610.11570v2)). I keyword-checked the PDF v2 ([PDF](https://arxiv.org/pdf/2610.11570v2)) for "limitation", "future", "impact statement" and "broader".
- Sections used: **no Limitations, Future Work, Discussion or Impact section.** Sec. 5 Conclusion only restates contributions, with no limitation and no future direction. The only self-stated caveat found is in App. A. Borderline paper.

| quote | section | L/F | tags |
|---|---|---|---|
| "These counts do not compare measured runtime." (the FLOP and layer-evaluation accounting versus FPRM) | App. A Compute and Persistent-State Accounting | L | T9 (borderline) |

No future directions are stated.

## 6. RecurTrace — Adaptive Latent Reasoning with Loop-Time Memory

- arXiv: 2609.03379 (title verified). Authors: Wang, Feng, Shen, Xu, + 2. — [arXiv abs](https://arxiv.org/abs/2609.03379)
- Venue: none stated in the arXiv metadata.
- Version read: arXiv v3 (19 Oct 2026). — [arXiv HTML v3](https://arxiv.org/html/2609.03379v3)
- Sections used: Sec. 5.3 "Limitations" (dedicated), Sec. 6 Conclusion, Sec. 4.2, App. H "Additional Ablations and Discussion".

| quote | section | L/F | tags |
|---|---|---|---|
| "halting is sequence-level and assigns one depth to an entire example rather than to individual tokens." | Sec. 5.3 Limitations | L | T5 |
| "Loop Memory Attention reads only same-position states from a fixed three-loop window, which keeps the cost bounded but excludes older states and cross-position memory." | Sec. 5.3 Limitations | L | T14 |
| "our experiments use frozen Qwen3 decoder-only models and direct-answer targets, leaving other architectures and long-form generation untested." | Sec. 5.3 Limitations | L | T16, T2 |
| "latent looping and chain-of-thought use different forms of compute and are complementary rather than direct substitutes, so combining them is a natural direction for future work." | Sec. 5.3 Limitations | L | T15 |
| (same) | Sec. 5.3 Limitations | F | T15 |
| "suggesting a natural extension to chain-of-thought decoding." | Sec. 6 Conclusion | F | T15 |
| "Without the floor the head itself mostly stops at one loop (Appendix G), so the floor guarantees the depth while the head allocates it." | Sec. 4.2 | L | T5 |
| "we treat this row as an initial scale check rather than a fully matched training-length replication." (4B) | App. H | L | T10, T1 |
| "we report this scaling row at a fixed two-loop depth and leave a full adaptive-halting evaluation at 4 B to future work." | App. H | L | T5, T1 |
| (same) "leave a full adaptive-halting evaluation at 4 B to future work." | App. H | F | T5, T1 |
| "plain looping extrapolates a little better than the memory variant at deep unrolls, which trails only slightly rather than collapsing." (pointer chasing) | App. H | L | T7, T14 |

## 7. FLT — Simply Stabilizing the Loop via Fully Looped Transformer

- arXiv: 2605.18797 (title verified). Authors: Fu, Yang, Zhang, Ma, + 3. — [arXiv abs](https://arxiv.org/abs/2605.18797)
- Venue: none stated in the arXiv metadata.
- Version read: arXiv v2 (25 May 2026). — [arXiv HTML v2](https://arxiv.org/html/2605.18797v2)
- Sections used: Sec. 6 "Limitation" (dedicated), Sec. 7 Conclusion, Sec. 5.4, App. B Broader Impacts.

| quote | section | L/F | tags |
|---|---|---|---|
| "When the loop count at inference exceeds the maximum used during pretraining, performance begins to fluctuate, indicating that the model struggles to adapt to excessive computational resources." | Sec. 5.4 | L | T7 |
| "explicit training over variable loop counts may be necessary for robust test-time compute adaptation." | Sec. 5.4 | F | T5, T7 |
| "Our study is limited to the model scales and architectures evaluated in this work" | Sec. 6 Limitation | L | T1, T16 |
| "whether Fully Looped Transformer remains effective for deeper backbones, larger models, and more diverse settings requires further validation." | Sec. 6 Limitation | F | T1, T16 |
| "we use a single run per configuration and do not report training-run error bars." | Sec. 6 Limitation | L | T10 |
| "Future work should examine residual-state supervision, interpretability, and robustness across random seeds and larger scales." | Sec. 6 Limitation | F | T12, T4, T10, T1 |
| "inspire future stable, parameter-efficient, and adaptive architectures." | Sec. 7 Conclusion | F | T6, T5 |
| "using more loop iterations at inference increases computation and energy consumption." | App. B Broader Impacts | L | T9, NEW: broader-impact-risk |
| "may lower the barrier to training capable generative systems, which could be misused for spam, misinformation, or other harmful content generation" | App. B Broader Impacts | L | NEW: broader-impact-risk |
| "may also inherit biases, factual errors, or harmful associations from the training data." | App. B Broader Impacts | L | NEW: broader-impact-risk |
| "future applications of this architecture should be accompanied by appropriate safety evaluations, bias and robustness analyses, content-misuse mitigation, and transparent reporting of compute and energy costs." | App. B Broader Impacts | F | NEW: broader-impact-risk, T9 |

## 8. Training-free looping — Training-Free Looped Transformers

- arXiv: 2605.23872. Exact title: "Training-Free Looped Transformers". Authors: Chen, Li, Liang, Lao, + 1 (UT Austin). The abstract confirms the method: "replacing one large update with smaller damped sub-steps". — [arXiv abs](https://arxiv.org/abs/2605.23872)
- Venue: none stated in the arXiv metadata.
- Version read: arXiv v1 (22 May 2026), the only version. — [arXiv HTML v1](https://arxiv.org/html/2605.23872v1)
- Sections used: **no Limitations or Future Work section, and Sec. 5 Conclusion states no limitations.** The caveats come from Sec. 3 ("Main findings", 3.5), App. G "Failed configurations log", App. I (wall-clock) and App. J (robustness). Borderline paper: candid about failures, but frames none of them as limitations and proposes no future directions.

| quote | section | L/F | tags |
|---|---|---|---|
| "Gains concentrate on knowledge-heavy multiple-choice benchmarks (MMLU-Pro ..., GPQA-Main ..., ARC-Challenge ...), where our method adds roughly +2 pp" | Sec. 5 Conclusion | L (scope; borderline) | T2 |
| "failures concentrated in small distilled checkpoints on some knowledge-MC tasks" | Sec. 3 Main findings | L | T1 |
| "Sub-scale models below ∼1.7B for Qwen3 and ∼3B for Llama lack the mid-layer redundancy the loop relies on." | App. G.8 | L | T1 |
| "undamped iteration diverges visibly within four applications, even on a window of just four mid layers." (naive variant) | App. G.1 | L | T7 |
| "Widening the window degrades performance monotonically" | App. G.3 | L | NEW: loop-placement |
| "Higher-order methods assume the underlying vector field is smooth ...; the looped block does not satisfy this in practice." | App. G.4 | L | T13 |
| "Position-search overfits to the 16-task aggregate" (subsection title) | App. G.5 | L | T10, NEW: loop-placement |
| "iterating individual layers more than twice produces strongly out-of-distribution per-layer states." | App. G.6 | L | T7 |
| "This is the strongest evidence that the block is not contractive" (Anderson at K=8: −18.06 pp) | App. G.7 | L | T7 |
| "Even full decode-loop is bounded at ≈22% slowdown for K=3 on a 4-layer window" | App. I | L | T9 |
| "The 16-task aggregate is a useful screen but a noisy target for sub-pp claims, especially on small MMLU subjects with 100–300 examples." | App. J.1 | L | T10 |

No future directions are stated. App. G says only that it lists failures "so future work knows what does not work"; I did not count that as a direction.

## 9. Adaptive per-layer looping — Adaptive Loops and Memory in Transformers: Think Harder or Know More?

- arXiv: 2603.08391. Exact title: "Adaptive Loops and Memory in Transformers: Think Harder or Know More?". Authors: Frey, Shomali, Bashir, Berghaus, + 2. — [arXiv abs](https://arxiv.org/abs/2603.08391)
- Venue: Latent & Implicit Thinking Workshop @ ICLR 2026, per the arXiv comments.
- Version read: arXiv v3 (11 Mar 2026). — [arXiv HTML v3](https://arxiv.org/html/2603.08391v3)
- Sections used: Sec. 4 "Discussion", which contains an explicit limitations paragraph. Also the Abstract, Sec. 1 and Sec. 3.2.

| quote | section | L/F | tags |
|---|---|---|---|
| "looped models lack the storage capacity of deeper models which use unique weights per layer." | Abstract | L | NEW: knowledge-capacity |
| "a looped model has fundamentally less capacity than a deeper model with N times the number of layers." | Sec. 1 | L | NEW: knowledge-capacity |
| "When we further increase the number of loops, the performance increase is modest relative to the initial improvement from the base model" | Sec. 3.2 | L | T7 |
| "commonsense performance shows a slight downward trend with more loops. ... additional iterations aid algorithmic computation (math) but do not help tasks that depend on stored knowledge" | Sec. 3.2 | L | T7, NEW: knowledge-capacity |
| "While looping seems to improve the routing of information, it cannot compensate for insufficient storage capacities." | Sec. 4 Discussion | L | NEW: knowledge-capacity |
| "our experiments are at a relatively small scale (∼200M parameters, 12 layers, 14B tokens). Whether our conclusions hold at multi-billion parameter scale ... is an open question." | Sec. 4 Discussion | L | T1 |
| "our math evaluation uses BPB instead of accuracy, which limits our ability to make strong claims about reasoning capabilities." | Sec. 4 Discussion | L | T10 |
| "we do not yet provide a full characterization of the efficiency tradeoff between adding loops or memory slots versus simply increasing depth or width under a continuous compute budget." | Sec. 4 Discussion | L | T9, T10 |
| "These limitations will be addressed in follow-up work." (refers to the scale, BPB and efficiency limitations above) | Sec. 4 Discussion | F | T1, T10, T9 |

---

## Tally for this group (9 papers)

Short names: Huginn, Ouro, Retrofit (McLeish et al.), Dreamer, InfiLoop, RecurTrace, FLT, TF-Loop (Training-Free Looped Transformers), AdaLoopMem (Adaptive Loops and Memory). Each paper counts at most once per tag per column.

| tag | papers stating it as L | papers stating it as F | papers stating it at all | paper short names |
|---|---|---|---|---|
| T1 model-scale | 8 | 6 | 8 | L: Huginn, Ouro, Retrofit, Dreamer, RecurTrace, FLT, TF-Loop, AdaLoopMem. F: Huginn, Retrofit, Dreamer, RecurTrace, FLT, AdaLoopMem |
| T2 task-breadth | 4 | 1 | 4 | L: Retrofit, Dreamer, RecurTrace, TF-Loop. F: Retrofit |
| T3 gap-to-explicit-CoT | 1 | 1 | 1 | Huginn (L+F; missing FLOP-matched CoT comparison) |
| T4 interpretability | 2 | 2 | 3 | L: Huginn, Dreamer. F: Dreamer, FLT |
| T5 fixed-vs-adaptive-compute | 3 | 4 | 5 | L: Ouro, Dreamer, RecurTrace. F: Retrofit, Dreamer, RecurTrace, FLT |
| T6 training-cost-or-stability | 3 | 1 | 4 | L: Huginn, Ouro, Retrofit. F: FLT |
| T7 extrapolation-past-trained-depth | 7 | 4 | 7 | L: Ouro, Retrofit, Dreamer, RecurTrace, FLT, TF-Loop, AdaLoopMem. F: Ouro, Retrofit, Dreamer, FLT |
| T8 supervision-dependence | 0 | 0 | 0 | none |
| T9 efficiency-claims | 6 | 4 | 6 | L: Ouro, Dreamer, InfiLoop (borderline), FLT, TF-Loop, AdaLoopMem. F: Ouro, Dreamer, FLT, AdaLoopMem |
| T10 evaluation-rigor | 6 | 4 | 6 | L: Huginn, Ouro, RecurTrace, FLT, TF-Loop, AdaLoopMem. F: Huginn, Ouro, FLT, AdaLoopMem |
| T11 test-time-scaling-and-verification | 0 | 0 | 0 | none |
| T12 RL-or-new-objectives | 1 | 3 | 3 | L: Ouro. F: Huginn, Ouro, FLT |
| T13 theory-gap | 2 | 0 | 2 | L: Ouro, TF-Loop |
| T14 memory-or-attention-over-steps | 3 | 1 | 3 | L: Ouro, Dreamer, RecurTrace. F: Dreamer |
| T15 hybrid-or-tool-integration | 1 | 2 | 2 | L: RecurTrace. F: Dreamer, RecurTrace |
| T16 architecture-generality | 4 | 4 | 6 | L: Huginn, Retrofit, RecurTrace, FLT. F: Huginn, Ouro, Dreamer, FLT |
| NEW: knowledge-capacity | 3 | 2 | 4 | L: Huginn, Ouro, AdaLoopMem. F: Huginn, Dreamer |
| NEW: loop-placement | 1 | 1 | 2 | L: TF-Loop. F: Retrofit |
| NEW: broader-impact-risk | 2 | 1 | 2 | L: Dreamer, FLT. F: FLT |

Per-paper tag sets, for merging:
- Huginn: L {T1, T3, T4, T6, T10, T16, NEW-K}; F {T1, T3, T10, T12, T16, NEW-K}
- Ouro: L {T1, T5, T6, T7, T9, T10, T12, T13, T14, NEW-K}; F {T7, T9, T10, T12, T16}
- Retrofit: L {T1, T2, T6, T7, T16}; F {T1, T2, T5, T7, NEW-LP}
- Dreamer: L {T1, T2, T4, T5, T7, T9, T14, NEW-BI}; F {T1, T4, T5, T7, T9, T14, T15, T16, NEW-K}
- InfiLoop: L {T9 (borderline)}; F {}
- RecurTrace: L {T1, T2, T5, T7, T10, T14, T15, T16}; F {T1, T5, T15}
- FLT: L {T1, T7, T9, T10, T16, NEW-BI}; F {T1, T4, T5, T6, T7, T9, T10, T12, T16, NEW-BI}
- TF-Loop: L {T1, T2, T7, T9, T10, T13, NEW-LP}; F {}
- AdaLoopMem: L {T1, T7, T9, T10, NEW-K}; F {T1, T9, T10}

(NEW-K = NEW: knowledge-capacity; NEW-LP = NEW: loop-placement; NEW-BI = NEW: broader-impact-risk.)

## Full-text access

- I read the full text of all nine papers; none were missing.
- **Ouro**: the arXiv HTML for v6 is only a stub, so I read the v6 PDF ([PDF](https://arxiv.org/pdf/2510.25741v6)) through text extraction. The quotes are verbatim, but section numbers came from the extracted layout and may be off by a subsection, especially in Sec. 5.3 and the appendices.
- **Huginn**: I used the NeurIPS 2025 camera-ready rather than arXiv v2 (Feb 2025), since the camera-ready is newer and adds a combined "Limitations and Conclusions" section.
- **InfiLoop**: I read the HTML in full and keyword-searched the PDF to confirm there is no limitations or impact text.
- **Retrofit**: venue status comes only from the neurips.cc listing page; I did not find an OpenReview entry.

## Observations, not counted

These are my own observations; none of them is an author statement, and none is in the tally.
- **Huginn arXiv v2 text missing from the camera-ready.** Several v2 caveats did not turn up in the camera-ready when I searched for them by keyword: "trained for only 47000 steps on a broadly untested mixture, and the learning rate is never cooled down" (v2 Sec. 5.1), "trained a single model at scale" (v2 Sec. 10), and "Future training runs are likely to train with more optimized learning rate schedules, data mixes and accelerators" (v2 Sec. 10). Counting them would add T6/T10 for Huginn, which the tally already has from other rows. — [arXiv HTML v2](https://arxiv.org/html/2502.05171v2)
- **The near-consensus in this group is T7, alongside T1.** Seven of nine papers say that looping past the trained depth, or adding loops, degrades or destabilizes results (Ouro, FLT, Dreamer, Retrofit, TF-Loop, AdaLoopMem, RecurTrace). Three of them propose explicit fixes as future work: RoPE alternatives (Dreamer), variable-loop training (FLT), and better extrapolation (Ouro).
- **Knowledge capacity recurs often enough to merit a cross-group tag.** Huginn, Ouro and AdaLoopMem say looping helps reasoning but not knowledge storage; Dreamer and Huginn propose retrieval or MoE memory as future work. TF-Loop's training-free looping reports the opposite pattern, with its largest gains on knowledge-heavy multiple-choice tasks. The authors do not discuss that tension.
- **No paper here states supervision dependence (T8) or verification (T11).** RecurTrace's halting head is distilled from an oracle "whether a deeper loop still lowers the loss" signal, and its transfer uses a separate halting head per benchmark with a threshold picked on held-out data. That is a T8-like dependence, but the authors do not frame it as a limitation.
- **InfiLoop is the outlier.** It is a ~7M-parameter puzzle reasoner (Sudoku-Extreme, Maze-Hard, ARC-AGI-1/2), not a language model, and "at scale" in its title refers to test-time iterations (up to 24,960 layer evaluations). Its puzzle-only scope would be T2 and its single model size T1, but the authors state neither.
- **TF-Loop reports failures but proposes nothing.** It has an unusually long failed-configurations log (App. G) yet no limitations section or future directions. Its gains are modest (about +1 to +2.6 pp), and the benchmarks behind its headline numbers are log-likelihood multiple choice. The authors state the second point (App. I) but do not frame it as a limitation.
- **FLT tests only small models.** Its models are 127M and 318M parameters, and its conclusion calls the test-time adaptability "preliminary". I did not count the word "preliminary" as a limitation.
- **Ouro's extrapolation result is mixed.** Benchmark accuracy degrades past T=4 while safety alignment improves with extrapolated depth (Sec. 5.3, 7.1). The authors present the safety result as a strength.
- **Ouro dropped Chinese.** Chinese data was removed after Stage 1 because the tokenizer lacks Chinese vocabulary (Sec. 4). The authors present this as a pipeline decision, not a limitation, so I did not tag it.
- **Retrofit's venue may be a workshop.** It is the only paper here with a likely workshop venue (NeurIPS 2025 Workshop on Efficient Reasoning). AdaLoopMem is also a workshop paper (ICLR 2026). Both have shorter limitation lists than the main-conference papers.
