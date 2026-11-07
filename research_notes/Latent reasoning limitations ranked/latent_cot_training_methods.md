# Self-stated limitations and future directions: latent CoT training methods (9 papers)

Group: CODI, SIM-CoT, Coconut, PCCoT, PMPS, CoLaR, KaVa, ICoT-KD (Deng et al. 2023), ICoT-SI / Stepwise Internalization (Deng et al. 2024).

## Per-paper records: sections used, stated limitations (L), stated future directions (F), theme tags

### Takeaway
I read all nine papers in full, using the latest arXiv PDF and, for SIM-CoT, also the ICLR 2026 camera-ready. The themes stated most often are model scale (T1), task breadth (T2) and training cost or stability (T6), each in 6 of 9 papers, then interpretability (T4) in 5 of 9. None of the nine states a limitation about measured latency or hardware (T9), evaluation rigor (T10), memory or KV growth across steps (T14), or transfer across architectures (T16). Only CoLaR states a failure past its trained range (T7). Five papers have a dedicated Limitations section (CODI, PCCoT, CoLaR, ICoT-KD, ICoT-SI); SIM-CoT, Coconut, PMPS and KaVa do not.

### Cited Findings

#### Tagging conventions used (for merging with other groups)
- One row per statement. When a statement is both a limitation and a direction, it is listed once as L and once as F.
- "Scale to pretraining" statements are tagged **T1**, with "(pretraining)" noted, not T12. T12 is kept for RL and for new or alternative training objectives, including the choice of distillation target or position.
- Multimodal directions are tagged **T15** only, not also T2.
- NEW tags defined in this group:
  - **NEW: safety-or-misuse** — societal risks, misuse, or the need to mitigate threats from capable latent reasoners.
  - **NEW: data-scale-or-diversity** — results limited by the amount or diversity of training data. This is narrower than T8, which is about needing CoT, teacher or step labels at all. A merger may fold it into T8, since the data in question is CoT data.
- Rows marked "*" come only from SIM-CoT's arXiv v2 appendix J, which the ICLR camera-ready dropped. Rows marked "†" are a mitigation recommendation rather than a research direction. See the sensitivity notes under the tally.

---

#### 1. CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation
- arXiv: 2502.21074 (ID and title verified). Version read: **v3 (23 Sep 2025)**, the latest — [arXiv PDF v3](https://arxiv.org/pdf/2502.21074v3). The v3 acknowledgments mention the "rebuttal stages".
- Venue: EMNLP 2025 per the assignment. The arXiv comment field says only "17 pages. Code available…"; I did not check this against the ACL Anthology.
- Sections used: dedicated **§7 Limitations**, which comes after **§6 Conclusion**.

| quote | section | L/F | tags |
|---|---|---|---|
| "Implicit CoT methods inherently trade off interpretability compared to explicit CoT." | §7 Limitations | L | T4 |
| "it operates at the token level and faces limitations in reconstructing multi-token entities." (the probing method) | §7 Limitations | L | T4 |
| "More sophisticated probing techniques may be necessary to recover and visualize full semantic units." | §7 Limitations | F | T4 |
| "probing the token (“:”) responsible for generating the first answer token. However, this choice may be suboptimal" | §7 Limitations | L | T12 (distillation-target design) |
| "probing the token that concludes the CoT reasoning—potentially summarizing the entire process—could offer alternative supervision signals." | §7 Limitations | F | T12 |
| "the current answer prompt, “The answer is:”, is an arbitrary design choice that may influence the effectiveness of knowledge transfer." | §7 Limitations | L | T12 |
| "Investigating these aspects further could enable CODI to extend its distillation framework to broader reasoning tasks." | §7 Limitations | F | T12, T2 |
| "Another limitation of the current continuous training approach is the absence of intermediate gradients until the end of the sequence." | §7 Limitations | L | T6 |
| "which may introduce optimization challenges. This issue could become more pronounced when scaling to more complex problems requiring longer continuous reasoning chains." | §7 Limitations | L | T6 |
| "we don’t have sufficient computation resources to scale the training of CODI on larger models" | §7 Limitations | L | T1 |
| "Future research should explore CODI’s application to more diverse and challenging tasks." | §6 Conclusion | F | T2 |

CODI tags: L = T1, T4, T6, T12; F = T2, T4, T12.

---

#### 2. SIM-CoT: Supervised Implicit Chain-of-Thought
- arXiv: 2509.20317 (verified). Versions read:
  - **arXiv v2 (25 Sep 2025, "Preprint")** — [arXiv PDF v2](https://arxiv.org/pdf/2509.20317v2)
  - **ICLR 2026 camera-ready** ("Published as a conference paper at ICLR 2026") — [ICLR proceedings page](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5d087955ee13fe9a7402eedec879b9c3-Abstract-Conference.html), [camera-ready PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/5d087955ee13fe9a7402eedec879b9c3-Paper-Conference.pdf)
- Venue: ICLR 2026.
- Sections used:
  - **Neither version has a Limitations section.** The §6 Conclusion in both versions states no limitations and no future work.
  - arXiv v2 **Appendix J "Future Directions"**. This appendix is **absent from the camera-ready**, whose appendix outline ends with "declarations on LLM usage and additional case studies".
  - Camera-ready **Appendix C, "Training-Time Overhead"** subsection (not in arXiv v2).
  - Camera-ready **§4.3 Ablation Studies** on decoder size; this ablation is in Appendix B in arXiv v2.

| quote | section | L/F | tags |
|---|---|---|---|
| "the overall parameter count and memory usage are approximately doubled compared with the implicit baselines Coconut and CODI." | App. C Training-Time Overhead (camera-ready) | L | T6 (the authors call the added training time "moderate" and "modest") |
| "simply scaling the decoder to larger variants (3B or 8B) does not yield additional benefits and instead slightly reduces accuracy." | §4.3 Ablation (camera-ready); App. B (v2) | L | T1 (auxiliary-decoder scale) |
| "excessively large decoders may introduce optimization challenges or misalignment with the 1B backbone, ultimately limiting generalization." | §4.3 Ablation (camera-ready); App. B (v2) | L | T6 |
| "(1) Multimodal extension. Incorporating intermediate supervision from images … or videos … could generalize implicit CoT to multimodal reasoning tasks" | App. J Future Directions (v2 only)* | F | T15 |
| "(2) Multi-path implicit reasoning. Exploring branched latent structures, inspired by Tree-of-Thought … methods, may enhance the diversity and robustness of implicit reasoning." | App. J (v2 only)* | F | T11 |
| "(3) Integration with RLHF. Step-level supervision can complement preference optimization and reinforcement learning …, leading to a more reliable and adaptive reasoning framework." | App. J (v2 only)* | F | T12 |
| "(4) Theoretical foundations. Future work may also study implicit reasoning from the perspectives of information theory and representation learning" | App. J (v2 only)* | F | T13 |

SIM-CoT tags: L = T1, T6; F = T11*, T12*, T13*, T15* (all F rows are arXiv-v2-only).

---

#### 3. Coconut: Training Large Language Models to Reason in a Continuous Latent Space
- arXiv: 2412.06769 (verified; the arXiv title has no "Coconut:" prefix). Version read: **v4 (23 Aug 2026)**, the latest — [arXiv PDF v4](https://arxiv.org/pdf/2412.06769v4).
- Venue: the arXiv comment says "Accepted to COLM 2025".
- Sections used:
  - **No dedicated Limitations section.**
  - **§3 Method** ("Training Procedure", "Inference Process")
  - **§5.3 Results and Discussion** ("The LLM still needs guidance…")
  - **§6 Conclusion**
  - **Appendix C "More Discussion"** (C.1 Using More Continuous Thoughts; C.2 Coconut with Larger Models)

| quote | section | L/F | tags |
|---|---|---|---|
| "the sequential nature of the multiple forward passes poses challenges for parallelism." | §3 Training Procedure | L | T6 |
| "Further optimizing the training efficiency of Coconut remains an important direction for future research." | §3 Training Procedure | F | T6 |
| "A challenge lies in determining when to switch between latent and language modes." (they pad to a constant length; a binary classifier "work[s] comparably well") | §3 Inference Process | L | T5 |
| "The LLM still needs guidance to learn latent reasoning." … "we found the models trained this way do not perform any better than no-CoT." (this refers to training without the curriculum) | §5.3 | L | T8, T6 |
| "We leave combining iCoT and Coconut as future work." | §5.3 | F | T6 |
| "further research is definitely needed to develop better and more general strategies for learning reasoning in latent space, especially without the supervision from language reasoning chains." | §5.3 | F | T8, T6 |
| "Further research is needed to refine and scale latent reasoning to pretraining, which could improve generalization across a broader range of reasoning challenges." | §6 Conclusion | F | T1 (pretraining), T2 |
| "adding three continuous thoughts at once – particularly during the final stage transition – leads to a sharp spike in training loss, causing instability." | App. C.1 | L | T6 |
| "Future work will explore finer-grained schedules, such as incrementally adding continuous thoughts one at a time while removing fewer language tokens" | App. C.1 | F | T6 |
| "combining language and latent reasoning—e.g., generating the reasoning skeleton in language and completing the reasoning process in latent space—could provide a promising direction for improving performance and stability." | App. C.1 | F | T15 |
| "these improvements are not as pronounced as those previously demonstrated with GPT-2. One possible reason is that larger models have already undergone extensive language-focused pre-training" (Llama 3.2-3B and Llama 3-8B) | App. C.2 | L | T1 |
| "Universally surpassing language-based CoT likely requires significant research efforts dedicated to latent space pre-training." | App. C.2 | L | T3 |
| (same sentence, recorded as a direction) | App. C.2 | F | T1 (pretraining) |
| "Integrating these recent advancements with Coconut presents an exciting and promising avenue for future research." (the advancements are latent-pretraining work: Geiping et al. 2025, Barrault et al. 2024, Gladstone et al. 2025) | App. C.2 | F | T1 (pretraining) |

Coconut tags: L = T1, T3, T5, T6, T8; F = T1, T2, T6, T8, T15.

---

#### 4. PCCoT: Parallel Continuous Chain-of-Thought with Jacobi Iteration
- arXiv: 2506.18582 (verified). Version read: **v2 (26 Feb 2026)**, the latest — [arXiv PDF v2](https://arxiv.org/pdf/2506.18582v2).
- Venue: the arXiv comment says "Accepted to EMNLP 2025 main conference".
- Sections used:
  - **Dedicated (unnumbered) Limitations section** after §6 Conclusion. The Conclusion itself contains no limitations or future work.
  - **Appendix B**: B.1 More Results, B.2 Convergence of Latent Thought Tokens, B.3 Similarities between Latent Thought Tokens, B.5 Comparison with Standard CoT.

| quote | section | L/F | tags |
|---|---|---|---|
| "The current training method of PCCoT (i.e., CODI) relies on distillation from the CoT teacher task." | Limitations | L | T8 |
| "Though PCCoT is much faster than continuous CoT, it is still much slower than the standard CoT in terms of training due to the distillation training strategy." | Limitations | L | T6 |
| "We have analyzed the behavior of PCCoT on latent thought tokens but fail to explain some of the findings." | Limitations | L | T13, T4 |
| "the latent tokens do not converge in multiple iterations after training, which is not a problem for CoT inference but is nonetheless counter-intuitive." | Limitations | L | T13 |
| "Before scaling up PCCoT to larger models and more diverse settings, it is necessary to figure out how PCCoT works and what the latent thought tokens are doing." | Limitations | F | T1, T2, T4, T13 |
| "with over T = 6 extra iterations, the model performance does not show stable improvement and starts to fluctuate heavily." … "This leads to instability in the training process" | App. B.1 | L | T6 |
| "This may indicate that Jacobi iterations may not operate as we expect: The success of PCCoT is not due to the fast convergence of the latent thought tokens." | App. B.2 | L | T13 |
| "Up to now, we have not found a clear explanation for this phenomenon." (an odd/even interleaved similarity pattern among latent tokens at T = 6) | App. B.3 | L | T13, T4 |
| "we still do not know what does this mean and how it may potentially affect PCCoT in terms of scaling up and extending to more general and complex tasks." | App. B.3 | L | T13 |
| "We are not sure about this and further investigation is needed to understand this phenomenon." (whether the student task regularizes the teacher) | App. B.5 | L | T13 |
| (same sentence, recorded as a direction) | App. B.5 | F | T13 |

PCCoT tags: L = T4, T6, T8, T13; F = T1, T2, T4, T13.

---

#### 5. PMPS: Structural Process Supervision for Latent Chain-of-Thought Reasoning
- arXiv: 2609.09928 (verified; the arXiv title is "Structural Process Supervision for Latent Chain-of-Thought Reasoning", and the method is Prototype-Mediated Process Supervision, PMPS). Version read: **v2 (28 Sep 2026)**, the latest — [arXiv PDF v2](https://arxiv.org/pdf/2609.09928v2).
- Venue: none listed on the arXiv page or the PDF (preprint).
- Sections used: **no Limitations or Future Work section**; **§5 Conclusion** only. Searching the appendix for limitation, future and overhead statements found nothing further.

| quote | section | L/F | tags |
|---|---|---|---|
| "However, the current evaluation of PMPS is mainly limited to mathematical reasoning." | §5 Conclusion | L | T2 |
| "Future work should further explore the effectiveness of PMPS on a broader range of reasoning tasks." | §5 Conclusion | F | T2 |

PMPS tags: L = T2; F = T2.

---

#### 6. CoLaR: Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains
- arXiv: **2505.16552 (ID verified by title)**. Version read: **v6 (3 Feb 2026)**, the latest — [arXiv PDF v6](https://arxiv.org/pdf/2505.16552v6).
- Venue: the PDF footer reads "39th Conference on Neural Information Processing Systems (NeurIPS 2025)".
- Sections used: dedicated **§5 Limitations**, and **§6 Conclusion**, which contains the future work.

| quote | section | L/F | tags |
|---|---|---|---|
| "the overall performance of CoLaR currently approximates explicit CoT reasoning without surpassing it." (the exception is GPQA, where it surpasses its CoT teacher) | §5 Limitations | L | T3 |
| "CoLaR struggles to generalize to non-integer compression factors (e.g., c = 1.5) or to values greater than the maximum training compression factor rmax." (attributed to "discrete tokenization constraints") | §5 Limitations | L | T7, T5 |
| "potential negative impacts include the risk of amplifying existing biases in reasoning processes and possible misuse for generating more convincing misinformation." | §5 Limitations | L | NEW: safety-or-misuse |
| "To mitigate these risks, we recommend careful monitoring of downstream applications." | §5 Limitations | F† | NEW: safety-or-misuse |
| "Future work will focus on addressing non-integer compression factors" | §6 Conclusion | F | T5, T7 |
| "exploring more sophisticated reinforcement learning approaches" | §6 Conclusion | F | T12 |
| "extending our dynamic compression mechanism to more diverse reasoning tasks beyond mathematics." | §6 Conclusion | F | T2 |

CoLaR tags: L = T3, T5, T7, NEW-safety; F = T2, T5, T7, T12, NEW-safety†.

---

#### 7. KaVa: Latent Reasoning via Compressed KV-Cache Distillation
- arXiv: 2510.02312 (verified). Version read: **v2 (7 May 2026)**, header "Published as a conference paper at ICLR 2026" — [arXiv PDF v2](https://arxiv.org/pdf/2510.02312v2). I also checked v1 ([arXiv PDF v1](https://arxiv.org/pdf/2510.02312v1)): its conclusion has the same text, and its §7 Ethics statement is absent.
- Venue: ICLR 2026.
- Sections used:
  - **No Limitations section.**
  - **§6 Conclusion and Discussion**
  - **§7 Ethics and Reproducibility Statements** (camera-ready only)
  - **Appendix G, MetaMathQA Experiments**

| quote | section | L/F | tags |
|---|---|---|---|
| "While the advancement of latent reasoning is linked to the availability of large-scale training data to instill novel reasoning dynamics" (hedged) | §6 Conclusion and Discussion | L | NEW: data-scale-or-diversity |
| "Regarding the interpretability of reasoning traces, a gap exists between latent reasoning and CoT-based reasoning." | §7 Ethics (v2) | L | T4 |
| "We believe it is important that future work focus on explaining the reasoning in both latent and non-latent approaches" | §7 Ethics (v2) | F | T4 |
| "as well as on mitigating any threats posed by increasingly capable models." | §7 Ethics (v2) | F | NEW: safety-or-misuse |
| "distilling a long reasoning trace into a short latent trajectory is challenging because the number of latent tokens remains fixed" (stated as a reason latent methods "only slightly outperform the no CoT baseline") | App. G | L | T5 |
| "each question in the MetaMathQA dataset (unlike in GSM8k-Aug) is associated with multiple reasoning traces, reducing the effective number of unique questions" | App. G | L | NEW: data-scale-or-diversity |

KaVa tags: L = T4, T5, NEW-data; F = T4, NEW-safety. (Hybrid reasoning in App. G is a remedy the paper implements, not future work, so it is not counted.)

---

#### 8. Implicit Chain of Thought Reasoning via Knowledge Distillation (Deng et al.) — "ICoT-KD"
- arXiv: 2311.01460 (verified). Version read: **v1 (2 Nov 2023)**, the only version — [arXiv PDF v1](https://arxiv.org/pdf/2311.01460v1).
- Venue: none listed on the arXiv page or the PDF.
- Sections used: dedicated **§8 Limitations**, **§9 Conclusion and Future Work**, and **§6 Analysis**.

| quote | section | L/F | tags |
|---|---|---|---|
| "implicit CoT, by virtue of its internal processing within hidden states, lacks this transparency." … "it sacrifices human interpretability" | §8 Limitations | L | T4 |
| "after the “Optimize” step, the predicted mixture components are no longer interpretable" | §6 Analysis | L | T4 |
| "our initial approach still relies heavily on the teacher’s thought processes for a starting point." | §8 Limitations | L | T8 |
| "Our current results of implicit CoT still lag behind the performance of explicit CoT." | §8 Limitations | L | T3 |
| "although explicitly producing a chain-of-thought improves the accuracy of final answers further, by a large margin." | §9 Conclusion | L | T3 |
| "one might explore a fully end-to-end joint training strategy using a variational auto encoder … by treating the model’s internal reasoning process as an unobserved latent variable." (proposed "instead of the three-step strategy") | §9 Conclusion and Future Work | F | T12, T6 |
| "Another direction would be using image modeling techniques that can model distributions with multiple modes such as diffusion … to train the thought emulator." | §9 | F | T12 |
| "incorporating this approach into the pretraining process, such that a pretrained language model can both do horizontal explicit chain-of-thought reasoning, and also do vertical implicit chain-of-thought reasoning" | §9 | F | T1 (pretraining), T15 |

ICoT-KD tags: L = T3, T4, T8; F = T1, T6, T12, T15.

---

#### 9. From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step (Deng et al.) — "ICoT-SI" (Stepwise Internalization)
- arXiv: **2405.14838 (ID verified by title)**. Version read: **v1 (23 May 2024)**, the only version — [arXiv PDF v1](https://arxiv.org/pdf/2405.14838v1).
- Venue: none listed on the arXiv page or the PDF.
- Sections used: dedicated **§8 Limitations**, **§9 Conclusions and Future Work**, a scope paragraph in **§1 Introduction**, and **§6.1 and §6.2** of the Analysis.

| quote | section | L/F | tags |
|---|---|---|---|
| "our empirical evaluation focuses on specific reasoning tasks like multi-digit multiplication and grade-school math problems." | §1 Introduction | L | T2 |
| "Due to limitations in available computational resources, experiments on other tasks are out of scope for this work." | §1 Introduction | L | T2 |
| "further research is needed to explore its efficacy across a broader range of tasks and more diverse CoT traces." | §1 Introduction | F | T2 |
| "Even when ICoT-SI is not completely successful due to model capacity limitations, such as on more challenging tasks where it cannot internalize all CoT steps" (GPT-2 Small, 11×11 multiplication) | §6.1 Accuracy-Speed Trade-off | L | T1 |
| "when using ∆ = 16, the training fails to converge" | §6.2 Ablation Studies | L | T6 |
| "Future work could explore adaptive ∆ schedules based on loss values to balance speed and stability more effectively." | §6.2 Ablation Studies | F | T6 |
| "One limitation of the proposed approach is its high training cost due to the finetuning required when removing each set of CoT tokens." | §8 Limitations (Training costs) | L | T6 |
| "where the reasoning chain length grows exponentially with N , training becomes expensive as N increases." | §8 Limitations (Training costs) | L | T6 |
| "Another practical issue we observed is the instability of training with aggressive ∆ values." | §8 Limitations (Instability) | L | T6 |
| "Identifying and addressing unstable dynamics early on, potentially by restarting training as suggested by Hu et al. [9], could be a valuable improvement." | §8 Limitations (Instability) | F | T6 |
| "models trained using our approach lose interpretable intermediate steps." | §8 Limitations (Interpretability) | L | T4 |
| "it might be possible to interpret the internal hidden states of these models using probing techniques" | §8 Limitations (Interpretability) | F | T4 |
| "combining implicit and explicit CoT training could allow users to choose between interpretability and latency" | §8 Limitations (Interpretability) | F | T15 |
| "Undoubtedly, explicit CoT still achieves higher accuracies compared to our approach to implicit CoT." | §8 Limitations (Accuracy) | L | T3 |
| "When scaled to larger models with hundreds of billions of parameters and up to a hundred layers, such as GPT-3 [4], they could potentially solve even more challenging reasoning tasks" | §8 Limitations (Accuracy) | F | T1 |
| "probing the internal processes as the model internalizes each reasoning step could provide insights into the learning mechanisms." | §9 Conclusions and Future Work | F | T4 |
| "developing a mixed-mode approach that combines implicit and explicit CoT reasoning could potentially offer the best of both worlds" | §9 | F | T15 |
| "scaling Stepwise Internalization to larger models and more extensive training/pretraining setups, which could further enhance its effectiveness on a broader range of reasoning tasks." | §9 | F | T1 (pretraining), T2 |

ICoT-SI tags: L = T1, T2, T3, T4, T6; F = T1, T2, T4, T6, T15.

---

#### Tally for this group (9 papers; each paper counted at most once per tag per column)

| tag | papers stating it as L | papers stating it as F | papers stating it at all | paper short names |
|---|---|---|---|---|
| T1 model-scale | 4 | 4 | 6 | L: CODI, SIM-CoT, Coconut, ICoT-SI. F: Coconut, PCCoT, ICoT-KD, ICoT-SI |
| T2 task-breadth | 2 | 6 | 6 | L: PMPS, ICoT-SI. F: CODI, Coconut, PCCoT, PMPS, CoLaR, ICoT-SI |
| T3 gap-to-explicit-CoT | 4 | 0 | 4 | L: Coconut, CoLaR, ICoT-KD, ICoT-SI |
| T4 interpretability | 5 | 4 | 5 | L: CODI, PCCoT, KaVa, ICoT-KD, ICoT-SI. F: CODI, PCCoT, KaVa, ICoT-SI |
| T5 fixed-vs-adaptive-compute | 3 | 1 | 3 | L: Coconut, CoLaR, KaVa. F: CoLaR |
| T6 training-cost-or-stability | 5 | 3 | 6 | L: CODI, SIM-CoT, Coconut, PCCoT, ICoT-SI. F: Coconut, ICoT-KD, ICoT-SI |
| T7 extrapolation-past-trained-depth | 1 | 1 | 1 | L and F: CoLaR (compression factor > trained r_max) |
| T8 supervision-dependence | 3 | 1 | 3 | L: Coconut, PCCoT, ICoT-KD. F: Coconut |
| T9 efficiency-claims | 0 | 0 | 0 | none |
| T10 evaluation-rigor | 0 | 0 | 0 | none |
| T11 test-time-scaling-and-verification | 0 | 1 | 1 | F: SIM-CoT* |
| T12 RL-or-new-objectives | 1 | 4 | 4 | L: CODI. F: CODI, SIM-CoT*, CoLaR, ICoT-KD |
| T13 theory-gap | 1 | 2 | 2 | L: PCCoT. F: SIM-CoT*, PCCoT |
| T14 memory-or-attention-over-steps | 0 | 0 | 0 | none |
| T15 hybrid-or-tool-integration | 0 | 4 | 4 | F: SIM-CoT*, Coconut, ICoT-KD, ICoT-SI |
| T16 architecture-generality | 0 | 0 | 0 | none |
| NEW: safety-or-misuse | 1 | 2 | 2 | L: CoLaR. F: CoLaR†, KaVa |
| NEW: data-scale-or-diversity | 1 | 0 | 1 | L: KaVa |

Ranked by "papers stating it at all":
1. T1, T2, T6 (6 each)
2. T4 (5)
3. T3, T12, T15 (4 each)
4. T5, T8 (3 each)
5. T13 and NEW-safety (2 each)
6. T7, T11 and NEW-data (1 each)
7. T9, T10, T14, T16 (0)

Sensitivity notes for the merger:
- **SIM-CoT camera-ready only** (dropping the * rows from arXiv v2 App. J):
  - T11 F becomes 0, and T11 at-all becomes 0.
  - T12 F becomes 3 and T12 at-all becomes 3.
  - T13 F becomes 1 and T13 at-all becomes 1 (PCCoT only).
  - T15 F becomes 3 and T15 at-all becomes 3.
- **Excluding CoLaR's mitigation advice (†):** NEW-safety F becomes 1 (KaVa). At-all stays 2.
- **Folding NEW-data into T8:** T8 L becomes 4 and T8 at-all becomes 4 (adds KaVa).
- **Excluding SIM-CoT's decoder-scaling ablation from T1** (it concerns the auxiliary decoder, not the backbone): T1 L becomes 3 and T1 at-all becomes 5.

Section types:
- Dedicated Limitations section: CODI, PCCoT, CoLaR, ICoT-KD, ICoT-SI (5 of 9).
- No Limitations section: SIM-CoT, Coconut, PMPS, KaVa (4 of 9). For these I used the Conclusion, Discussion, Ethics and appendices.

### Inferences
This section is **Observations, not counted**: my own observations, not author statements, and not in any tally.
- **Nobody states test-time failure past the trained number of latents or loops.** CODI, SIM-CoT, PCCoT, Coconut, KaVa and PMPS all fix the latent count, yet none states that adding steps beyond the trained count hurts. CoLaR's T7 statement is about compression-factor range, not loop depth. Relevance for this project: the CODI drift past 6 steps seen in the project's own experiments is not something these authors acknowledge.
- **PCCoT's App. B.3 matches the project's odd-even finding.** It reports an "interleaved pattern" in which odd-index and even-index latent tokens are more similar to each other at T = 6, and gives no explanation. This matches the odd-even cycle structure that PMPS describes in CODI ("odd-even collapse"; [PMPS v2 §1](https://arxiv.org/pdf/2609.09928v2)). PMPS states this as a limitation of *baselines*, not of itself, so it is not counted.
- **PMPS critiques other methods, not itself.** It says CODI "optimizes latent embeddings solely via the final answer", and that SIM-CoT's decoder "increases training overhead" and "induces information bottlenecks". These are not self-stated. PMPS claims "only a small amount of training overhead" for itself; I did not count that, since it is framed as a strength.
- **KaVa's App. G instability remarks are about baselines.** It says "CODI and PCCoT … often diverged". KaVa is a KV-cache method, yet it states no KV-memory (T14) or hardware (T9) limitation.
- **CODI reports a gap to CoT-SFT without calling it a limitation.** It reports "99% of [CoT-SFT] accuracy" (GPT-2) and "90% of CoT-SFT performance" (LLaMA-1b), and worse scores on GSM8k-Aug-NL. Because none of this is framed as a limitation, CODI gets no T3 ([CODI v3 §5](https://arxiv.org/pdf/2502.21074v3)).
- **SIM-CoT's appendix K describes trailing latents.** It says implicit reasoning "continues to produce latent tokens even after the correct answer has been reached", and that these latents "simply repeat the final prediction" ([SIM-CoT arXiv v2](https://arxiv.org/pdf/2509.20317v2)). It is presented as a visualization note, not a limitation. It is relevant to T5 and T7.
- **SIM-CoT's ICLR camera-ready removed the Future Directions appendix**, so in the published version it states no future directions at all. That is a meaningful difference from the arXiv record.
- **CoLaR's §5 counts its limitations inconsistently.** It says "two important limitations" but then lists three; the third is societal.
- **Scale is usually phrased as "scale to pretraining".** Coconut, ICoT-KD and ICoT-SI each do this, and CODI cites Geiping et al. (2025) as evidence that scaling is feasible. Other groups may tag these as T12, so mergers should check how they handled pretraining.
- **The 2023–2024 papers state more limitations than the 2025–2026 ones.** ICoT-KD and ICoT-SI have the most explicit, itemized Limitations sections. PMPS (2026) states only one limitation, and KaVa only hedged ones.

### Gaps
This section covers **full-text access, plus unresolved items**.
- **Full-text access:** I accessed the full text of all nine papers through arXiv PDFs. No paper was unavailable.
- **Camera-ready versions:**
  - Checked: SIM-CoT (ICLR proceedings PDF) and KaVa (arXiv v2 is the ICLR camera-ready).
  - Not checked: the CODI and PCCoT ACL Anthology versions, and the Coconut COLM/OpenReview version. For these I used the latest arXiv versions instead: CODI v3 (Sep 2025), PCCoT v2 (Feb 2026), Coconut v4 (Aug 2026). They may differ slightly from the camera-ready.
- **Venues not verified:**
  - CODI: EMNLP 2025 per the assignment, not on its arXiv page.
  - ICoT-KD and ICoT-SI: no venue on arXiv or in the PDF.
  - PMPS: an arXiv preprint with no venue listed.
- **Earlier versions:** I compared only KaVa v1 with v2 and SIM-CoT v2 with its camera-ready. The other papers' earlier versions (Coconut v1–v3, CoLaR v1–v5, CODI v1–v2, PCCoT v1) were not checked; they might contain statements that later versions removed.
- **Extraction caveats:** Text was extracted with pdftotext. I checked the quotes against the extraction and repaired line-break hyphenation. Small typographic differences from the typeset PDF (dashes, quote marks, ligatures) are possible.
