# Latent test-time scaling, sampling, halting and per-token iteration: self-stated limitations and future work (8 papers)

Group: test-time scaling, sampling, halting and per-token iteration for latent or continuous reasoning. Compiled 2026-11-06.

## How this was done, and tagging conventions (read before merging)

- **Sources.** I read the full text of all 8 papers, never abstracts alone. For the two papers with an ACL 2026 camera-ready I used that PDF, because both camera-readies contain a Limitation section that the latest arXiv version lacks (paper 2) or that did not exist before (paper 3). The other 6 come from the newest arXiv HTML version. I also grepped each full text, appendices included, for limitation and future-work wording (limitation, future, leave, open question/challenge, only, preliminary, beyond the scope) and for any other caveats the authors state in passing.
- **Quotes.** Every quote below was checked by a script against the full text of the version read. The check ignores case, punctuation and line-break hyphenation, and confirms the quote is verbatim and at most 30 words. "..." marks text I left out.
- **L/F column values.**
  - `L-own` = the authors state a limitation of their own study or method.
  - `L-field` = the authors state an *unresolved* limitation of latent reasoning, or of the approach they study, in their own concluding, limitation or discussion text. This does **not** cover motivation statements about prior work that the paper claims to fix; those are left out and listed under Observations.
  - `F` = a future direction.
  - A statement worded as "remains an open question/challenge" counts as both, so it is recorded once as L and once as F, as the brief asked.
  - The tally's L column counts `L-own` and `L-field` together. Only two tally cells rest on `L-field` rows alone: T2 for LatentTTS and T4 for ITS-Continuous. A merger who wants own-study limitations only should subtract those two.
- **NEW tags defined** (used only where no T-tag fits):
  - `NEW: hyperparameter-sensitivity`: results depend on hand-tuned inference or sampling hyperparameters, such as noise scale, dropout rate, exit-threshold calibration, or sampling settings for reward-model labels. Four papers in this group state it. **I suggest the other groups check for it**, because it recurs here and has no T-tag.
  - `NEW: white-box-access`: the method needs access to and perturbation of internal hidden states, so it works only with open weights or APIs that expose activations.
  - `NEW: language-mixing`: spontaneous code-switching under latent perturbation, which the authors can neither predict nor control.
- **Untagged rows.** Generic statements such as "early exploration" or "support further progress" are listed but left untagged, so they are not tallied.
- **Short names used in the tally:** SoftCoT++; LatentTTS (You et al., Parallel TTS); ITS-Continuous (Wang et al.); NoisyCoconut; Popescu; TwoScale; FPRM; TaH.

---

## 1. SoftCoT++: Test-Time Scaling with Soft Chain-of-Thought Reasoning

- **arXiv ID:** 2505.11484. Authors: Yige Xu, Xu Guo, Zhiwei Zeng, Chunyan Miao.
- **Venue/date:** arXiv preprint. v1 was posted 16 May 2025 and v2 on 27 May 2025. The arXiv comment says only "14 pages", and I found no evidence of a peer-reviewed venue ([arXiv abs](https://arxiv.org/abs/2505.11484)).
- **Version read:** v2 HTML full text ([arxiv.org/html/2505.11484v2](https://arxiv.org/html/2505.11484v2)).
- **Section(s) used:** §5.4 "Limitations and Future Work". This is a dedicated section, but it is one short paragraph inside §5 Results and Discussions. §6 Conclusion states no limitations or future work, and Appendix C (Discussion) has no limitation statements.
- **Statement count:** 2 L, 2 F.

| quote | section | L/F | tags |
|---|---|---|---|
| "Despite the promising results of SoftCoT++, the exploration of the latent thought distribution remains preliminary." | §5.4 Limitations and Future Work | L-own | T11 |
| "In this work, we focus solely on inference with a fixed 8B-scale model." | §5.4 Limitations and Future Work | L-own | T1 |
| "Extending SoftCoT++ to larger, trainable LLMs opens up several promising research directions." | §5.4 Limitations and Future Work | F | T1 |
| "investigating and understanding how the distribution of soft thoughts evolves during training, and how it interacts with model scale and architecture, is a compelling avenue for future work." | §5.4 Limitations and Future Work | F | T13, T1, T16 |

## 2. Parallel Test-Time Scaling for Latent Reasoning Models (You et al.)

- **arXiv ID:** 2510.07745. Authors: Runyang You, Yongqi Li, Meng Liu, Wenjie Wang, Liqiang Nie, Wenjie Li.
- **Venue/date:** ACL 2026 Main Conference, long paper 2026.acl-long.2069, pp. 44703–44717 ([ACL Anthology](https://aclanthology.org/2026.acl-long.2069/)). arXiv v4 is dated 19 Apr 2026 ("Accepted at ACL 2026 Main Conference").
- **Version read:** the ACL camera-ready PDF ([aclanthology.org/2026.acl-long.2069.pdf](https://aclanthology.org/2026.acl-long.2069.pdf)). I cross-checked it against arXiv v4 ([HTML](https://arxiv.org/html/2510.07745v4) and PDF) and v1 ([HTML](https://arxiv.org/html/2510.07745v1)).
  - **Version conflict:** arXiv v4, and v2/v3 too, has **no Limitation section**. In the v2–v4 HTML the Conclusion's future-work sentence is cut off after "(1) ...; and"; the v4 PDF has the full sentence.
  - The camera-ready restores the unnumbered "Limitation" section that v1 had, with identical wording, and includes direction (2) in full.
- **Section(s) used:** unnumbered "Limitation" section (dedicated); §8 Conclusion and Future Work; App. E (harder benchmarks); App. F.1.
- **Statement count:** 4 L-own, 1 L-field, 2 F.

| quote | section | L/F | tags |
|---|---|---|---|
| "(1) Engineering considerations for real-time deployment. ... real-time use may require additional engineering optimization." | Limitation (unnumbered, after §8) | L-own | T9 |
| "(2) Hyperparameter sensitivity. ... But optimal performance depends on tuning the dropout rate (p) and noise scale (σ)." | Limitation (unnumbered, after §8) | L-own | NEW: hyperparameter-sensitivity, T11 |
| "However, absolute performance remains limited. We attribute to two factors: (a) small model size (e.g., GPT-2-124M, Llama-3.2-1B)" | App. E Additional Results on Harder Benchmarks | L-own | T1 |
| "(b) the inherent challenges of the "latent reasoning" framework, which is still an evolving paradigm and struggle on advanced mathematics (AIME) and Ph.D.-level benchmark (GPQA)." | App. E Additional Results on Harder Benchmarks | L-field | T2 |
| "while a performance gap persists between the Best-of-N with LatentRM and the Coverage@N upper bound, integrating LatentRM provides consistent gains." | App. F.1 Comparison of Latent and Explicit Reasoning with TTS | L-own | T11 |
| "(1) integrating sampling and aggregation into a reinforcement learning framework to optimize latent trajectories through iterative feedback and reward shaping" | §8 Conclusion and Future Work | F | T12, T11 |
| "(2) investigating how to adapt ensemble techniques ... to latent reasoning, addressing the key challenge of applying token-based ensemble methods to continuous latent representations." | §8 Conclusion and Future Work | F | T11 |

## 3. Towards Inference-time Scaling for Continuous Space Reasoning (Wang et al.)

- **arXiv ID:** 2510.12167. Authors: Minghan Wang, Thuy-Trang Vu, Ehsan Shareghi, Gholamreza Haffari.
- **Venue/date:** Findings of ACL 2026, 2026.findings-acl.1338 ([ACL Anthology](https://aclanthology.org/2026.findings-acl.1338/)). arXiv v2 is dated 17 Oct 2025, with the comment "Submitted to AAAI 2026 on July 25, 2025. Under review".
- **Version read:** the ACL Findings camera-ready PDF ([aclanthology.org/2026.findings-acl.1338.pdf](https://aclanthology.org/2026.findings-acl.1338.pdf)). I compared it with arXiv v2 ([HTML](https://arxiv.org/html/2510.12167v2)).
  - **The camera-ready differs materially from arXiv:** it adds §6 "Post-training with Geometric Regularization" and an unnumbered "Limitation" section.
  - It also moves the old §6 Discussion to App. C and the perturbation analysis to App. A.3.
  - arXiv v2 has no limitations section; its only limitation statements are in §6 Discussion and §7 Conclusion.
- **Section(s) used:** unnumbered "Limitation" section (dedicated); §6.3; §7 Conclusion; App. A.3; App. C Discussion.
- **Statement count:** 6 L-own, 4 L-field, 4 F.

| quote | section | L/F | tags |
|---|---|---|---|
| "This work represents an early study of inference-time scaling for continuous reasoning using COCONUT as the primary testbed, and thus has several limitations." | Limitation (unnumbered, after §7) | L-own | T16, T11 |
| "our empirical analysis is conducted on a single reasoning benchmark (GSM8k) with a fixed continuous reasoning configuration" | Limitation (unnumbered, after §7) | L-own | T2, T10 |
| "the observed limitations of reward-model reranking may not directly generalize to other tasks or continuous reasoning architectures." | Limitation (unnumbered, after §7) | L-own | T2, T16 |
| "the training and evaluation of PRM/ORM rely on automatic annotation and sampling-based procedures, which introduce sensitivity to sampling hyperparameters and may affect the stability of reward signals." | Limitation (unnumbered, after §7) | L-own | T11, NEW: hyperparameter-sensitivity |
| "our geometric analysis and post-training intervention focus on a small set of trajectory-level metrics and a lightweight regularization design" | Limitation (unnumbered, after §7) | L-own | T10, T12 |
| "exploring richer geometric representations and more principled training objectives remains an open direction for future work." | Limitation (unnumbered, after §7) | F | T12 |
| "Although we do not re-run the full reward model training pipeline, PRM and ORM still achieve modest gains over the baseline" | §6.3 Experimental Result | L-own | T10, T11 |
| "we view this as a proof-of-concept and leave more effective metric selection and training objectives for further improving inference-time scaling to future work." | §6.3 Experimental Result | F | T12, T11 |
| "established text-based reward modeling methods transfer poorly to continuous reasoning." | §7 Conclusion | L-field | T11 |
| "continuous thoughts lack the geometric separability needed to distinguish correct from incorrect reasoning, with only minimal structural differences between them." | §7 Conclusion | L-field | T11, T4 |
| "effective inference-time scaling in continuous reasoning requires training frameworks that explicitly encourage geometric differentiation in continuous thought space." | §7 Conclusion | F | T12, T11 |
| "supervision applies only to text tokens while latent thoughts are generated without explicit guidance." | App. C Discussion | L-field | T12 |
| "Promising directions include encouraging higher isotropy in thought representations, promoting trajectory diversity through varying geometric patterns, and incorporating contrastive learning" | App. C Discussion | F | T12 |
| "COCONUT's reasoning does not exclusively depend on latent thoughts ... This raises questions about continuous thoughts' actual contribution." | App. A.3 Reasoning under Perturbation | L-field | T4 |

## 4. NoisyCoconut: Counterfactual Consensus via Latent Space Reasoning

- **arXiv ID:** 2605.08221. The title is verified: "NoisyCoconut: Counterfactual Consensus via Latent Space Reasoning". Authors: Michael Jerge, David Evans (University of Virginia).
- **Venue/date:** arXiv preprint, v1 dated 6 May 2026. arXiv metadata lists no journal reference or comment; I did not search further for a venue ([arXiv abs](https://arxiv.org/abs/2605.08221)).
- **Version read:** v1 HTML ([arxiv.org/html/2605.08221v1](https://arxiv.org/html/2605.08221v1)), the only version.
- **Section(s) used:**
  - The dedicated §5.1 Limitations, with sub-paragraphs Generalizability, Discrete Responses, Design space exploration and Cost.
  - The dedicated §5.2 Future Directions: Integration with Learned Latent Reasoning, Language Mixing, Theoretical Foundations.
  - One caveat each from §3.3 and from the §5 Discussion preamble.
- **Statement count:** 13 L, 16 F. This is by far the most self-critical paper in the group. Several of its "limitations" paragraphs are mostly future-work proposals.

| quote | section | L/F | tags |
|---|---|---|---|
| "NoisyCoconut requires the ability to access and perturb internal model states, limiting applicability to open-weight models or settings with sufficient API access." | §5.1 Limitations: Generalizability | L-own | NEW: white-box-access |
| "API designs that expose intermediate representations could enable broader application of techniques like ours" | §5.1 Limitations: Generalizability | F | NEW: white-box-access |
| "Our method's effectiveness varies across architectures. Most notably, gpt-oss-20B required using first-layer hidden states rather than final-layer states" | §5.1 Limitations: Generalizability | L-own | T16 |
| "optimal noise injection strategies may need architecture-specific tuning." | §5.1 Limitations: Generalizability | L-own | T16, NEW: hyperparameter-sensitivity |
| "Understanding why certain architectures respond differently to latent perturbation remains an open question." | §5.1 Limitations: Generalizability | L-own | T13, T16 |
| "Understanding why certain architectures respond differently to latent perturbation remains an open question." | §5.1 Limitations: Generalizability | F | T13, T16 |
| "Our evaluation focused on mathematical reasoning and knowledge-intensive tasks where the set of responses is small (multiple-choice questions) and agreement is well-defined." | §5.1 Limitations: Discrete Responses | L-own | T2 |
| "Extending NoisyCoconut to open-ended generation tasks where "agreement" requires semantic similarity rather than exact matching poses a difficult, but we think not impenetrable, challenge." | §5.1 Limitations: Discrete Responses | F | T2, T11 |
| "exploring whether latent perturbation can improve factuality in long-form generation, where hallucination risks compound, represents a practically important direction." | §5.1 Limitations: Discrete Responses | F | T2 |
| "Our current approach uses fixed hyperparameters, but adaptive noise scaling based on input uncertainty or model confidence could improve robustness." | §5.1 Limitations: Design space exploration | L-own | NEW: hyperparameter-sensitivity, T11 |
| "Our current approach uses fixed hyperparameters, but adaptive noise scaling based on input uncertainty or model confidence could improve robustness." | §5.1 Limitations: Design space exploration | F | NEW: hyperparameter-sensitivity, T11 |
| "Our evaluation was limited to consensus by voting as the aggregation strategy, where outputs are compared via exact match." | §5.1 Limitations: Design space exploration | L-own | T11 |
| "The vast space of possible aggregation strategies remains largely unexplored." | §5.1 Limitations: Design space exploration | L-own | T11 |
| "learned aggregation functions that consider output semantics, confidence-weighted schemes based on token-level probabilities, or methods that leverage partial agreement structure may yield improved performance." | §5.1 Limitations: Design space exploration | F | T11 |
| "our exact-match criterion for agreement ... may be overly restrictive for domains where semantically equivalent but lexically distinct outputs are valid." | §5.1 Limitations: Design space exploration | L-own | T2, T11 |
| "Making the aggregation adaptive would also offer additional opportunities, using additional executions only when necessary to reduce cost" | §5.1 Limitations: Design space exploration | F | T5, T9, T11 |
| "this may be prohibitively expensive for many applications where inference costs already dominate." | §5.1 Limitations: Cost | L-own | T9 |
| "Current NoisyCoconut generates paths independently, missing opportunities for computation sharing." | §5.1 Limitations: Cost | L-own | T9 |
| "architectures that share early computation across paths while branching only at key decision points could reduce overhead while maintaining diversity." | §5.1 Limitations: Cost | F | T9, T11 |
| "Techniques for early termination when paths show rapid convergence could mitigate this cost." | §5.1 Limitations: Cost | F | T5, T9 |
| "multiple executions of a smaller model that can be run locally within the NoisyCoconut framework can be used to provide more reliable results for lower cost" | §5.1 Limitations: Cost | F | T9 |
| "allocate more reasoning paths to instances showing early signs of disagreement while terminating early when paths rapidly converge." | §5.1 Limitations: Cost | F | T5, T9, T11 |
| "Investigating whether our agreement signal remains predictive—or becomes even more informative—when applied to such architectures represents a natural extension." | §5.2 Future Directions: Integration with Learned Latent Reasoning | F | T16, T11 |
| "we did not pursue this direction further, as we lacked a principled mechanism to predict when code-switching would be beneficial or to reliably induce it." | §5.2 Future Directions: Language Mixing | L-own | NEW: language-mixing |
| "Developing methods to detect instances where language mixing improves reasoning—and techniques to controllably trigger such behavior—represents a promising avenue for future work" | §5.2 Future Directions: Language Mixing | F | NEW: language-mixing |
| "Developing a formal framework connecting latent geometry, perturbation stability, and output correctness could guide principled improvements to noise injection strategies" | §5.2 Future Directions: Theoretical Foundations | F | T13 |
| "Although one can imagine future work that attempts to find the most likely answer even with no explicit agreement ... we do not explore that in this work" | §3.3 Aggregating Outputs | L-own | T11 |
| "Although one can imagine future work that attempts to find the most likely answer even with no explicit agreement ... we do not explore that in this work" | §3.3 Aggregating Outputs | F | T11 |
| "Our finding that weaker baseline models exhibit larger absolute improvements under high-agreement filtering merits further investigation." | §5 Discussion | F | T13 |

## 5. Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts (Popescu et al.)

- **arXiv ID:** 2607.20519. The title is verified as above. Authors: Andrei Cristian Popescu, Haitz Sáez de Ocáriz Borde, Pietro Liò (Cambridge).
- **Venue/date:** arXiv preprint, v1 dated 8 Jul 2026. arXiv lists no venue ([arXiv abs](https://arxiv.org/abs/2607.20519)).
- **Version read:** v1 HTML ([arxiv.org/html/2607.20519v1](https://arxiv.org/html/2607.20519v1)).
- **Section(s) used:** **There is no Limitations section and no Future Work section.**
  - I used §6 Conclusion (one future-work sentence) and one future-design sentence in §1 Introduction.
  - I also used caveats stated in passing in §5.5 and App. C.4.
- **Statement count:** 3 L, 2 F.

| quote | section | L/F | tags |
|---|---|---|---|
| "several of these differences are small and reported on single runs." | §5.5 From controlled tasks to large-scale checkpoints | L-own | T10 |
| "Unlike MANO, we cannot retrain under a different prior, so the trajectory is whatever pretraining produced." | §5.5 From controlled tasks to large-scale checkpoints | L-own | T10 |
| "This experiment is not a length-generalization benchmark. Accuracy may degrade once expressions exceed the training range." | App. C.4 Difficulty Extrapolation on MANO | L-own | T7 |
| "We hope this trajectory–readout perspective motivates future work on alternative pretraining objectives and looped Transformer architectures." | §6 Conclusion | F | T12, T5 |
| "future adaptive-depth methods should treat both components as first-class design choices." | §1 Introduction | F | T5 |

## 6. Two-Scale Latent Dynamics for Recurrent-Depth Transformers (Pappone et al.)

- **arXiv ID:** 2509.23314. The title is verified as above. Authors: Francesco Pappone, Donato Crisostomi, Emanuele Rodolà (Sapienza).
- **Venue/date:** NeurIPS 2025 Workshop on Unifying Representations in Neural Models. The venue comes from the lab's publication page ([GLADIA](https://gladia.di.uniroma1.it/publication/2025-pappone-twoscale/)), and NeurIPS lists a 2025 virtual entry ([neurips.cc](https://neurips.cc/virtual/2025/127222)). arXiv v2 is dated 13 Nov 2025.
- **Version read:** v2 HTML ([arxiv.org/html/2509.23314v2](https://arxiv.org/html/2509.23314v2)).
- **Section(s) used:** §5 "Conclusions and limitations", which has a dedicated "Limitations." paragraph that also holds the future work. No other part of the paper states limitations.
- **Statement count:** 4 L, 3 F.

| quote | section | L/F | tags |
|---|---|---|---|
| "Our evidence is observational and iteration-based rather than mechanistic" | §5 Conclusions and limitations: Limitations | L-own | T13, T4 |
| "PCA reduces geometry to 2D" | §5 Conclusions and limitations: Limitations | L-own | T4, T10 |
| "experiments use a single GPT-2-scale model and a specific recurrence pattern" | §5 Conclusions and limitations: Limitations | L-own | T1, T16, T10 |
| "measurements are averaged over tokens" | §5 Conclusions and limitations: Limitations | L-own | T10 |
| "Future work includes scaling to larger models and datasets" | §5 Conclusions and limitations: Limitations | F | T1 |
| "learning per-block calibration for the exit rule" | §5 Conclusions and limitations: Limitations | F | T5, NEW: hyperparameter-sensitivity |
| "integrating geometry-aware controllers with dynamic-depth or token-level TTC strategies." | §5 Conclusions and limitations: Limitations | F | T5 |

## 7. Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers (FPRM; Movahedi et al.)

- **arXiv ID:** 2606.18206. The title is verified as above. Authors: Sajad Movahedi, Vera Milovanović, Shlomo Libo Feigin, Alexander Theus, Thomas Hofmann, Valentina Boeva, T. Konstantin Rusch, Antonio Orvieto.
- **Venue/date:** arXiv preprint, v1 dated 16 Jun 2026. Code is at github.com/nilskiKonjIzDunava/fprm. arXiv lists no venue ([arXiv abs](https://arxiv.org/abs/2606.18206)).
- **Version read:** v1 HTML ([arxiv.org/html/2606.18206v1](https://arxiv.org/html/2606.18206v1)).
- **Section(s) used:**
  - The "Limitations." paragraph inside §5 Discussion (dedicated but short).
  - The other §5 Discussion paragraphs: "Enabling ACT at inference time", "The role of hierarchy", "Scaling behavior of FPRM".
  - One caveat in App. D, and §6 Conclusion, which has only a generic future-work sentence.
- **Statement count:** 6 L-own, 1 L-field, 5 F.

| quote | section | L/F | tags |
|---|---|---|---|
| "we test our model only on algorithmic tasks and not on natural language." | §5 Discussion: Limitations | L-own | T2 |
| "It is an open challenge to demonstrate that the compositional reasoning behavior that latent models exhibit on algorithmic tasks translates to other domains." | §5 Discussion: Limitations | L-own | T2 |
| "It is an open challenge to demonstrate that the compositional reasoning behavior that latent models exhibit on algorithmic tasks translates to other domains." | §5 Discussion: Limitations | F | T2 |
| "even though the base architecture of FPRM could adopt any model (e.g. CNN, MLP, state-space models), we limit our experiments to Transformers." | §5 Discussion: Limitations | L-own | T16 |
| "note that the halted samples remain in the batch until the last sample in the batch halts. We leave the efficient implementation for future work." | §5 Discussion: Enabling ACT at inference time | L-own | T9 |
| "note that the halted samples remain in the batch until the last sample in the batch halts. We leave the efficient implementation for future work." | §5 Discussion: Enabling ACT at inference time | F | T9 |
| "We identify the theoretical explanation of the role of hierarchy through the lens of optimization and signal propagation as an interesting direction for future work." | §5 Discussion: The role of hierarchy in HRM and TRM | F | T13 |
| "the optimality of the algorithms learned by looped models is not guaranteed, with great variation not only possible but likely." | §5 Discussion: Scaling behavior of FPRM | L-field | T13 |
| "This suboptimal scaling has also been observed in recurrent-in-depth state-space models (Movahedi et al., 2025), a behavior that we also observe in Figure 4." | §5 Discussion: Scaling behavior of FPRM | L-own | T13 |
| "we propose it as an open challenge to find a latent reasoning architecture that achieves a solution with logarithmic complexity to state-tracking while remaining Turing-complete" | §5 Discussion: Scaling behavior of FPRM | F | T13 |
| "we exclude the hardest puzzles from the groups, since they fail to halt at the current max." | App. D Description of Figure 1 | L-own | T5 |
| "We hope these architectural modifications and the accompanying insights will support further progress on latent reasoning models." | §6 Conclusion | F | untagged (generic) |

## 8. Think-at-Hard: Dynamic Looped Transformers for Improved Reasoning (TaH; Fu et al.)

- **arXiv ID:** 2511.08577. The title is verified as "Think-at-Hard: Dynamic Looped Transformers for Improved Reasoning". Authors: Tianyu Fu, Yichen You, Zekai Chen, Guohao Dai, Huazhong Yang, Yu Wang.
- **Venue/date:** ICML 2026; the arXiv v4 comment reads "Accepted by ICML'26". v4 is dated 28 Aug 2026 ([arXiv abs](https://arxiv.org/abs/2511.08577)).
- **Version read:** v4 HTML ([arxiv.org/html/2511.08577v4](https://arxiv.org/html/2511.08577v4)). I did not fetch a PMLR proceedings PDF, but v4 has an ICML-style Impact Statement and post-acceptance date, so it is probably the camera-ready text.
- **Section(s) used:**
  - The dedicated limitations section exists **only in the appendix**: App. A.6 "Limitations and Future Work". §7 Conclusion states no limitations.
  - I also used caveats in App. A.2.4, A.2.2, A.1.3, A.2.1, §6.3 and the Impact Statement.
- **Statement count:** 9 L, 5 F (of which 1 L and 1 F are generic and untagged).

| quote | section | L/F | tags |
|---|---|---|---|
| "our models use SFT only on limited, publicly accessible data. Consequently, performance may differ between the two." | App. A.6 Limitations and Future Work | L-own | T10 |
| "Due to resource constraints, this work focuses on SFT with public data; we leave these combinations for future work." | App. A.6 Limitations and Future Work | L-own | T12 |
| "(1) integrating TaH with online distillation to provide stronger supervision" | App. A.6 Limitations and Future Work | F | T12 |
| "(2) applying reinforcement learning to optimize the compute–accuracy trade-off and to learn continuation policies beyond imitation." | App. A.6 Limitations and Future Work | F | T12, T5 |
| "We note that our current implementation is not yet optimized at the system level, so there remains room for further efficiency improvements." | App. A.2.4 Real-world Efficiency | L-own | T9 |
| "the LoRA Switching and Other overheads (bookkeeping) are relatively high due to the Python-level implementation of dynamic control flow." | App. A.2.4 Real-world Efficiency | L-own | T9 |
| "we plan to continue refining the implementation to further reduce latency and memory overhead." | App. A.2.4 Real-world Efficiency | F | T9 |
| "However, the ideal oracle policy achieves even higher gains, indicating future potential." | App. A.2.2 Additional Oracle Analysis | L-own | T5 |
| "However, the ideal oracle policy achieves even higher gains, indicating future potential." | App. A.2.2 Additional Oracle Analysis | F | T5 |
| "confirming the practical benefits of selective iteration, even with imperfect decisions from the neural decider." | §6.3 Design Choice Exploration | L-own | T5 |
| "The focus of this work is how to selectively attribute iterations to different tokens, rather than increasing the average iteration depth." | App. A.1.3 Iteration Depth Flexibility | L-own | T5 |
| "AlwaysThink-4B is not evaluated due to out-of-memory during training." | App. A.2.1 Math-Only Training | L-own | T10 |
| "While this is an early exploration in this direction" | Impact Statement | L-own | untagged (generic) |
| "opening a new direction for better reasoning under parameter constraints." | §7 Conclusion | F | untagged (generic) |

---

## Tally for this group (8 papers)

Each paper counts at most once per tag per column. "Papers stating it as L" includes `L-own` and `L-field`; only two cells rest on `L-field` rows alone: T2 for LatentTTS and T4 for ITS-Continuous. In the names column, each paper is marked (L), (F) or (L+F).

| tag | papers stating it as L | papers stating it as F | papers stating it at all | paper short names |
|---|---|---|---|---|
| T1 | 3 | 2 | 3 | SoftCoT++ (L+F), LatentTTS (L), TwoScale (L+F) |
| T2 | 4 | 2 | 4 | LatentTTS (L), ITS-Continuous (L), NoisyCoconut (L+F), FPRM (L+F) |
| T3 | 0 | 0 | 0 | — |
| T4 | 2 | 0 | 2 | ITS-Continuous (L), TwoScale (L) |
| T5 | 2 | 4 | 5 | NoisyCoconut (F), Popescu (F), TwoScale (F), FPRM (L), TaH (L+F) |
| T6 | 0 | 0 | 0 | — |
| T7 | 1 | 0 | 1 | Popescu (L) |
| T8 | 0 | 0 | 0 | — |
| T9 | 4 | 3 | 4 | LatentTTS (L), NoisyCoconut (L+F), FPRM (L+F), TaH (L+F) |
| T10 | 4 | 0 | 4 | ITS-Continuous (L), Popescu (L), TwoScale (L), TaH (L) |
| T11 | 4 | 3 | 4 | SoftCoT++ (L), LatentTTS (L+F), ITS-Continuous (L+F), NoisyCoconut (L+F) |
| T12 | 2 | 4 | 4 | LatentTTS (F), ITS-Continuous (L+F), Popescu (F), TaH (L+F) |
| T13 | 3 | 3 | 4 | SoftCoT++ (F), NoisyCoconut (L+F), TwoScale (L), FPRM (L+F) |
| T14 | 0 | 0 | 0 | — |
| T15 | 0 | 0 | 0 | — |
| T16 | 4 | 2 | 5 | SoftCoT++ (F), ITS-Continuous (L), NoisyCoconut (L+F), TwoScale (L), FPRM (L) |
| NEW: hyperparameter-sensitivity | 3 | 2 | 4 | LatentTTS (L), ITS-Continuous (L), NoisyCoconut (L+F), TwoScale (F) |
| NEW: white-box-access | 1 | 1 | 1 | NoisyCoconut (L+F) |
| NEW: language-mixing | 1 | 1 | 1 | NoisyCoconut (L+F) |

**Ranking within this group** (papers stating it at all):
- 5 papers: T5 adaptive compute and T16 architecture generality.
- 4 papers: T2, T9, T10, T11, T12, T13 and NEW: hyperparameter-sensitivity.
- 3 papers: T1.
- 2 papers: T4.
- 1 paper: T7, NEW: white-box-access and NEW: language-mixing.
- 0 papers: T3, T6, T8, T14 and T15.

Two splits are worth noting:
- **T5** is stated mainly as a future direction: 4 papers as F against 2 as L.
- **T10** is stated only as a limitation: 4 as L, none as F.

## Papers whose full text I could not access

- **None.** All eight papers were read in full: six from arXiv HTML and two from ACL Anthology camera-ready PDFs.
- **Caveats on versions:**
  - **TaH:** I read arXiv v4, which says "Accepted by ICML'26", rather than the PMLR proceedings version.
  - **No venue confirmed:** NoisyCoconut, Popescu and FPRM have only arXiv v1, which lists no venue, and I did not search for one. If they later appeared with a Limitations section, it is not reflected here.
  - **SoftCoT++:** I found no venue; arXiv v2 was used.

## Observations, not counted

These are my own observations, or statements the authors make about *other* work. None of them is in the tally.

**Version-related**
- **LatentTTS versions.** The arXiv v1 Conclusion had a different future-work sentence: TTS could become "an adaptive reasoning process—capable of learning exploration strategies, adjusting compute allocation dynamically, and generalizing across tasks". That would have been T5/T2. The camera-ready replaced it with the RL and ensemble directions, so I did not count it.
  - arXiv v2–v4 dropped the Limitation section entirely. Anyone reading only the latest arXiv version would miss the T9 and hyperparameter-sensitivity statements.

**Motivation statements and diagnoses not counted**
- **LatentTTS** calls earlier noise-injection attempts (CoLaR, SoftCoT++) "preliminary" (§2.2). That is a claim about other papers.
- **SoftCoT++** motivates itself with a limitation of its predecessor SoftCoT: a fixed latent thought for a given input prevents diverse exploration (Abstract, §2.2). The paper claims to fix this, so it is not counted.
- **Popescu's** central result is itself a critique of learned halting gates: they "conflate" trajectory formation and exit selection, and "current learned-gate objectives can fail". This is a T5-type diagnosis of *other* models, such as the Ouro ponder gate, and the paper offers post-hoc readouts as its fix, so I did not count it as an unresolved limitation.
  - App. I passes on the Ouro paper's report that RLVR gave no significant gains, with dynamic early exit as a central difficulty for RL. That is Ouro's claim, not Popescu's.
  - The Ouro latency figures are described as "estimated latency".
- **FPRM's** critique that TRM/HRM's ACT "either halts too early... or too late", and that ACT does not scale compute with difficulty, is a limitation of prior work that FPRM claims to fix. Not counted.
  - App. E notes that hard Sudoku inputs "peak beyond the training compute limit". That is relevant to T7, but it is stated as a positive finding, not as a limitation.
- **TaH:**
  - It attributes "latent overthinking" to always-iterate baselines such as AlwaysThink and Ouro, where a second pass revises correct tokens into wrong ones. That is T7-like, but it describes other methods.
  - It reports that a dynamic oracle policy "causing training collapse" (T6-like). That concerns an alternative design it rejected.
  - Its decider learns from oracle labels produced by a reference model, which is T8-like, but the authors do not frame this as a limitation.
  - Duo-causal attention adds depth-2 KV entries, and App. A.2.5 models this, but the paper never calls KV growth a limitation (T14).

**Patterns across the group**
- **Hidden-state convergence or geometry as the halting signal.** Three papers use it: TwoScale's second-order "acceleration" exit, FPRM's fixed-point residual, and Popescu's hidden-cosine and L2 readouts. None of them states, as a limitation, that convergence-based exits might fire on a stable but wrong attractor. NoisyCoconut's Discussion reads agreement as a "stable attractor basin" without addressing this case either.
- **No paper here states T3** (gap to explicit CoT). LatentTTS App. F.1 claims the opposite: COCONUT plus TTS beats explicit CoT on the same GPT-2 backbone at lower compute.
- **No paper states T6, T8, T14 or T15.** For a group about test-time compute, T14 (KV and memory growth over steps) being absent is notable.
- **Two papers find that COCONUT-style latents carry little usable signal for verification.** In ITS-Continuous, separability is poor, and Pass@5 stays at 12.59% even under full noise replacement. In LatentTTS, LatentRM still leaves a gap to Coverage@N.
- **NoisyCoconut's models.** It runs off-the-shelf open-weight LLMs (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mixtral-8B-Instruct-v0.1, gpt-oss-20B and DeepSeek-R1-Distill-Qwen-7B; §4.1 Table 1) with a hidden-state feedback loop, not models trained for latent reasoning. Its own §5.2 proposes testing trained recurrent latent architectures as future work.
- **Missing generalization caveat in TwoScale.** Its exit is evaluated by perplexity and latency on web text with one GPT-2-scale model, and it states no task-breadth (T2) limitation even though no reasoning benchmarks are run.
