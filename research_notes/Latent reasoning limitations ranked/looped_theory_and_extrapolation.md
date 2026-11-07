# Looped and recurrent transformers: theory, stability and depth extrapolation. Self-stated limitations and future work (7 papers)

Group scope: theory, stability and depth extrapolation of looped/recurrent transformers. All seven papers were found, their IDs checked against the titles, and each read in full: arXiv HTML and PDF, latest version, plus the NeurIPS camera-ready for Bansal. The per-paper sections come first, then the group tally, the access notes, and observations that are not counted.

Counting rules used throughout:
- Only statements the authors make are counted. A statement that is both a limitation and a direction gets two rows: once as L, once as F.
- Statements are taken from every place the authors make them: a Limitations or Future Work paragraph, Conclusion, Discussion, appendix, NeurIPS checklist, and explicit "we leave X for future work" or "X is a weakness of our method" sentences in results sections. Rows from outside a limitations or conclusion section are marked "(results)", "(appendix)" or "(checklist)" so the writer can filter them out.
- Statements about prior work, and motivational claims about the whole field, are not counted. They are listed under Observations.

Tag conventions, so this group can be merged with the others:
- Compute cost during training (BPTT memory, gradient tracking, loss variants that save passes) → T6. Compute cost of analysis or inference (influence matrices, compute-saving claims, hardware) → T9.
- Length generalization, overthinking, "conversion" of extra depth into accuracy, and whether a stabilised computation extrapolates → T7.
- Halting or stopping rules and when-to-stop schedules → T5.
- Missing proofs, mechanisms shown only as toy or sufficient-condition arguments, and unexplained phenomena → T13.
- Loop topology (cyclic vs sequential or multiple loops, middle looping) and positional-embedding choice → T16.
- Caveats on the validity of a diagnostic or attribution tool → T4, plus T10 when they concern measurement reliability.
- One NEW tag: **NEW: memorization-perplexity-tradeoff**, meaning looping worsens perplexity or memorization (closed-book QA) while helping reasoning.

---

## 1. Reasoning with Latent Thoughts: On the Power of Looped Transformers (Saunshi, Dikkala, Li, Kumar, Reddi)

- arXiv: 2502.17416. Title verified on [arXiv abs](https://arxiv.org/abs/2502.17416).
- Venue/date: ICLR 2025 (arXiv comment; the PDF header reads "Published as a conference paper at ICLR 2025"). v1 was submitted 24 Feb 2025 and is the only version.
- Version read: v1, both the [HTML](https://arxiv.org/html/2502.17416v1) and the PDF. v1 is the camera-ready.
- Sections used: §7 "Conclusions, limitations and future work" (a combined section; there is no separate Limitations section), §3.1, §3.3, §3.4.
- Takeaway: the authors state one main limitation, that only a subset of reasoning problems was tested. Their directions are theoretical (why the inductive bias exists, a formalization of reasoning) and inference-time scaling with loops.

| # | quote | section | L/F | tags |
|---|---|---|---|---|
| S1 | "While we test looped models on a subset of reasoning problems, a natural question is whether the results hold for many other forms of reasoning (e.g. multimodal and common-sense reasoning)." | §7 | L | T2 |
| S1F | (same quote) | §7 | F | T2, T15 |
| S2 | "a succinct formalization of reasoning problems itself is an interesting future direction." | §7 | F | T13 |
| S3 | "the inductive bias towards improved reasoning performance at the same perplexity is very intriguing and deserves further exploration." | §7 | F | T13 |
| S4 | "We hope this inspires future exploration on using looped models for more efficient inference-time scaling to aid with deeper reasoning." | §7 | F | T11, T5 |
| S5 | "Why such an inductive bias occurs is still an open question, and we believe that understanding this is an important future direction." | §3.3 | L | T13 |
| S5F | (same quote) | §3.3 | F | T13 |
| S6 | "We leave the exploration of the best looping strategies for future work." (said of the middle-looping variant) | §3.3 | F | T16 |
| S7 | "We believe that leveraging looping explicitly for inference-time scaling is a very promising future direction." | §3.4 | F | T11, T5 |
| S8 | "looped models have a bias towards improving reasoning, despite having worse perplexity and memorization abilities." (also in the Table 3 caption: "worse on perplexity and closed book QA (memorization benchmarks)") | §3.1 (results) | L | NEW: memorization-perplexity-tradeoff |

Sources: [arXiv HTML v1](https://arxiv.org/html/2502.17416v1), [arXiv PDF](https://arxiv.org/pdf/2502.17416v1)

---

## 2. Looped Transformers for Length Generalization (Fan, Du, Ramchandran, Lee)

- arXiv: 2409.15647. Title verified on [arXiv abs](https://arxiv.org/abs/2409.15647).
- Venue/date: ICLR 2025 (arXiv comment; the PDF header reads "Published as a conference paper at ICLR 2025"). There are five versions, v1 of 24 Sep 2024 through v5 of 12 May 2025.
- Version read: v5, both the [HTML](https://arxiv.org/html/2409.15647v5) and the PDF.
- Sections used: §7 "Limitations and conclusion" (a combined section), §5 Related work (positional-embedding paragraph), §6.4 (stopping criterion) and Appendix F (experimental details).
- Takeaway: this is the most explicit limitations paragraph in the group. It covers the formal task class (no sequential loops), training compute, limited steps and lengths, NoPE only, and the need for ground-truth step counts.

| # | quote | section | L/F | tags |
|---|---|---|---|---|
| F1 | "Our current definition of n-RASP-L does not support tasks that require multiple loops followed by each other." | §7 | L | T13, T2, T16 |
| F1F | "one important future direction would be extending our current definition to support multiple loops and identifying a larger class of tasks that can be implemented only with multiple loops" | §7 | F | T13, T2, T16 |
| F2 | "For training, direct looped training could be computationally demanding when the number of looped steps is too large." | §7 | L | T6 |
| F2F | "A possible workaround for more efficient training could be stopping the gradient tracking for earlier steps like Clark et al. (2023), but there might be a trade-off" | §7 | F | T6 |
| F3 | "We only train the looped Transformers for a limited number of steps and lengths due to a lack of computing resources." | §7 | L | T1, T7 |
| F3F | "With more diverse training data, the looped model has the potential to generalize to even longer test lengths." | §7 | F | T7 |
| F4 | "We use NoPE for simplicity" | §7 | L | T16 |
| F4F | "an orthogonal direction is to use more delicate positional embedding to further improve length generalization performance." | §7 | F | T16, T7 |
| F5 | "our step-dependent supervision requires the ground-truth number of steps in the training data, which is an additional requirement compared with normal end-to-end training." | §7 | L | T8 |
| F6 | "We leave further investigation with different positional embeddings to future work." | §5 Related work | F | T16, T7 |
| F7 | "we believe that our method has the potential to generalize to more challenging tasks." | §7 | F | T2 |
| F8 | "the per-sample stopping criterion performs almost as well as when using B = N_test to decide for converging tasks, but the performance is worse in non-converging tasks." | §6.4 (results) | L | T5 |
| F9 | "non-converging tasks are less tolerant when choosing which step to stop" | App. F (appendix) | L | T5 |

Note: right after F5 the authors add, "However, we still require fewer assumptions than CoT training."

Sources: [arXiv HTML v5](https://arxiv.org/html/2409.15647v5), [arXiv PDF v5](https://arxiv.org/pdf/2409.15647v5)

---

## 3. End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking (Bansal, Schwarzschild, Borgnia, Emam, Huang, Goldblum, Goldstein)

- arXiv: 2202.05826. Title verified on [arXiv abs](https://arxiv.org/abs/2202.05826). The NeurIPS proceedings page uses a shorter subtitle, "Extrapolation without Overthinking" ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7f70331dbe58ad59d83941dfa7d975aa-Abstract.html)), but the camera-ready PDF keeps the "Logical Extrapolation" title.
- Venue/date: NeurIPS 2022 (Advances in NeurIPS 35). arXiv v3 is dated 14 Oct 2022.
- Version read: arXiv v3 PDF and HTML, which include the appendix, plus the [NeurIPS camera-ready main PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/7f70331dbe58ad59d83941dfa7d975aa-Paper-Conference.pdf). The camera-ready Conclusion matches v3 word for word.
- Sections used: **there is no Limitations section.** I used §6 Conclusion and Appendix A.3–A.5. No checklist appears in the arXiv v3 or camera-ready main PDFs, and the supplemental-PDF link did not return a PDF.
- Takeaway: the only general limitation is domain scope. The other statements are appendix-level: run-time-penalty losses, a cheaper progressive-loss variant, an extrapolation ceiling on chess, and chess training that often fails to converge.

| # | quote | section | L/F | tags |
|---|---|---|---|---|
| B1 | "Our findings are limited to the domains we consider" | §6 Conclusion | L | T2 |
| B1F | "further exploration into how our methods will perform in other settings is needed." | §6 Conclusion | F | T2 |
| B2 | "We believe a more thorough exploration of Deep Thinking networks with variants of such run-time penalties would be interesting future work." | App. A.3 (appendix) | F | T12, T5 |
| B3 | "We believe such optimizations could be an interesting avenue for future work" (a progressive-loss variant with n = 0 that saves forward-pass computation) | App. A.4 (appendix) | F | T6, T12 |
| B4 | "there is an apparent ceiling on generalizing from the easy puzzles to much harder ones." (chess) | App. A.5 (appendix) | L | T7 |
| B4F | "We leave investigation into this phenomenon for future work." | App. A.5 (appendix) | F | T7 |
| B5 | "For all categories, about half of the chess models did not converge." (models below 99% train accuracy were filtered out) | App. A.5 (appendix) | L | T6 |

Sources: [arXiv PDF v3](https://arxiv.org/pdf/2202.05826v3), [NeurIPS 2022 camera-ready](https://proceedings.neurips.cc/paper_files/paper/2022/file/7f70331dbe58ad59d83941dfa7d975aa-Paper-Conference.pdf)

---

## 4. Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping (Kuo, Chayti, Reizinger, Brendel, Jaggi)

- arXiv: 2606.29983. Title verified on [arXiv abs](https://arxiv.org/abs/2606.29983).
- Venue/date: NeurIPS 2026 (arXiv comment "Published at NeurIPS 2026"). v1 is dated 29 Jun 2026 and v2 14 Oct 2026.
- Version read: v2 PDF and HTML.
- Sections used: an unnumbered Conclusion with a paragraph headed "Limitations and future work.", §4.2, §4.3, §5.2, Appendices A.1, C.3, C.4, C.7, C.8, D and E, and the NeurIPS checklist (Q2 Limitations, Q10 Broader impacts). The checklist points to §§4.2, 4.3 and 5 and Appendices D and E as the places limitations are discussed.
- Takeaway: the central self-stated limitation is that stability does not imply extrapolation. It recurs in about six places. Others are the synthetic-only setting (pretrained models and real reasoning are untested) and toy theory that is only local or sufficient-condition. The directions are stopping objectives that track which computations generalise, and richer tasks.

| # | quote | section | L/F | tags |
|---|---|---|---|---|
| K1 | "their impact on pretrained models and realistic reasoning tasks remains to be established." | Conclusion, Lim. & FW | L | T1, T2 |
| K2 | "future stopping objectives should account for which computations generalise, beyond maintaining stochasticity or minimising in-distribution loss." | Conclusion, Lim. & FW | F | T5, T7, T12 |
| K3 | "Promising directions include incorporating priors that favour reusable computation and combining learned stochastic stopping with architectural stabilisation." | Conclusion, Lim. & FW | F | T5, T7, T12 |
| K4 | "Evaluating these approaches on richer tasks would clarify when training-time depth selection supports both reliable extrapolation and effective use of test-time computation." | Conclusion, Lim. & FW | F | T2, T7, T5 |
| K5 | "RL-Halting often improves this accuracy–stability trade-off, but can also stabilise a suboptimal computation." | Conclusion | L | T5, T7 |
| K6 | "However, this stabilisation does not by itself guarantee that the selected trajectory extrapolates best." | §4.2 (Observation 2) | L | T7 |
| K7 | "On Copy, RL-Halting reduces oracle OOD standard deviation from 13.2 to 2.7 relative to K = n, but has lower oracle accuracy and a shorter extrapolation frontier." | §4.3 | L | T5, T7 |
| K8 | "Adaptive stopping improves the trade-off, but does not solve selection." … "a stable schedule can still select a non-extrapolating computation." | §4.3 (Observation 3) | L | T5, T7 |
| K9 | "this is only a stabilising mechanism: it encourages consistency across nearby depths, but does not guarantee that the consistent computation is the extrapolating one." | §5.2 | L | T13, T7 |
| K10 | "The result should be read as a local mechanism for stabilisation, not as a guarantee that randomisation always improves extrapolation." | App. D (appendix) | L | T13 |
| K11 | "The assumption is intentionally local. It is not expected to hold for arbitrary loop depths or arbitrarily large windows." | App. D (appendix) | L | T13 |
| K12 | "The result is a sufficient-condition argument: it does not claim that low loss always identifies a unique hidden representation" | App. E (appendix) | L | T13 |
| K13 | "they do not directly establish hidden-state consistency" … the results "support, rather than uniquely identify, the proposed toy mechanisms." | App. C.3 (appendix) | L | T13, T10 |
| K14 | "A small oracle–policy gap does not by itself imply strong extrapolation: both accuracies may be low." | App. C.4 (appendix) | L | T10 |
| K15 | "Without entropy regularisation, the learned stopping distribution becomes nearly deterministic around a small loop depth, and oracle OOD accuracy is only 12.51%" (also: "stronger entropy regularisation is not uniformly beneficial") | App. C.8 (appendix) | L | T6 |
| K16 | "RL-Halting is not designed as an early-exit or compute-saving method." (also: "The hand-designed schedules can require fewer block evaluations") | App. A.1 (appendix) | L | T9 |
| K17 | "additional available depth alone does not ensure a better extrapolating computation." | App. C.7 (appendix) | L | T7, T5 |
| K18 | "stochastic schedules can stabilise suboptimal computations, RL-Halting underperforms on Copy, and the toy analyses explain stabilisation rather than guaranteeing extrapolation." | NeurIPS checklist Q2 | L | T5, T7, T13 |
| K19 | "We therefore limit our claims to the experimental setting studied and do not provide a broader societal-impact analysis." | NeurIPS checklist Q10 | L | T2 |

Sources: [arXiv abs](https://arxiv.org/abs/2606.29983), [arXiv PDF v2](https://arxiv.org/pdf/2606.29983v2), [arXiv HTML v2](https://arxiv.org/html/2606.29983v2)

---

## 5. A Mechanistic Analysis of Looped Reasoning Language Models (Blayney, Arroyo, Obando-Ceron, Castro, Courville, Bronstein, Dong)

- arXiv: 2604.11791. Title verified on [arXiv abs](https://arxiv.org/abs/2604.11791). The PDF running head shortens it to "A Mechanistic Analysis of Looped Language Models".
- Venue/date: preprint. v1 is dated 13 Apr 2026 and is the only version; the arXiv comment says "39 pages, 63 figures". It uses the ICML template, but no acceptance is stated and a web search found none.
- Version read: v1, both HTML and PDF. The PDF's two-column layout cuts the conclusion, so the HTML was used for §6.
- Sections used: §6 Conclusion, which has the sub-paragraphs "Implications of Findings" and "Limitations and Future Work", plus Appendices B, C.1 and C.2.
- Takeaway: the authors name two limitations, cyclic recurrence only and no analytic explanation of stable limiting behaviour. Appendix caveats add small scale and fragile behaviour-classification hyperparameters. The paragraph is headed "Limitations and Future Work" but makes no explicit future-direction statement. The only direction-like statement is an efficiency implication in "Implications of Findings".

| # | quote | section | L/F | tags |
|---|---|---|---|---|
| Bl1 | "We focus exclusively on cyclic recurrence" … "this means our analysis does not extend to sequential recurrence with multiple separate recurrent blocks" | §6 Lim. & FW | L | T16 |
| Bl2 | "we have not established analytically why this is the case, nor whether this stable limiting behavior is desirable or restrictive for reasoning tasks." | §6 Lim. & FW | L | T13, T7 |
| Bl3 | "predictable stages offer actionable pathways for efficient architectural design, including stage-dependent attention sparsification and the leaner parameterization of middle-stage MLPs" | §6 Implications of Findings | F | T9 |
| Bl4 | "this norm does not result in the stability issues reported by … but this is likely because we are operating at a far smaller scale." | App. B (appendix) | L | T1, T6 |
| Bl5 | "these exact results need to be treated with caution: the absolute values vary significantly depending on the algorithm hyperparameters chosen" (chosen "manually via visual inspection") | App. C.1 (appendix) | L | T10, T4 |
| Bl5F | "We leave in-depth classification and explanation of this phenomena to future work." (non-fixed-point orbits and sliders) | App. C.1 (appendix) | F | T13, T4 |
| Bl6 | "“Unknown” behavior often co-occurs with orbits, which we suggest may be due to mis-classification of the behavior algorithm." | App. C.2 (appendix) | L | T10, T4 |

Sources: [arXiv HTML v1](https://arxiv.org/html/2604.11791v1), [arXiv PDF v1](https://arxiv.org/pdf/2604.11791v1)

---

## 6. Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth (Viakhirev, Borodin, Almutairi, Barannikov, Abramov, Mkrtchian)

- arXiv: 2608.18222. Title verified on [arXiv abs](https://arxiv.org/abs/2608.18222).
- Venue/date: under review. v1 is dated 18 Aug 2026 and is the only version; the arXiv comment says "Submitted to the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-27)".
- Version read: v1, both HTML and PDF. The §8 text was checked against both.
- Sections used: §8 Limitations (a dedicated section with four labelled items), §9 Conclusion, §7.1, and Appendices G, H and N.
- Takeaway: limitations only, with **no future-work statements anywhere** (checked by searching for "future", "leave", "open"). They cover synthetic tasks and a single generator, tiny models, a repair shown on one checkpoint, seed-dependent formation of settling operators, causal mediation that is only partial, and extra depth that never converts into higher accuracy at 3.5B scale.

| # | quote | section | L/F | tags |
|---|---|---|---|---|
| TS1 | "Sudoku boards come from one construction’s symmetry orbit, so the headline is extrapolation on that generator" | §8 (Synthetic generators) | L | T2, T10 |
| TS2 | "producing a settling operator is not guaranteed (on mazes, a seed lottery)" | §8 (Controlled protocol) | L | T6, T10 |
| TS3 | "Models are tiny and tasks synthetic" | §8 (Small scale) | L | T1, T2 |
| TS4 | "the scale repair (Sec. 7.1) is operation-family-scoped, one checkpoint." | §8 (Small scale) | L | T1, T2, T10 |
| TS5 | "We identify the objective as the causal lever and the latent attractor as its needed target … not the regime as sole mediator; finite-time estimates bound adjectives, not regime separations" | §8 (Mediation) | L | T13 |
| TS6 | "Depth-safety is provable at small displacement (Prop. 1); conversion stays empirical and task-dependent." | §9 Conclusion | L | T13, T7 |
| TS7 | "The patch restores depth-safety, not new capability, on tasks never seen in training" | §7.1 (results) | L | T7 |
| TS8 | "Conversion never appears, and the depth-hard pointer-chase task stays unlearnable under every intervention tried" | §7.1 (results) | L | T7, T1 |
| TS9 | "No treated model exceeds the base’s peak competence at any depth, in any arm." | App. H (appendix) | L | T7, T1 |
| TS10 | "The fixed-point loss itself does not transfer to this scale: trained alongside the task it merely freezes an already-frozen loop" | App. H (appendix) | L | T1, T6 |
| TS11 | "Behavioral distillation from the base’s peak depth … never holds the deep tail on held-out domains" | App. H (appendix) | L | T2, T7 |
| TS12 | "domains the base itself cannot perform poison the mix." (the first of the "two failure laws" of the repair) | App. H (appendix) | L | T6 |
| TS13 | "The behavioral probes are small (304 prompts) … the base model is prompted zero-shot, so absolute accuracies are lower bounds." | App. G (appendix) | L | T10 |
| TS14 | "arm identity partially confounds, so it is a monitoring candidate, not a law; … (the checkpoint grid is too coarse)" | App. N, Formation caveats (appendix) | L | T10 |

Sources: [arXiv HTML v1](https://arxiv.org/html/2608.18222v1), [arXiv PDF v1](https://arxiv.org/pdf/2608.18222v1)

---

## 7. Step-resolved data attribution for looped transformers (Kaissis, Mildenberger, Gomez, Menten, Triantafillou)

- arXiv: 2602.10097. **Exact title: "Step-resolved data attribution for looped transformers"** ([arXiv abs](https://arxiv.org/abs/2602.10097)). The period-4 parity finding is confirmed in §4: "SDI exhibits a sawtooth-like oscillation with apparent period 4", with a "clear 4-state cycle in PCA space", in a looped transformer trained on parity following the Fan et al. (2024) setup.
- Venue/date: preprint (the PDF reads "Preprint. February 11, 2026."). v1 is dated 10 Feb 2026 and is the only version.
- Version read: v1, both HTML and PDF.
- Sections used: §5 "Discussion and future work", which has the paragraphs "Limitations" and "Future Work" (the Future Work paragraph opens with two limitation sentences), plus the §4 Nanochat case study.
- Takeaway: the limitations concern the attribution method itself: only TracIn-style faithfulness, step indices that are not comparable under adaptive or truncated loops, and cost. The directions are ANN-based scaling, RLHF and RLVR attribution, and turning influence signals into stopping criteria or regularisers.

| # | quote | section | L/F | tags |
|---|---|---|---|---|
| Ka1 | "our estimator should be viewed as a TracIn-style gradient-similarity influence computed along the training trajectory rather than a faithful approximation to an influence function" | §5 Limitations | L | T4, T13 |
| Ka1F | "A promising direction is to incorporate optimizer geometry more explicitly, e.g., by defining a preconditioned inner product." | §5 Limitations | F | T4, T13 |
| Ka2 | "When training uses randomized loop counts, adaptive halting, or truncated BPTT, step indices are not always comparable across examples, and truncation can systematically remove long-range credit assignment" | §5 Limitations | L | T4, T5, T6 |
| Ka3 | "scaling full-horizon SDI to very long recurrences remains expensive." | §5 Limitations | L | T6, T9 |
| Ka3F | "Addressing this may require improved systems techniques, e.g., more aggressive activation recomputation/checkpointing or hardware scaling." | §5 Limitations | F | T6, T9 |
| Ka4 | "it does not on its own certify that removing or reweighting an example will change behaviour in a specific way, nor does it disentangle confounds such as shared features" | §5 Future Work | L | T4 |
| Ka5 | "computing dense train × test influence matrices is still costly at very large \|D_train\|." | §5 Future Work | L | T9 |
| Ka5F | "treat the sketched per-example (and per-step) vectors as an indexable embedding space … enabling scalable workflows such as depth-targeted data curation and debugging." | §5 Future Work | F | T9, T4 |
| Ka6 | "Another promising direction is extending SDI beyond supervised next-token losses to modern alignment pipelines." (RLHF-style preference optimization) | §5 Future Work | F | T12, T4 |
| Ka7 | "in RL settings with verifiable rewards, SDI could help localize which trajectories and reward signals drive improvements at specific recurrent steps and diagnose step-local reward hacking." | §5 Future Work | F | T12, T4 |
| Ka8 | "step-energy curves and influence horizons could be turned into instance-wise stopping criteria or training-time regularisers that encourage useful computation to persist deeper into the loop" | §5 Future Work | F | T5, T7, T12 |
| Ka9 | "We regard a deeper investigation of this phenomenon as a promising direction for future work." (an implicit "loop step counter", with GSM8K performance "plateauing for all test-time τ>4") | §4 Nanochat (results) | F | T7, T4 |

Sources: [arXiv HTML v1](https://arxiv.org/html/2602.10097v1), [arXiv PDF v1](https://arxiv.org/pdf/2602.10097v1)

---

## Group tally (7 papers)

Each paper is counted at most once per tag in each column. Short names: Saunshi25, Fan25, Bansal22, Kuo26, Blayney26, ThinkShallow26 (Viakhirev et al.), Kaissis26.

| tag | papers stating it as L | papers stating it as F | papers stating it at all | paper short names (L / F) |
|---|---|---|---|---|
| T1 model-scale | 4 | 0 | 4 | L: Fan25, Kuo26, Blayney26, ThinkShallow26 / F: none |
| T2 task-breadth | 5 | 4 | 5 | L: Saunshi25, Fan25, Bansal22, Kuo26, ThinkShallow26 / F: Saunshi25, Fan25, Bansal22, Kuo26 |
| T3 gap-to-explicit-CoT | 0 | 0 | 0 | none |
| T4 interpretability | 2 | 2 | 2 | L: Blayney26, Kaissis26 / F: Blayney26, Kaissis26 |
| T5 fixed-vs-adaptive-compute | 3 | 4 | 5 | L: Fan25, Kuo26, Kaissis26 / F: Saunshi25, Bansal22, Kuo26, Kaissis26 |
| T6 training-cost-or-stability | 6 | 3 | 6 | L: Fan25, Bansal22, Kuo26, Blayney26, ThinkShallow26, Kaissis26 / F: Fan25, Bansal22, Kaissis26 |
| T7 extrapolation-past-trained-depth | 5 | 4 | 6 | L: Fan25, Bansal22, Kuo26, Blayney26, ThinkShallow26 / F: Fan25, Bansal22, Kuo26, Kaissis26 |
| T8 supervision-dependence | 1 | 0 | 1 | L: Fan25 / F: none |
| T9 efficiency-claims | 2 | 2 | 3 | L: Kuo26, Kaissis26 / F: Blayney26, Kaissis26 |
| T10 evaluation-rigor | 3 | 0 | 3 | L: Kuo26, Blayney26, ThinkShallow26 / F: none |
| T11 test-time-scaling-and-verification | 0 | 1 | 1 | L: none / F: Saunshi25 |
| T12 RL-or-new-objectives | 0 | 3 | 3 | L: none / F: Bansal22, Kuo26, Kaissis26 |
| T13 theory-gap | 6 | 4 | 6 | L: Saunshi25, Fan25, Kuo26, Blayney26, ThinkShallow26, Kaissis26 / F: Saunshi25, Fan25, Blayney26, Kaissis26 |
| T14 memory-or-attention-over-steps | 0 | 0 | 0 | none |
| T15 hybrid-or-tool-integration | 0 | 1 | 1 | L: none / F: Saunshi25 (multimodal reasoning) |
| T16 architecture-generality | 2 | 2 | 3 | L: Fan25, Blayney26 / F: Saunshi25, Fan25 |
| NEW: memorization-perplexity-tradeoff (looping hurts perplexity or memorization while helping reasoning) | 1 | 0 | 1 | L: Saunshi25 |

Ranking within this group by "stated at all": T6, T7 and T13 are tied at 6 of 7. T2 and T5 follow at 5, then T1 at 4, then T9, T10, T12 and T16 at 3, then T4 at 2, and T8, T11, T15 and NEW at 1. T3 and T14 are 0.

Sensitivity check for the merge: the strict count. If another group extracted only from Limitations, Future Work, Conclusion or Discussion text and main-text "we leave X for future work" sentences, it would drop every row marked "(results)", "(appendix)" or "(checklist)". Under that rule this group's "stated at all" counts become:

| tag | strict count | papers (strict) | change from full count |
|---|---|---|---|
| T13 theory-gap | 6 | Saunshi25, Fan25, Kuo26, Blayney26, ThinkShallow26, Kaissis26 | 0 |
| T2 task-breadth | 5 | Saunshi25, Fan25, Bansal22, Kuo26, ThinkShallow26 | 0 |
| T7 extrapolation-past-trained-depth | 5 | Fan25, Kuo26, Blayney26, ThinkShallow26, Kaissis26 | −1 (Bansal22 is appendix-only) |
| T1 model-scale | 3 | Fan25, Kuo26, ThinkShallow26 | −1 (Blayney26 is appendix-only) |
| T5 fixed-vs-adaptive-compute | 3 | Saunshi25, Kuo26, Kaissis26 | −2 (Fan25 only in results and appendix; Bansal22 appendix-only) |
| T6 training-cost-or-stability | 3 | Fan25, ThinkShallow26, Kaissis26 | −3 (Bansal22, Kuo26, Blayney26 appendix-only) |
| T16 architecture-generality | 3 | Saunshi25, Fan25, Blayney26 | 0 |
| T9 efficiency-claims | 2 | Blayney26, Kaissis26 | −1 (Kuo26 appendix-only) |
| T12 RL-or-new-objectives | 2 | Kuo26, Kaissis26 | −1 (Bansal22 appendix-only) |
| T4 interpretability | 1 | Kaissis26 | −1 (Blayney26 appendix-only) |
| T10 evaluation-rigor | 1 | ThinkShallow26 | −2 (Kuo26, Blayney26 appendix-only) |
| T8 supervision-dependence | 1 | Fan25 | 0 |
| T11 test-time-scaling | 1 | Saunshi25 | 0 |
| T15 hybrid-or-tool-integration | 1 | Saunshi25 | 0 |
| NEW: memorization-perplexity-tradeoff | 0 | none (Saunshi25 is results-only) | −1 |
| T3, T14 | 0 | none | 0 |

---

## Papers whose full text could not be accessed

- None. All seven were read in full from arXiv HTML and PDF, latest versions: 2502.17416v1, 2409.15647v5, 2202.05826v3, 2606.29983v2, 2604.11791v1, 2608.18222v1, 2602.10097v1.
- Minor gaps:
  - Bansal: the NeurIPS 2022 supplemental PDF did not download (the link returned HTML). The arXiv v3 appendix was used instead, and the camera-ready main-text Conclusion was confirmed identical to v3.
  - Kuo: no separate NeurIPS 2026 proceedings or OpenReview copy was checked. arXiv v2 carries the "Published at NeurIPS 2026" comment and includes the NeurIPS checklist.
  - Blayney: no venue could be confirmed by web search.

---

## Observations, not counted (things the authors did not state as limitations or directions about their own work)

- **ThinkShallow26 states no future directions at all.** Its §8 is limitations only, and its Conclusion only restates results.
- **Blayney26's "Limitations and Future Work" paragraph has no explicit future-direction sentence.** The analytic gap (Bl2) implies one but I recorded it as L only. A writer who counts heading-implied directions would add an F for T13 and T16 for Blayney26.
- **Kuo26 K1 ("remains to be established")** was recorded as L only. The F rows for Kuo26 come from the sentences that propose directions explicitly.
- **Saunshi25 diminishing returns.** §3.4 says "the returns are diminishing with depth for both looped and non-looped models". This is reported as a scaling finding, not a limitation, so it is not counted (it would be T7).
- **Bansal22 model size.** App. A.4 says depth and width of the maze and chess models were kept "only as large as necessary to avoid excessive memory and computational costs". This is a design-choice statement and not counted (it would be T1/T6).
- **Bansal22 run-time penalty.** Its preliminary results ("worse generalization") concern an alternative loss suggested by a reviewer, not the authors' method. Only the future-work sentence (B2) is counted.
- **Kaissis26, statement about the whole field.** App. A says "the opacity of latent space in latent reasoning models remains a significant hurdle". This is motivation for the work, not a limitation of it, so it is not counted (it would be T4).
- **Kaissis26 GSM8K plateau.** "performance on GSM8K plateauing for all test-time τ>4" is a finding offered as evidence for the authors' hypothesis. It is counted only through the future-work sentence Ka9.
- **ThinkShallow26 citing prior work.** "recurrence 4→32 yields marginal gains far below chain-of-thought" (§7.1) cites prior work on Huginn. It is not counted, but it is the only near-T3 statement in the group.
- **Blayney26 citing prior work.** It notes that extrapolation beyond training recurrences "harms the performance of Ouro" (Zhu et al.). Also prior work, not counted. Its Huginn-0125 "negative result" (no stages of inference) is a finding about a studied model, not counted.
- **Fan25 vs explicit CoT.** Fan25's "we still require fewer assumptions than CoT training" frames CoT as more demanding. The authors do not concede an accuracy gap to CoT.
- **Pattern in this group.** T3 (gap to explicit CoT) and T14 (memory/attention across steps) never appear. These papers mostly study synthetic algorithmic tasks or analyse existing models, and rarely compare against explicit CoT. The group's distinctive concerns are T7 (stability does not imply extrapolation, overthinking, extra depth never converting into accuracy) and T13 (toy, local or sufficient-condition theory only).
- **The same idea recurs across three papers.** Bansal22 says fixed-point convergence removes the need for stopping rules. ThinkShallow26 says settling operators are depth-safe but conversion is empirical and task-dependent. Kuo26 says stabilisation does not guarantee extrapolation. All three say, in different words, that convergence or stability of the loop is not the same as better answers at more depth.
- **Format.** The notes follow the per-paper and tally format this assignment asked for, not the generic three-section research template.
