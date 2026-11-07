# Q4: prior work that keeps or merges multiple latent candidates (latent / continuous-CoT models)

Working notes, 2026-11-06. Web research only. Scope: Coconut/CODI/SIM-CoT-style latent reasoners, plus close
relatives (Huginn-style recurrent depth, soft-thought prefixes, latent diffusion). Token-level Soft Thinking /
Multiplex / CoT2 were left to another researcher.

Read-level tags:
- **[full text]**: I read the paper body (and appendix where noted). For LatentTTS I read the ACL camera-ready PDF
  directly (pdftotext plus rendered figure pages). For the others, "full text" means the fetch tool processed the
  arXiv HTML and I asked targeted questions; numbers came through that extraction and were not re-checked
  line by line against a PDF.
- **[abstract only]**: arXiv abstract page or search snippet.
- **[unverified]**: seen only in search results or reference lists.
- "Read off figure" = my reading of plotted points; approximate (about ±0.2 points).

---

## 1. LatentTTS: Parallel Test-Time Scaling for Latent Reasoning Models [full text]

- You, Li, Liu, Wang, Nie, Li (PolyU et al.). arXiv 2510.07745 (v4, 19 Apr 2026); ACL 2026 main,
  pp. 44703 ff. ([ACL PDF](https://aclanthology.org/2026.acl-long.2069.pdf), [arXiv](https://arxiv.org/abs/2510.07745),
  [code](https://github.com/ModalityDance/LatentTTS)).
- Models: COCONUT GPT-2 and CODI GPT-2 (T = 6 latents), CoLaR Llama-3.2-1B; plus Latent-SFT (Llama-3.2-1B) and
  Render-of-Thought 2B/4B for coverage only.

### Sampling
- MC dropout (dropout after the FFN in each block, kept on at inference; Algorithm 1 samples one mask per latent
  step) or additive Gaussian noise (AGN) on each latent.
- p and σ found by binary search to maximise coverage@64. Heuristic bands: GPT-2-style p ∈ [0.1, 0.3],
  σ ∈ [0.5, 0.7]; Llama-1B-style p, σ ∈ [0.01, 0.03]. Fig. 2 legend: CODI used σ = 1.4, p = 0.2; Coconut σ = 0.6,
  p = 0.2; CoLaR σ = 0.01, p = 0.03.
- Aggregation experiments use MC dropout p = 0.2.

### LatentRM
- GPT-2 initialised **from the COCONUT checkpoint**, with a scalar head.
- Data: GSM8K train; N = 8 dropout trajectories per problem.
- Label per thought: success rate over M = 128 stochastic rollouts. Problems where all trajectories are right, or
  all wrong, are excluded.
- Loss: step-wise softmax over the N concurrent thoughts at each step (contrastive NLL). BCE "performs poorly".
- Released as one RM checkpoint (`ModalityDance/latent-tts-rm`, base GPT-2). README shows only `train_coconut.yaml`.
  **I found no CODI RM.**

### Aggregation procedures (Appendix D, verbatim mechanics)
- **Best-of-N:** R(n) = Σ_t r_t(n). Pick the argmax trajectory, then decode its answer.
- **Beam search:** at step t, each partial trajectory in beam B_{t−1} is extended by one step into K stochastic
  candidates. LatentRM scores each candidate with r_t; the cumulative reward is R_t(k) = R_{t−1}(b) + r_t(k), and
  the top-B by cumulative reward are kept. It stops when all beams emit end-of-thinking or reach T, and the final
  pick is by cumulative reward.
  - Settings: B = √N and K = B, so total forward passes match best-of-N. Rewards are pre-normalised by trajectory
    length.
  - The README script sweeps beam sizes 1, 2, 4, 8 (N = 1, 4, 16, 64).
  - Beam search never keeps fewer than B partial chains. There is **no merge**; the final aggregation is
    argmax, not a vote.
- **Majority voting:** the mode of the N decoded answers.

### Which model the aggregation runs on
- **COCONUT only.** The paper never says so outright, but Fig. 6 shows a single model per benchmark and the RM is
  Coconut-initialised.
- **CODI appears only in the coverage plots (Figs. 2 and 3).**

### Numbers (COCONUT GPT-2, MC dropout p = 0.2; mean of 3 runs)
- Table 3 (GSM-Test / GSM-Hard):

  | Setting | GSM-Test | GSM-Hard |
  |---|---|---|
  | deterministic | 34.0 | 7.0 |
  | Coverage@4 | 46.7 | 9.5 |
  | Best-of-4 | 34.5 | 7.5 |

  For comparison, explicit CoT is 42.9 / 9.8 deterministic and 50.7 / 11.6 at Coverage@4.
- Table 2 (best-of-8 with LatentRM; the text says N = 8, the caption says N = 32, and Fig. 6 matches N = 8):

  | Variant | GSM-Test | GSM-Hard |
  |---|---|---|
  | Best-of-8, LatentRM | 35.4 | 7.8 |
  | w/o contrastive (BCE) | 33.5 | 7.4 |
  | w/o stochastic rollouts | 30.7 | 6.0 |
  | random head | 28.9 | 5.8 |
  | Majority voting | 33.6 | 6.1 |

- Fig. 6, read off figure. Accuracy vs N = 1…64 (x-axis 2^0 to 2^6). All curves start at the *sampled* N = 1
  point, about 31.3 on GSM-Test, which is below the deterministic 34.0. Beam points exist only at N = 1, 4, 16, 64.

  | Benchmark | Method | N=2 | N=4 | N=8 | N=16 | N=32 | N=64 |
  |---|---|---|---|---|---|---|---|
  | GSM-Test | majority | 31.8 | 33.3 | 33.6 | 33.5 | 33.9 | 34.2 |
  | GSM-Test | best-of-N | 32.7 | 33.6 | 35.4 | 35.3 | 34.9 | 35.5 |
  | GSM-Test | beam (B = 2/4/8) | | 34.4 | | 34.7 | | 35.5 |
  | GSM-Hard (start ≈7.05) | majority | 6.7 | 7.0 | 7.3 | 7.35 | 7.45 | 7.6 |
  | GSM-Hard | best-of-N | 7.4 | 7.5 | 7.85 | 8.0 | 8.35 | 8.2 |
  | GSM-Hard | beam | | 7.35 | | 7.65 | | 7.75 |
  | MultiArith (start ≈73.5) | majority | 74.3 | 78.8 | 79.8 | 80.7 | 80.8 | 81.0 |
  | MultiArith | best-of-N | 77.1 | 79.0 | 81.5 | 81.5 | 82.1 | 81.9 |
  | MultiArith | beam | | 81.4 | | 81.7 | | 82.3 |

  - Inconsistency: Table 3's Best-of-4 is 34.5, but the Fig. 6 point at N = 4 is about 33.6. They are probably
    different runs; the paper doesn't say.
  - My reading: best-of-64 beats deterministic Coconut by only about +1.5 on GSM-Test. Majority at N = 64 (about
    34.2) roughly equals the deterministic 34.0. Beam beats best-of-N at N = 4 on GSM-Test and MultiArith, but not
    on GSM-Hard.
- Authors' text: accuracy "increases monotonically with N"; best-of-N and beam "consistently outperform Majority
  Voting"; beam is "comparable with Best-of-N on GSM-Test and MultiArith but trails on GSM-Hard, suggesting that
  early-step score noise can cause premature pruning". They call best-of-N "the most effective route".
- Table 5, variable-length Coconut (4–6 steps, mean step reward), GSM-Test:

  | Setting | Accuracy |
  |---|---|
  | deterministic | 32.44 |
  | BoN-4 | 33.81 |
  | BoN-8 | 33.95 |
  | BoN-16 | 34.04 |

### CODI specifics (coverage only)
- Fig. 2a, read off figure: CODI deterministic about 42.5 on GSM-Test. MC dropout (p = 0.2) coverage is about 37
  at N = 1 and about 63.5 at N = 64. AGN (σ = 1.4) reaches about 55.5 at N = 64.
- Text: "at N=64, COCONUT and CODI achieve nearly equivalent coverage, despite CODI's clear superiority at N=1."
- Fig. 3 (CODI, coverage vs diversity): with MC dropout, CODI's coverage peaks at very low diversity (about 0.03–0.05)
  and collapses to about 0.1–0.25 at diversity above 0.15. AGN is flatter.
  - My inference: CODI's dropout window is narrow. This matters for SIM-CoT-on-CODI candidates.

### Other
- Wall-clock: Coconut N = 32 takes 1.708 s per question on an H100; aggregation is about 20% of that.
- Limitation section: real-time engineering, and sensitivity to p and σ.

---

## 2. LaDi-RL: Latent Diffusion Reasoning Prevents Entropy Collapse in RL [full text]

- Kang, Zhang, Kuang, Ma, Qin (UCSD, Apple). arXiv 2602.01705 v3, 17 May 2026. No venue stated.
- **Not a Coconut-style stepwise latent.** A diffusion policy denoises a fixed latent *block* (B = 64 latent tokens,
  cold-started from LaDiR's VAE). The LLM then writes the answer text conditioned on the block.
- Base models: DeepSeek-R1-Distill-Qwen-7B for math; Qwen3-8B-Base for code.

### Multi-candidate mechanism (training only)
- A hierarchical rollout: N latent trajectories per query, then M text answers per latent.
- The latent advantage standardises the N per-latent mean rewards; each latent's reward is the mean of its M
  answers. The text advantage is computed within the M answers of each latent.
- Loss: α·L_latent + (1−α)·L_text, best at α = 0.90.
- Flow-GRPO-style SDE denoising, plus a repulsion ("diversity guidance") term that pushes nearby latents apart
  during denoising. Removing it costs −3.1 on the code average.
- Hyperparameters: Table 6 gives N = 16, M = 5 for training. The hierarchy ablation uses N = 4, M = 4.

### Rollout ablation (code average)

| Method | Rollouts (N×M) | Code avg |
|---|---|---|
| GRPO | 1×1 | 65.5 |
| GRPO | 4×4 | 67.7 |
| LaDi-RL | 1×1 | 60.0 |
| LaDi-RL | N=4, M=1 | 68.5 |
| LaDi-RL | N=1, M=4 | 63.5 |
| LaDi-RL | N=4, M=4 | 77.1 |

### Inference
- **No aggregation.** Evaluation draws 16 samples per problem and reports pass@1 and pass@k.
  - Math pass@1 average: 43.45, vs 37.73 for vanilla GRPO and 41.55 for Multiplex Thinking.
  - Code average: 77.08 vs 67.68.
  - Code pass@128: 86.1 vs 73.2 for GRPO and 76.4 for the base model.
- **Takeaway:** multi-latent × multi-answer is used for *credit assignment*, not for test-time keep/merge.

## 3. SoftCoT++ [full text]

- Xu et al., arXiv 2505.11484 v2 (27 May 2025). No venue found.
- A frozen small assistant model produces soft thoughts, which are projected into a frozen backbone (LLaMA-3.1-8B-I,
  Qwen3-8B). The backbone then reasons in **discrete tokens**.
- Multiple soft thoughts:
  - N distinct [INI] initial-token sets (one per path), so the soft-thought prefixes differ.
  - A contrastive loss pushes the soft thoughts apart. Eq. 6 as printed looks like it has a typo.
  - Only the projection is trained. L = 4 soft tokens.
- Aggregation: majority vote (self-consistency) over the chains. **There is no merge of soft thoughts.**
- Table 1 (N = 10, mean over 5 seeds):

  | Backbone | Method | GSM8K | ASDiv-Aug | AQuA | StrategyQA | DU | Avg |
  |---|---|---|---|---|---|---|---|
  | LLaMA | token SC | 90.36 | 89.23 | 63.23 | 70.13 | 65.76 | 75.74 |
  | LLaMA | SoftCoT-SC | 90.63 | 89.75 | 65.51 | 71.14 | 67.36 | 76.88 |
  | LLaMA | SoftCoT++ | 90.99 | 90.09 | 66.85 | 71.18 | 68.72 | 77.57 |
  | Qwen3 | token SC | 92.22 | 91.97 | 76.77 | 70.96 | 84.56 | 83.30 |
  | Qwen3 | SoftCoT-SC | 93.19 | 92.14 | 80.63 | 71.18 | 87.20 | 84.87 |
  | Qwen3 | SoftCoT++ | 93.65 | 92.41 | 84.09 | 71.22 | 88.16 | 85.91 |

  Coconut-SC, GSM8K: 87.03 (LLaMA) and 90.37 (Qwen3).
- Scaling to N = 100 (10 soft thoughts × 10 chains), GSM8K:

  | Backbone | SoftCoT++ | SoftCoT-SC | token SC |
  |---|---|---|---|
  | LLaMA | 92.71 | 92.52 | 92.42 |
  | Qwen3 | 94.12 | 93.40 | 92.98 |

- Contrastive-loss ablation (GSM8K only), SoftCoT+ (no contrastive loss) vs SoftCoT++:

  | Backbone | N | SoftCoT+ | SoftCoT++ |
  |---|---|---|---|
  | LLaMA | 10 | 90.67 | 90.99 |
  | LLaMA | 100 | 92.63 | 92.71 |
  | Qwen3 | 10 | 93.28 | 93.65 |
  | Qwen3 | 100 | 93.97 | 94.12 |

  These are gaps of about 0.1–0.4 points.
- **Relevance:** diversity is injected only into a static soft prefix. The reasoning itself is discrete. It is
  not a stepwise latent chain.

## 4. ITS-Continuous: Towards Inference-time Scaling for Continuous Space Reasoning [full text]

- M. Wang, Vu, Shareghi, Haffari (Monash). arXiv 2510.12167 v2. Findings of ACL 2026 according to the earlier
  project notes, which used the camera-ready.
- Model: COCONUT GPT-2 (GSM8K, c = 2, T = 6). Dropout is on only during the latent phase, at the training rate (the
  value isn't stated). N ∈ {1, …, 32}.
- Pass@N:

  | N | 1 | 2 | 4 | 8 | 16 | 32 |
  |---|---|---|---|---|---|---|
  | Pass@N | 31.08 | 35.10 | 38.67 | 41.02 | 42.61 | 44.43 |
  | Unique answers | 1.00 | 1.33 | 1.72 | 2.16 | 2.62 | 3.17 |

- Reranking (Table 1, accuracy):

  | N | Confidence | Self-consistency | PRM-HE | PRM-SE | ORM |
  |---|---|---|---|---|---|
  | 2 | 31.08 | 31.01 | 31.69 | 30.86 | 31.61 |
  | 4 | 30.48 | 31.61 | 32.45 | 32.37 | 32.15 |
  | 8 | 29.87 | 31.24 | 33.06 | 32.52 | 31.46 |
  | 16 | 31.39 | 32.15 | **33.36** | 32.37 | 32.37 |
  | 32 | 30.71 | 32.15 | 32.83 | 33.28 | 31.39 |

  - The best result is +2.3 over the deterministic 31.08, against a Pass@16 of 42.61.
  - Step-score aggregation (min, max, mean, last) makes negligible difference.
- Classifier quality on 3,014 test samples:

  | Model | Accuracy | F1 | Precision |
  |---|---|---|---|
  | PRM | 62.98 | 54.09 | 41.60 |
  | ORM | 73.72 | 51.59 | 39.11 |

- Geometry, correct vs incorrect:

  | Measure | Correct | Incorrect | Note |
  |---|---|---|---|
  | IsoScore* | 0.0134 | 0.013 | |
  | Hoyer | 0.21 | 0.22 | |
  | Compactness, all | 19.81 | 19.39 | p = 0.023, d = 0.17 |
  | Compactness, PRM+ subset | 20.72 | 18.55 | d = 1.14 |

  Conclusion: correct and incorrect thoughts "lack the geometric separability" a scorer needs.
- Noise replacement: Pass@5 is 39.20 at noise ratio 0 and 12.59 at ratio 1.0. The model sometimes answers without
  using the latents.
- Suggested remedies (untested): isotropy objectives, trajectory diversity, contrastive training.

## 5. NoisyCoconut [full text]

- Jerge and Evans (UVA). arXiv 2605.08221 v1, 6 May 2026. No venue.
- **Not a trained latent reasoner.** It runs a Coconut-style hidden-state feedback loop, at inference, on
  off-the-shelf LLMs: Qwen2.5-7B, Llama-3.1-8B, Mixtral, gpt-oss-20B, DeepSeek-R1-Distill-Qwen-7B.
- Noise: h_{t+1} = f(h_t + η_t), σ_t = σ0·e^{−λt}, scaled by norm; σ0 = 0.2. K = 5 paths, at most 8 latent steps.
- Aggregation:
  - It answers only if more than K/2 paths agree; otherwise it abstains. Unanimity (5/5) is the high-precision mode.
  - GSM8K at 5/5 agreement:

    | Model | Baseline | Unanimous accuracy | Coverage |
    |---|---|---|---|
    | Qwen2.5-7B | 71.00 | 91.25 | 51.4% |
    | Llama-3.1-8B | 63.38 | 93.57 | 31.1% |
    | gpt-oss-20B | 73.50 | 95.11 | 53.2% |

- Ablation (Qwen, GSM8K):

  | Setting | Accuracy |
  |---|---|
  | baseline | 71.0 |
  | K = 5, ≥3/5 agree | 78.2 |
  | σ0 = 0 | 76.4 |
  | K = 1 | 72.0 |
  | K = 10, ≥6/10 agree | 80.2 |

  The σ0 = 0 row means most of the gain does not come from latent noise.
- No matched-compute token self-consistency control. The paper itself calls learned or confidence-weighted
  aggregation future work.

## 6. Other multi-sample / multi-candidate work found

### Selection or aggregation at inference over latent candidates

- **LTO: Latent Thinking Optimization** (Du, Dong, Ning, OSU). arXiv 2509.26314 v3; ICLR 2026 per the PDF header in
  search results. [full text]
  - Huginn-3.5B. N = 20 trajectories per problem, drawn by different *initial-noise seeds* of the recurrent state;
    T = 32.
  - Latent reward model: a 2-layer transformer classifier over mean-pooled latent states, trained with BCE on 5
    samples per training problem.

    | Dataset | Accuracy | ROC-AUC |
    |---|---|---|
    | GSM8K | 0.820 | 0.884 |
    | SVAMP | 0.960 | 0.987 |
    | CSQA | 0.623 | 0.671 |
    | MBPP | 0.790 | 0.807 |

  - Selection is acceptance-rejection with φ = exp((r − r_max)/β); β → 0 gives best-of-N. One answer is chosen
    (M = 1).
  - GSM8K: base 0.326, majority 0.333, LRM-weighted majority 0.375, LTO 0.385. The other benchmarks gain
    +0.017 to +0.059.
  - "Performance steadily improves as N increases" (figure only).
  - **Selection is over whole trajectories only.** The LRM can score prefixes, but the paper doesn't use per-step
    selection.
  - Contrast with ITS-Continuous: separability on Huginn is high, while on Coconut GPT-2 it is low.

- **GTS: Learnable Gaussian Thought Sampler** (M. Wang, Bai, Vu, Shareghi, Haffari). arXiv 2602.14077 v2, 18 Mar
  2026. [full text]
  - A small head predicts a context-dependent diagonal Gaussian perturbation of each latent: h̃ = h_det + z.
  - Trained with GRPO (32 rollouts per prompt, KL to an EMA reference, reward ±r0 plus a log-prob shaping term) with
    the **backbone frozen**.
  - Models: Coconut GPT-2 and CODI Llama-3.2-1B.
  - Metric: pass@N only (figure). No voting, BoN, beam, or merge. Small-N (N = 2) accuracy can dip.
  - This is the closest "RL-trained component, frozen latent generator" design, but it is an RL-trained
    **proposer**, not a selector.

- **C-voting** (Kubo et al.). arXiv 2604.13521; ICLR 2026 per the ICLR virtual-site listing. [abstract only]
  - Recurrent models start from several random latent initialisations; the trajectory with the highest mean top-1
    confidence is kept.
  - Results: Sudoku-hard +4.9 over energy voting; ItrSA++ with C-voting gets 95.2 on Sudoku-extreme vs 55.0 for HRM.
  - Puzzles, not language reasoning.

- **Latent Thought Flow (LTF)** (Zou et al.). arXiv 2606.16222 v1. [full text]
  - A continuous GFlowNet sampler over variable-length latent trajectories, on Llama 1B/3B/8B and R1-Qwen-1.5B.
  - Table 9 (1B): average accuracy 59.68 at N = 1, 61.26 at N = 5, 62.13 at N = 10. **The aggregation rule is not
    stated.**
  - Mentions a "tree of latent thoughts" for larger budgets but gives no details.

- **ATF: Next Thoughts Are Distributions** (Li et al.). arXiv 2609.33271 v1. [full text, appendix F not read]
  - A flow head samples each next continuous thought; Qwen3 and Llama backbones.
  - Reports Pass@1 and Pass@100: for example GSM8K, Llama-8B, 84.6 / 97.6 at text T = 0.7.
  - Self-consistency at K = 16 appears in Appendix F only; numbers not read. No per-step branching or merge.

### Multi-sample analyses (CODI / SIM-CoT GPT-2 coverage-vs-vote gap)

- **How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision?** (Cui et al., Amazon/MSU). arXiv
  2602.22441 v1, 25 Feb 2026. [full text]
  - Latents are deterministic; "stochastically sample the final answer 100 times" (temperature not stated).
  - GPT-2 GSM8K (greedy / Pass@100 / Maj@100):

    | Model | Greedy | Pass@100 | Maj@100 |
    |---|---|---|---|
    | CODI | 43.59 | 70.43 | 42.23 |
    | SIM-CoT | 42.23 | 69.60 | 43.21 |
    | Coconut | 34.09 | 69.07 | 39.42 |
    | CoLaR | 18.44 | 23.28 | 18.42 |

  - Hybrid Coconut rollouts (latent prefix, then text at T = 1): latent-prefix Pass@100 is more than 20 points
    above explicit, but Maj@100 is 3–4 points lower at longer prefixes.
  - Quote: "increased outcome-level diversity alone is insufficient to guarantee effective aggregation."
  - The number of distinct candidates peaks early and then shrinks: implicit pruning, "not a strict BFS".

- **PLaT: Latent CoT as Planning** (J. Wang et al.). arXiv 2601.21358 v3. [full text]
  - The planner's latents are deterministic; branching is in the text decoder (T = 0.9, 128 samples).
  - Pass@k only. The paper says "realizing this coverage as final-answer accuracy requires answer selection" and
    does not do selection.
  - GPT-2 GSM8K Pass@1 / @32 / @128:

    | Model | @1 | @32 | @128 |
    |---|---|---|---|
    | CODI | 40.16 | 63.00 | 69.90 |
    | Coconut | 32.94 | 55.69 | 66.19 |
    | PLaT | 18.35 | 59.51 | 76.08 |

- **SLPO** (You, Liu, Li, Li; same group as LatentTTS). arXiv 2607.19691 v3. [full text]
  - RLOO with G = 8 MC-dropout (p = 0.1) trajectories. K = 4 dropout passes per prefix give a Gaussian surrogate
    (mean and variance) used **only for the likelihood**.
  - Inference is deterministic. Pass@k reports oracle coverage only:

    | Model, GSM8K | Base P@8 / P@16 | +SLPO P@8 / P@16 |
    |---|---|---|
    | CODI GPT-2 | 52.08 / 54.36 | 54.13 / 56.71 |
    | Coconut GPT-2 | 45.79 / 49.13 | 49.13 / 51.55 |

  - No voting, BoN, or LatentRM; it does not combine with LatentTTS.

### Generator-side RL with multiple latent rollouts (no test-time keep or merge)

- **LTC: Latent Thought Credit** (Zhao et al., Nankai). arXiv 2608.01593 v1, 3 Aug 2026. [full text]
  - K Gumbel-softmax latent thoughts per prompt; M answers per frozen post-thought context. Thought value = mean of
    its M rewards, giving a two-level advantage.
  - Qwen2.5-3B/7B. GSM8K 3B: LTC 84.46 vs GRPO 82.71 vs HRPO 83.32.
  - Budget allocation at B = 16, GSM8K:

    | K×M | 2×8 | 4×4 | 8×2 |
    |---|---|---|---|
    | GSM8K | 84.52 | 85.15 | 83.10 |

  - Inference is greedy.
  - Diagnostic: within-thought variance is larger than between-thought variance (noise-to-signal 1.54 at the start,
    13.10 at the end), so one answer per thought gives a noisy thought value.

- **Dropout-GRPO** (Jung, UCSD). arXiv 2606.10184 v1. [full text]
  - One dropout mask per rollout, held across all latent steps. K = 32.
  - Coconut on Qwen2.5-1.5B, GSM8K pass@1: 27.29 → 29.01.
  - Inference is deterministic. The authors note a train/deploy mismatch: the mask-marginal policy is trained,
    but the deterministic policy is deployed.

- **CoLaR** [full text]: Gaussian latent head with GRPO G = 8. No pass@k or voting. The deterministic-head ablation is
  about equal: average 41.7 vs 41.3.

- **HRPO** [full text]: NeurIPS 2025. Gated token/hidden mixing; g = 4 or 8. Greedy evaluation, no aggregation.

- **Latent-GRPO** (Deng et al.). arXiv 2604.27998. [full text]
  - Its "optimal correct-path first-token selection" is a *training* gradient mask over correct rollouts, not
    inference selection.
  - Pass@64 uses Gumbel noise at inference.

- **SIM-CoT** [full text]: no multi-sample use. The step decoder is removed at inference and used only for
  visualisation.

- **LatentSeek** [full text, first half]: test-time REINFORCE on per-token latents of a *text* CoT. One greedy decode
  per iteration; no keep-several.

- **LTPO** (ICLR 2026) [full text, first 100k chars]:
  - Optimises prompt-side [THINK] embeddings with one Gaussian perturbation per step.
  - Keeps the *best-reward* latents across iterations, a sequential keep-best, not parallel.
  - No BoN/SC comparison.

- **RLTT** (2602.10520, ICML 2026 per comments), **ATP-Latent** (2601.21598), **SWITCH** (2606.13106, Findings
  EMNLP 2026): [abstract only]. Generator-side RL; nothing on keep or merge visible.

### Variational / multi-trajectory latent models (no aggregation)

- **PLRM** (Lee, Lee, Lee, Korea Univ.). arXiv 2610.29385 v2. [full text, first 100k chars]
  - A mixture prior over strategies; the strategy is sampled once and fixed for the whole trajectory.
  - GPT-2: GSM8K 38.40 vs CODI 41.17 and SIM-CoT 38.97; MultiArith 88.83.
  - No voting, selection, or merge.

### Merge-related theory or analysis

- **Reasoning by Superposition** (Zhu, Hao, …, Tian), NeurIPS 2025. [abstract only]
  - A continuous thought can encode multiple search frontiers at once (graph reachability, 2-layer construction).
  - This motivates "merge" as a natural operation, but it is theory on a synthetic task.

- **Continuous Mixture Collapse** (Backour). arXiv 2609.02049. [abstract only]
  - Mixtures fed back autoregressively either amplify, so one component wins, or contract, so components become
    indistinguishable. Pretrained rollouts mostly amplify.
  - It covers soft-token mixtures; Coconut is not confirmed.
  - Implication (mine): naive merges may not stay "multi-path".

- **Latent-SFT / Chain of Superposition** (2510.15522) [abstract only]: claims latents carry "a superposition of
  diverse reasoning trajectories". Vocabulary-space latents.

- **MUX** (2607.18264) [abstract only]: superposes *sequential* subword spans per latent token, not alternative
  candidates.

### Not found / unverified

- PLR for sequential recommendation (2601.03153) [unverified, snippet]: mixture-of-streams aggregation in a
  recommender, not language reasoning.
- LTA-Thinker [unverified, citation only].

---

## 7. What has and has not been done (my inference unless quoted)

| Mechanism | Done in latent/continuous-CoT models? | Where |
|---|---|---|
| Sample N whole latent chains, aggregate answers (majority / BoN / RM) | Yes, many | LatentTTS (Coconut), ITS-Continuous (Coconut), LTO (Huginn), NoisyCoconut (LLM loop), SoftCoT++ (soft prefix), C-voting (puzzles) |
| Per-step multi-keep (beam) over latent steps | Yes, once | LatentTTS beam, Coconut only: B = √N, K = B, LatentRM cumulative reward; ties or trails BoN |
| Per-step keep-one-of-K (beam width 1 with K > 1) | **Not found** | LatentTTS's beam sweep ties B = K, so B = 1 means K = 1 |
| Any per-step selection on CODI or SIM-CoT | **Not found** | CODI appears only in coverage or pass@k tables (LatentTTS, SLPO, PLaT, Cui et al., GTS on Llama) |
| Merge K latent candidates into one latent / KV entry | **Not found** for Coconut/CODI/SIM-CoT | Only token-level merges (Multiplex, Soft Thinking, Latent-GRPO's top-K embedding superposition), plus theory (superposition) and negative dynamics (mixture collapse) |
| Final aggregation over a latent *tree* (branch at each step, vote over leaves) | **Not found** | The nearest are two-level trees used for RL credit (LaDi-RL N×M, LTC K×M) and text-readout branching (PLaT, Cui et al. hybrid) |
| RL-trained selector over latent candidates | **Not found** | RL-trained *proposer* with frozen backbone: GTS. Supervised selectors: LatentRM (step-wise contrastive on 128-rollout labels), ITS PRM/ORM, LTO LRM (BCE). Heuristic: C-voting confidence |
| Selector that reads candidates via a step decoder | **Not found** | SIM-CoT's decoder is used only for visualisation |

### Points for the project (inference)

1. **The aggregation gap is the consistent finding on GPT-2 latent models.**
   - Coverage is large: CODI GPT-2 reaches about 63.5 at N = 64 with dropout, 52.08 at P@8 with p = 0.1, and
     70.43 at Pass@100 with answer sampling.
   - Selection converts little: +1.5 (Coconut BoN-64 vs deterministic) and +2.3 (ITS PRM).
   - Majority vote is about greedy everywhere: CODI 42.23 vs 43.59, SIM-CoT 43.21 vs 42.23.
   - A per-step selector must beat this pattern, not just raise coverage.
2. **LatentTTS's beam search is the only per-step multi-keep baseline, and it is weak.**
   - It is ahead of BoN only at small N on GSM-Test and MultiArith.
   - The authors blame noisy early-step scores. That is exactly where Stage 0 says plans go wrong (slots 0–2),
     and where SIM-CoT's decoder reads latents best. This is the clearest gap our selector can target.
3. **Merging is open but risky.**
   - The mean of K MC-dropout reruns approximates the dropout-free forward pass (the standard weight-scaling
     approximation). A uniform merge of dropout candidates would therefore mostly reproduce candidate 0.
   - A merge has to be selector-weighted to differ, and mixture-collapse results suggest a merged latent fed back
     may drift toward one component anyway.
   - A cheap check: compare the uniform mean of K candidates with candidate 0, by cosine and by answer, on the
     Stage 0 data.
4. **Scorer feasibility looks architecture-dependent.** Coconut latents separate poorly (ITS-Continuous; PRM F1
   about 54), while Huginn latents separate well (LTO, AUC 0.88 on GSM8K). SIM-CoT's decoded steps may sit in
   between. Stage 0's AUC test answers this directly.
5. **CODI is sensitive to dropout** (LatentTTS Fig. 3): its coverage collapses beyond diversity of about 0.15. The
   p = 0.2 rate in Stage 0 should be checked against the identity and coverage numbers. LatentTTS used p = 0.2 for
   CODI coverage; SLPO used p = 0.1.
6. **Keep parallel chains vs keep one:** LTC's budget table (4×4 > 2×8 > 8×2 at B = 16) and LaDi-RL's (4×4 ≫ 4×1,
   1×4) say mid-width allocations win for *training*. There is no test-time evidence on CODI-type models.

## Sources

- LatentTTS: https://aclanthology.org/2026.acl-long.2069.pdf ; https://arxiv.org/abs/2510.07745 ;
  https://github.com/ModalityDance/LatentTTS ; https://huggingface.co/ModalityDance/latent-tts-rm
- LaDi-RL: https://arxiv.org/abs/2602.01705
- SoftCoT++: https://arxiv.org/abs/2505.11484
- ITS-Continuous: https://arxiv.org/abs/2510.12167
- NoisyCoconut: https://arxiv.org/abs/2605.08221
- LTO: https://arxiv.org/abs/2509.26314
- GTS: https://arxiv.org/abs/2602.14077
- C-voting: https://arxiv.org/abs/2604.13521
- Latent Thought Flow: https://arxiv.org/abs/2606.16222
- ATF: https://arxiv.org/abs/2609.33271
- Cui et al. weak/strong supervision: https://arxiv.org/abs/2602.22441
- PLaT: https://arxiv.org/abs/2601.21358
- SLPO: https://arxiv.org/abs/2607.19691
- LTC: https://arxiv.org/abs/2608.01593
- Dropout-GRPO: https://arxiv.org/abs/2606.10184
- CoLaR: https://arxiv.org/abs/2505.16552
- HRPO: https://arxiv.org/abs/2505.18454
- Latent-GRPO: https://arxiv.org/abs/2604.27998
- SIM-CoT: https://arxiv.org/abs/2509.20317
- LatentSeek: https://arxiv.org/abs/2505.13308
- LTPO: https://arxiv.org/abs/2510.04182
- RLTT: https://arxiv.org/abs/2602.10520
- ATP-Latent: https://arxiv.org/abs/2601.21598
- SWITCH: https://arxiv.org/abs/2606.13106
- PLRM: https://arxiv.org/abs/2610.29385
- Reasoning by Superposition: https://arxiv.org/abs/2505.12514
- Continuous Mixture Collapse: https://arxiv.org/abs/2609.02049
- Latent-SFT: https://arxiv.org/abs/2510.15522
- MUX: https://arxiv.org/abs/2607.18264
