# Latent-reasoning authors concede scale, promise adaptive depth

Across the 42 latent-reasoning and looped-transformer papers, the limitations and directions authors state most often are **model scale (27 of 42 papers), task breadth (26) and fixed-versus-adaptive compute (24)**. Next come extrapolation past the trained depth (20), architecture generality (19), evaluation rigor (18), training cost or stability and new objectives or RL (17 each), interpretability and efficiency claims (16 each), and theory gaps (15). Four of the top six say a study was not run widely enough, not that a method fails, so frequency is a poor guide to where accuracy can be gained. The method-level consensus is narrower and sharper. **13 of the 16 papers in the two looped-transformer families say extra depth past training stops paying or makes results worse.** Only one of the nine latent-CoT training papers says anything similar (CoLaR, and it is about compression factors), and none of the four CODI analysis papers runs CODI past its six trained steps. The direction with the biggest gap between being promised and being admitted as a limitation is **new objectives or RL: 16 papers propose it as future work, but only 5 call it a limitation**. Almost no paper states these: memory or attention across steps (4), test-time scaling and verification (5, four of them from the papers that built samplers), supervision dependence and hybrid language-latent reasoning (6 each), and the gap to explicit CoT (7, mostly from pre-2025 training papers). Thirteen papers have no section labelled as limitations, and the choice of version moves several counts by one. Compared with what the project has tried, the directions it has already tested are the ones the looped-model papers report as giving stability without gains: more steps, an inner loop, step-decoder supervision, a key-to-key bias and halting signals. My inference is that the strongest untested candidates for GPT-2 are three. The first is noise-sampled latent chains with better selection. The second is RL on those samples. The third is objective changes that the authors of CODI and Li et al. propose themselves, such as a new distillation target and supervision applied only to selected steps.

## Scale, breadth and adaptive compute top a sixteen-theme ranking

Five reading groups each read their papers in full and tagged every self-stated limitation (L) and future direction (F) against a shared taxonomy of 16 themes (T1–T16), adding NEW tags where nothing fit. The 42 papers, with the short names used below, are linked here to the version each group read.

| Family (papers) | Papers |
|---|---|
| Latent CoT training methods (9) | [CODI](https://arxiv.org/pdf/2502.21074v3), [SIM-CoT](https://proceedings.iclr.cc/paper_files/paper/2026/file/5d087955ee13fe9a7402eedec879b9c3-Paper-Conference.pdf) ([arXiv v2](https://arxiv.org/pdf/2509.20317v2)), [Coconut](https://arxiv.org/pdf/2412.06769v4), [PCCoT](https://arxiv.org/pdf/2506.18582v2), [PMPS](https://arxiv.org/pdf/2609.09928v2), [CoLaR](https://arxiv.org/pdf/2505.16552v6), [KaVa](https://arxiv.org/pdf/2510.02312v2), [ICoT-KD](https://arxiv.org/pdf/2311.01460v1), [ICoT-SI](https://arxiv.org/pdf/2405.14838v1) |
| Latent hybrids and analyses (9) | [LOTUS](https://arxiv.org/html/2606.31779v2), [Penelope](https://arxiv.org/html/2607.25915v1), [AGCLR](https://arxiv.org/html/2606.07720v1), [Pause tokens](https://arxiv.org/html/2310.02226v3), [Liang & Pan](https://arxiv.org/html/2602.00449v2), [Chang et al.](https://arxiv.org/html/2606.01243v1), [Li et al.](https://arxiv.org/html/2602.08783v3), [Duraipandian et al.](https://arxiv.org/html/2607.09698v1), [Dilgren & Wiegreffe](https://arxiv.org/html/2604.04902v2) |
| Latent test-time scaling and halting (8) | [SoftCoT++](https://arxiv.org/html/2505.11484v2), [LatentTTS](https://aclanthology.org/2026.acl-long.2069.pdf), [ITS-Continuous](https://aclanthology.org/2026.findings-acl.1338.pdf), [NoisyCoconut](https://arxiv.org/html/2605.08221v1), [Popescu et al.](https://arxiv.org/html/2607.20519v1), [TwoScale](https://arxiv.org/html/2509.23314v2), [FPRM](https://arxiv.org/html/2606.18206v1), [TaH](https://arxiv.org/html/2511.08577v4) |
| Looped LMs at scale (9) | [Huginn](https://proceedings.neurips.cc/paper_files/paper/2025/file/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Paper-Conference.pdf), [Ouro](https://arxiv.org/pdf/2510.25741v6), [Retrofit](https://arxiv.org/html/2511.07384v1), [Dreamer](https://arxiv.org/html/2601.21582v2), [InfiLoop](https://arxiv.org/html/2610.11570v2), [RecurTrace](https://arxiv.org/html/2609.03379v3), [FLT](https://arxiv.org/html/2605.18797v2), [TF-Loop](https://arxiv.org/html/2605.23872v1), [AdaLoopMem](https://arxiv.org/html/2603.08391v3) |
| Looped theory and extrapolation (7) | [Saunshi et al.](https://arxiv.org/html/2502.17416v1), [Fan et al.](https://arxiv.org/html/2409.15647v5), [Bansal et al.](https://arxiv.org/pdf/2202.05826v3), [Kuo et al.](https://arxiv.org/html/2606.29983v2), [Blayney et al.](https://arxiv.org/html/2604.11791v1), [ThinkShallow](https://arxiv.org/html/2608.18222v1), [Kaissis et al.](https://arxiv.org/html/2602.10097v1) |

### Harmonized counting rules

A paper counts **at most once per theme in each column**: as a limitation (L), as a future direction (F), and "at all". A statement that is both a limitation and a direction counts in both columns, a rule all five groups already followed. "As L" and "as F" are the limitation-only and direction-only counts. A paper that states a theme both ways appears in both, and the text notes when a theme is stated mostly or only one way.

The **headline count includes field-level statements that authors make in their own limitations, discussion or conclusion text**. These are the hybrids group's "scope: field" rows and the test-time group's "L-field" rows. Every group excluded critiques of prior work that only motivate a paper in its introduction or related work, and so does this merge.

The **own-study-only count** removes those field rows. It differs from the headline only in the hybrids and test-time groups. The training-methods and theory groups counted only own-study statements by rule. The looped-at-scale group counted statements about "the paradigm they work in" without separating them, so its paradigm-level rows remain in the own-only count, which therefore runs slightly high for that group.

Weak fits that the hybrids group flagged as "not tallied" stay out. Rows from appendices, results sections and NeurIPS checklists stay in, because every group read appendices. The theory group marked these rows, which makes a stricter sensitivity possible.

**Versions follow the version of record**: the camera-ready where a group read it, otherwise the latest arXiv version. This is the one place where the merge departs from a group's tally. It drops SIM-CoT's arXiv-v2-only Appendix J (four directions), which the training-methods group had counted, and it keeps the Limitation sections in the camera-readies of LatentTTS and ITS-Continuous.

Two pairs of NEW tags name the same idea, so each pair is merged here. Looped-at-scale "knowledge-capacity" is merged with theory "memorization-perplexity tradeoff", and training-methods "safety-or-misuse" with looped-at-scale "broader-impact-risk". Every other NEW tag is kept as its group coined it.

| Rank | Theme | Papers at all (headline) | At all (own-study only) | As L | As F | Training (9) | Hybrids (9) | Test-time (8) | At scale (9) | Theory (7) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | T1 model scale | **27** | 26 | 23 (own 22) | 16 | 6 | 6 | 3 | 8 | 4 |
| 2 | T2 task breadth | **26** | 25 | 22 (own 21) | 18 (own 17) | 6 | 7 | 4 | 4 | 5 |
| 3 | T5 fixed vs adaptive compute | **24** | 22 | 17 (own 15) | 18 (own 17) | 3 | 6 | 5 | 5 | 5 |
| 4 | T7 extrapolation past trained depth | **20** | 20 | 17 | 12 | 1 | 5 | 1 | 7 | 6 |
| 5 | T16 architecture generality | **19** | 19 | 15 | 11 | 0 | 5 | 5 | 6 | 3 |
| 6 | T10 evaluation rigor | **18** | 18 | 18 | 6 | 0 | 5 | 4 | 6 | 3 |
| 7= | T6 training cost or stability | **17** | 17 | 15 | 8 | 6 | 1 | 0 | 4 | 6 |
| 7= | T12 RL or new objectives | **17** | 15 | 5 | 16 (own 14) | 3 | 4 | 4 | 3 | 3 |
| 9= | T4 interpretability | **16** | 15 | 15 (own 14) | 10 | 5 | 4 | 2 | 3 | 2 |
| 9= | T9 efficiency claims | **16** | 16 | 15 | 10 | 0 | 3 | 4 | 6 | 3 |
| 11 | T13 theory gap | **15** | 15 | 13 | 10 | 1 | 2 | 4 | 2 | 6 |
| 12 | T3 gap to explicit CoT | 7 | 5 | 7 (own 5) | 1 | 4 | 2 | 0 | 1 | 0 |
| 13= | T8 supervision dependence | 6 | 5 | 6 (own 5) | 1 | 3 | 2 | 0 | 0 | 1 |
| 13= | T15 hybrid or tool integration | 6 | 6 | 1 | 6 | 3 | 0 | 0 | 2 | 1 |
| 15 | T11 test-time scaling and verification | 5 | 5 | 4 | 4 | 0 | 0 | 4 | 0 | 1 |
| 16 | T14 memory or attention over steps | 4 | 3 | 3 | 2 (own 1) | 0 | 1 | 0 | 3 | 0 |
| NEW | Knowledge capacity (merged) | ≥5 | ≥5 | 4 | 2 | 0 | 0 | 0 | 4 | 1 |
| NEW | Hyperparameter sensitivity | ≥4 | ≥4 | 3 | 2 | 0 | 0 | 4 | 0 | 0 |
| NEW | Safety or broader impact (merged) | ≥4 | ≥4 | 3 | 3 | 2 | 0 | 0 | 2 | 0 |
| NEW | Loop placement | ≥2 | ≥2 | 1 | 1 | 0 | 0 | 0 | 2 | 0 |
| NEW | Latent underuse; latent intervention; latent-token design; white-box access; language mixing; data scale or diversity | ≥1 each | 0–1 | — | — | — | — | — | — | — |

The asymmetries matter more than the ranks. **T12 is the field's favourite next step that almost nobody frames as a deficit.** Twelve of its 17 papers state it only as a direction. **T10 runs the other way: all 18 papers admit an evaluation-rigor limitation, and only 6 promise to fix it.** T15 is almost purely aspirational (6 directions, 1 limitation). T5 is the only top-three theme stated about as often as a promise (18) as a limitation (17).

The rankings are robust to the sensitivities the notes document. Restoring SIM-CoT's arXiv v2 Appendix J adds one paper each to T11 (6), T12 (18), T13 (16) and T15 (7). Dropping SIM-CoT's decoder-scaling ablation from T1, since it concerns the auxiliary decoder rather than the backbone, gives T1 26. Including the hybrids group's weak fits adds one each to T1, T3, T6 and T14. Applying the theory group's strict rule removes appendix-only rows: T6 falls to 14, T5 to 22, T10 to 16, and T1, T4, T7, T9 and T12 each lose one. None of these changes the top five.

## Eleven themes reach fifteen papers, in the authors' own words

### Scale and breadth are near-universal, and scale cuts both ways

Model scale (T1, 27 papers) is the closest thing to boilerplate in the corpus. It is nearly unanimous among looped LMs at scale (8 of 9; only InfiLoop omits it). CODI states it as a resource limit: "we don't have sufficient computation resources to scale the training of CODI on larger models" ([CODI](https://arxiv.org/pdf/2502.21074v3)). Huginn, trained on 800B tokens, still calls its models "ultimately still a proof-of-concept" ([Huginn](https://proceedings.neurips.cc/paper_files/paper/2025/file/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Paper-Conference.pdf)).

The direction of the scale effect is contested, though. Coconut reports that gains on Llama 3B and 8B "are not as pronounced as those previously demonstrated with GPT-2," and suggests that "larger models have already undergone extensive language-focused pre-training" ([Coconut](https://arxiv.org/pdf/2412.06769v4)). Pause tokens expected smaller models to benefit more, but "a preliminary comparison between our two model sizes surprisingly suggests the opposite" ([Pause tokens](https://arxiv.org/html/2310.02226v3)). Training-free looping states that "sub-scale models below ∼1.7B for Qwen3 and ∼3B for Llama lack the mid-layer redundancy the loop relies on" ([TF-Loop](https://arxiv.org/html/2605.23872v1)). Scale is therefore not a single limitation: GPT-2 sits on the favourable side for Coconut and on the unfavourable side for retrofitted loops.

Task breadth (T2, 26 papers) means **mathematics-only for the latent-CoT papers and synthetic-only for the theory papers**. PMPS writes that "the current evaluation of PMPS is mainly limited to mathematical reasoning" ([PMPS](https://arxiv.org/pdf/2609.09928v2)). LOTUS notes that it follows prior work and evaluates "on math benchmarks" ([LOTUS](https://arxiv.org/html/2606.31779v2)), and FPRM says "we test our model only on algorithmic tasks and not on natural language" ([FPRM](https://arxiv.org/html/2606.18206v1)). Neither theme is a lever for raising GSM8K accuracy on GPT-2. Their prominence mostly shows that authors use limitations sections to scope their claims.

### Adaptive compute is the most-promised fix for fixed latent budgets

Fixed versus adaptive compute (T5, 24 papers; 22 own-study) is the top theme that is about method design. Its directions fall into three kinds:

- **Adaptive budgets.** LOTUS: "Making K, c, and R adaptive is a natural and promising direction for future work" ([LOTUS](https://arxiv.org/html/2606.31779v2)).
- **Confidence-driven stopping.** Pause tokens: "better determining the number of `<pause>` tokens (perhaps using model confidence)" ([Pause tokens](https://arxiv.org/html/2310.02226v3)).
- **Stability-aware stopping.** Li et al.: stopping rules "should consider not only when an answer first becomes decodable, but also when the supporting latent representation becomes stable" ([Li et al.](https://arxiv.org/html/2602.08783v3)).

Two papers state the budget limit directly. Liang & Pan conclude that "with a fixed latent-compute budget, CODI can struggle when the underlying computation is effectively incompressible" ([Liang & Pan](https://arxiv.org/html/2602.00449v2)). KaVa blames latent methods' small margins over no-CoT on the fact that "the number of latent tokens remains fixed" ([KaVa](https://arxiv.org/pdf/2510.02312v2)).

The papers that actually built adaptive stopping report a gap they could not close. TaH's learned decider trails its oracle, and "the ideal oracle policy achieves even higher gains, indicating future potential" ([TaH](https://arxiv.org/html/2511.08577v4)). RecurTrace admits that without a depth floor "the head itself mostly stops at one loop" ([RecurTrace](https://arxiv.org/html/2609.03379v3)). Kuo et al. warn that "a stable schedule can still select a non-extrapolating computation" ([Kuo et al.](https://arxiv.org/html/2606.29983v2)).

### Extra depth stops paying in 13 of 16 looped-family papers

Extrapolation past the trained depth (T7, 20 papers) is where the literature comes closest to agreeing on a finding rather than a disclaimer. Seven of nine looped LMs at scale and six of seven theory papers state it. Ouro reports that its SFT models' "performance degrades as they extrapolate to deeper, unseen recurrent steps (T = 6 − 8), reinforcing that performance is optimized for the depth seen during training" ([Ouro](https://arxiv.org/pdf/2510.25741v6)). LOTUS finds that "running more than at training time brings no further gain" ([LOTUS](https://arxiv.org/html/2606.31779v2)). Dreamer calls reliable generalization "beyond the trained depth regime" "a critical and still underexplored problem" ([Dreamer](https://arxiv.org/html/2601.21582v2)).

The proposed fixes are variable-loop training and positional changes. FLT says "explicit training over variable loop counts may be necessary for robust test-time compute adaptation" ([FLT](https://arxiv.org/html/2605.18797v2)), and Dreamer suggests "dynamic RoPE scaling" or "alternatives to RoPE" ([Dreamer](https://arxiv.org/html/2601.21582v2)).

The theory papers then temper those fixes. ThinkShallow reports that its repair "restores depth-safety, not new capability" and that "conversion never appears" at 3.5B scale ([ThinkShallow](https://arxiv.org/html/2608.18222v1)). Kuo et al. find that "additional available depth alone does not ensure a better extrapolating computation" ([Kuo et al.](https://arxiv.org/html/2606.29983v2)). Bansal et al. report "an apparent ceiling on generalizing from the easy puzzles to much harder ones" ([Bansal et al.](https://arxiv.org/pdf/2202.05826v3)).

The latent-CoT side is nearly silent on this theme. Only CoLaR states T7, and it does so about compression factors greater than "the maximum training compression factor" ([CoLaR](https://arxiv.org/pdf/2505.16552v6)). The hybrids group found that none of the four CODI analysis papers runs CODI with more latent steps than it was trained on ([Duraipandian et al.](https://arxiv.org/html/2607.09698v1), [Liang & Pan](https://arxiv.org/html/2602.00449v2)).

### Generality, rigor, cost and efficiency are admitted more than promised

These four themes have strong L columns and weak F columns.

**Architecture generality (T16, 19 papers).** The analysis papers use it to flag that their results are specific to CODI. Liang & Pan say "further work is needed to determine how broadly the observed mechanisms extend to other latent-reasoning architectures, such as looped transformers, recurrent latent-state models, or hybrid latent-text approaches" ([Liang & Pan](https://arxiv.org/html/2602.00449v2)). The looped LMs use it to flag topology: "we also focus on a single recurrence, where prior work has considered multiple successive recurrent stages" ([Huginn](https://proceedings.neurips.cc/paper_files/paper/2025/file/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Paper-Conference.pdf)). Retrofit adds that its per-FLOP gain from larger training recurrences, seen for OLMo and Llama, "does not hold for TinyLlama" ([Retrofit](https://arxiv.org/html/2511.07384v1)).

**Evaluation rigor (T10, 18 papers).** It is admitted by every paper that states it and promised by only six. AGCLR states that "our evaluation uses single-seed GPT-2 124M on three benchmarks" ([AGCLR](https://arxiv.org/html/2606.07720v1)). FLT says "we use a single run per configuration and do not report training-run error bars" ([FLT](https://arxiv.org/html/2605.18797v2)). Popescu et al. concede that "several of these differences are small and reported on single runs" ([Popescu et al.](https://arxiv.org/html/2607.20519v1)).

**Training cost or stability (T6, 17 papers).** This theme belongs to the latent-CoT trainers and the theorists. CODI flags "the absence of intermediate gradients until the end of the sequence" ([CODI](https://arxiv.org/pdf/2502.21074v3)). Coconut reports that "adding three continuous thoughts at once ... leads to a sharp spike in training loss" ([Coconut](https://arxiv.org/pdf/2412.06769v4)). SIM-CoT's camera-ready concedes that its "parameter count and memory usage are approximately doubled compared with the implicit baselines Coconut and CODI" ([SIM-CoT](https://proceedings.iclr.cc/paper_files/paper/2026/file/5d087955ee13fe9a7402eedec879b9c3-Paper-Conference.pdf)).

**Efficiency claims (T9, 16 papers).** These are about measured runtime. Dreamer reports inference slowed "by ∼13% with an unoptimized DA implementation" ([Dreamer](https://arxiv.org/html/2601.21582v2)). Penelope says it "does not parameter-match the methods or equalize training FLOPs, wall-clock time, or target presentations" ([Penelope](https://arxiv.org/html/2607.25915v1)), and InfiLoop's only caveat is that "these counts do not compare measured runtime" ([InfiLoop](https://arxiv.org/html/2610.11570v2)).

### New objectives are promised by sixteen papers and conceded by five

RL or new objectives (T12, 17 papers; 15 own-study) is the largest pool of concrete proposals for raising accuracy, and it is the most heterogeneous. Its main strands:

- **Change the distillation target.** CODI's own Limitations section proposes it: "probing the token that concludes the CoT reasoning—potentially summarizing the entire process—could offer alternative supervision signals." It also calls the answer prompt "The answer is:" "an arbitrary design choice that may influence the effectiveness of knowledge transfer" ([CODI](https://arxiv.org/pdf/2502.21074v3)).
- **Supervise steps by role, not uniformly.** Li et al. argue that "future objectives could move beyond uniform CoT imitation and instead impose functional constraints on high-leverage steps or high-influence routes." They add that "early states should preserve uncertainty or multiple candidate modes, while later states consolidate them into a perturbation-stable decision" ([Li et al.](https://arxiv.org/html/2602.08783v3)).
- **RL over sampled trajectories.** LatentTTS proposes "integrating sampling and aggregation into a reinforcement learning framework to optimize latent trajectories" ([LatentTTS](https://aclanthology.org/2026.acl-long.2069.pdf)), and CoLaR and TaH name RL as well ([CoLaR](https://arxiv.org/pdf/2505.16552v6), [TaH](https://arxiv.org/html/2511.08577v4)).
- **Geometric and contrastive terms.** ITS-Continuous proposes "encouraging higher isotropy in thought representations, promoting trajectory diversity ... and incorporating contrastive learning" ([ITS-Continuous](https://aclanthology.org/2026.findings-acl.1338.pdf)).

The one RL attempt reported as a limitation is a caution. Ouro's RL "did not yield significant performance gains over the final SFT checkpoint," and the authors attribute this to "the model's dynamic early-exit mechanism" ([Ouro](https://arxiv.org/pdf/2510.25741v6)).

### Interpretability and theory gaps leave latent dynamics unexplained

Interpretability (T4, 16 papers) concerns either the opacity of latent reasoning or the reach of the probing tools used on it. On opacity, CODI writes that "implicit CoT methods inherently trade off interpretability compared to explicit CoT" ([CODI](https://arxiv.org/pdf/2502.21074v3)). On tool reach, Dilgren & Wiegreffe note that vocabulary projection "only reveals single-token concepts" ([Dilgren & Wiegreffe](https://arxiv.org/html/2604.04902v2)). Duraipandian et al. add that decreasing displacement "does not necessarily imply convergence toward a correct solution" ([Duraipandian et al.](https://arxiv.org/html/2607.09698v1)).

Theory gaps (T13, 15 papers) dominate the theory family (6 of 7). They also include the unexplained regularities that the project has met itself. PCCoT finds that its latent tokens "do not converge in multiple iterations after training," and that odd and even latent positions form an interleaved pattern: "Up to now, we have not found a clear explanation for this phenomenon" ([PCCoT](https://arxiv.org/pdf/2506.18582v2)). Saunshi et al. say of their looping bias, "why such an inductive bias occurs is still an open question" ([Saunshi et al.](https://arxiv.org/html/2502.17416v1)). Kuo et al. describe their own mechanism as "only a stabilising mechanism" with no guarantee "that the consistent computation is the extrapolating one" ([Kuo et al.](https://arxiv.org/html/2606.29983v2)).

### Who states each theme

"(field)" marks a paper whose only statement of the theme is field-level; it drops out of the own-study count.

| Theme | Papers stating it | Stated only as a direction |
|---|---|---|
| T1 scale (27) | CODI, SIM-CoT, Coconut, PCCoT, ICoT-KD, ICoT-SI; Penelope, AGCLR, Pause, Li et al., Duraipandian, Dilgren (field); SoftCoT++, LatentTTS, TwoScale; Huginn, Ouro, Retrofit, Dreamer, RecurTrace, FLT, TF-Loop, AdaLoopMem; Fan et al., Kuo, Blayney, ThinkShallow | PCCoT, ICoT-KD, Penelope, Li et al. |
| T2 breadth (26) | CODI, Coconut, PCCoT, PMPS, CoLaR, ICoT-SI; LOTUS, Penelope, AGCLR, Liang & Pan, Li et al., Duraipandian, Dilgren; LatentTTS (field), ITS-Continuous, NoisyCoconut, FPRM; Retrofit, Dreamer, RecurTrace, TF-Loop; Saunshi, Fan et al., Bansal, Kuo, ThinkShallow | CODI, Coconut, PCCoT, CoLaR |
| T5 adaptive compute (24) | Coconut, CoLaR, KaVa; LOTUS, Penelope, Pause, Liang & Pan (field), Li et al., Dilgren (field); NoisyCoconut, Popescu, TwoScale, FPRM, TaH; Ouro, Retrofit, Dreamer, RecurTrace, FLT; Saunshi, Fan et al., Bansal, Kuo, Kaissis | NoisyCoconut, Popescu, TwoScale, Retrofit, FLT, Saunshi, Bansal |
| T7 past trained depth (20) | CoLaR; LOTUS, Penelope, Pause, Li et al., Duraipandian; Popescu; Ouro, Retrofit, Dreamer, RecurTrace, FLT, TF-Loop, AdaLoopMem; Fan et al., Bansal, Kuo, Blayney, ThinkShallow, Kaissis | Li et al., Duraipandian, Kaissis |
| T16 architecture (19) | Pause, Liang & Pan, Li et al., Duraipandian, Dilgren; SoftCoT++, ITS-Continuous, NoisyCoconut, TwoScale, FPRM; Huginn, Ouro, Retrofit, Dreamer, RecurTrace, FLT; Saunshi, Fan et al., Blayney | SoftCoT++, Ouro, Dreamer, Saunshi |
| T10 rigor (18) | Penelope, AGCLR, Liang & Pan, Li et al., Dilgren; ITS-Continuous, Popescu, TwoScale, TaH; Huginn, Ouro, RecurTrace, FLT, TF-Loop, AdaLoopMem; Kuo, Blayney, ThinkShallow | none |
| T6 cost or stability (17) | CODI, SIM-CoT, Coconut, PCCoT, ICoT-KD, ICoT-SI; Pause; Huginn, Ouro, Retrofit, FLT; Fan et al., Bansal, Kuo, Blayney, ThinkShallow, Kaissis | ICoT-KD, FLT |
| T12 objectives or RL (17) | CODI, CoLaR, ICoT-KD; Pause, Chang, Li et al. (field), Dilgren (field); LatentTTS, ITS-Continuous, Popescu, TaH; Huginn, Ouro, FLT; Bansal, Kuo, Kaissis | all except CODI, Pause, ITS-Continuous, TaH, Ouro |
| T4 interpretability (16) | CODI, PCCoT, KaVa, ICoT-KD, ICoT-SI; AGCLR, Li et al., Duraipandian, Dilgren; ITS-Continuous (field), TwoScale; Huginn, Dreamer, FLT; Blayney, Kaissis | FLT |
| T9 efficiency (16) | LOTUS, Penelope, Chang; LatentTTS, NoisyCoconut, FPRM, TaH; Ouro, Dreamer, InfiLoop, FLT, TF-Loop, AdaLoopMem; Kuo, Blayney, Kaissis | Blayney |
| T13 theory (15) | PCCoT; Pause, Liang & Pan; SoftCoT++, NoisyCoconut, TwoScale, FPRM; Ouro, TF-Loop; Saunshi, Fan et al., Kuo, Blayney, ThinkShallow, Kaissis | Liang & Pan, SoftCoT++ |
| T3 CoT gap (7) | Coconut, CoLaR, ICoT-KD, ICoT-SI; Liang & Pan (field), Dilgren (field); Huginn | none |
| T8 supervision (6) | Coconut, PCCoT, ICoT-KD; Li et al. (field), Dilgren; Fan et al. | none |
| T15 hybrid (6) | Coconut, ICoT-KD, ICoT-SI; Dreamer, RecurTrace; Saunshi | all except RecurTrace |
| T11 TTS and verification (5) | SoftCoT++, LatentTTS, ITS-Continuous, NoisyCoconut; Saunshi | Saunshi |
| T14 memory over steps (4) | Li et al. (field); Ouro, Dreamer, RecurTrace | Li et al. |

## Each paper family has a signature worry, and a third skip the limitations section

The five families differ more in what they admit than in how much.

| Family | What its papers state most | What none of them state |
|---|---|---|
| Latent-CoT training methods | Scale, breadth and training cost (6 of 9 each), interpretability (5 of 9), and the gap to explicit CoT (4 of 9). This is the only family where conceding the CoT gap is common. | Efficiency, evaluation rigor, architecture generality, memory over steps |
| Hybrids and analyses | Breadth (7 of 9), then scale and adaptive compute (6 of 9). The analysis papers spend their limitations on scope: the fixed six-step budget (Li et al., "a fixed latent budget (T=6)") and CODI-only coverage ([Li et al.](https://arxiv.org/html/2602.08783v3)). | Test-time scaling, hybrid reasoning |
| Test-time scaling and halting | Adaptive compute and architecture generality (5 of 8). It is the only family with real weight on T11 (4 of 8) and on hyperparameter sensitivity (4 of 8). | Gap to explicit CoT, training cost, supervision dependence, memory, hybrid reasoning |
| Looped LMs at scale | Scale (8 of 9) and extrapolation (7 of 9), then efficiency, rigor and generality (6 of 9). It is the only family with more than one paper on memory over steps (3 of 9) and the source of the knowledge-capacity tag (4 of 9). | Supervision dependence, verification |
| Theory and extrapolation | Training cost, extrapolation and theory gaps (6 of 7), with breadth and adaptive compute at 5 of 7 | Gap to explicit CoT, memory over steps |

One reading follows from that table. **The latent-CoT training methods and the looped families describe the same failure from opposite ends.** The looped families treat extra depth as the central open problem (13 of 16 papers state T7). The latent-CoT trainers fix the number of latents and never report what happens beyond it, so the drift CODI shows past six steps falls outside both literatures' stated limitations. The training-methods group adds that SIM-CoT's Appendix K shows trailing latents that "simply repeat the final prediction," but presents this as a visualization note ([SIM-CoT arXiv v2](https://arxiv.org/pdf/2509.20317v2)).

A second pattern is a convergence across three theory papers. Bansal et al., ThinkShallow and Kuo et al. each say, in different words, that a loop settling or stabilizing is not the same as giving better answers at more depth. Meanwhile, the three test-time papers that use hidden-state convergence as a halting signal never acknowledge that convergence can land on a stable but wrong state: TwoScale, FPRM and Popescu et al. ([TwoScale](https://arxiv.org/html/2509.23314v2), [FPRM](https://arxiv.org/html/2606.18206v1)).

Self-criticism also tracks era and venue. The 2023–2024 training papers, ICoT-KD and ICoT-SI, have the most itemized Limitations sections ([ICoT-SI](https://arxiv.org/pdf/2405.14838v1)), while PMPS (2026) states a single limitation. NoisyCoconut is the most self-critical paper in the corpus, with 13 limitation rows and 16 direction rows ([NoisyCoconut](https://arxiv.org/html/2605.08221v1)). The two workshop papers in the looped-at-scale family, Retrofit and AdaLoopMem, have shorter lists than the main-conference ones.

### Thirteen papers have nothing labelled as limitations

**Thirteen of 42 papers have no section or paragraph labelled as limitations.** They are SIM-CoT, Coconut, PMPS and KaVa among the training methods; Penelope (only "Discussion and Future Work"), Duraipandian et al. and Dilgren & Wiegreffe among the hybrids; Popescu et al. among the test-time papers; Ouro, Retrofit, InfiLoop and TF-Loop among the looped LMs at scale; and Bansal et al. among the theory papers. For these, the groups extracted statements from conclusions, discussions, ethics statements and appendices.

InfiLoop is the extreme case. It has no limitations, future-work, discussion or impact text, and its single caveat sits in an appendix ([InfiLoop](https://arxiv.org/html/2610.11570v2)). TF-Loop keeps a long log of failed configurations but frames none of them as limitations and proposes nothing ([TF-Loop](https://arxiv.org/html/2605.23872v1)).

**Four papers state no future direction at all in their version of record**: SIM-CoT, InfiLoop, TF-Loop and ThinkShallow. Blayney et al. head a paragraph "Limitations and Future Work" without writing an explicit direction in it ([Blayney et al.](https://arxiv.org/html/2604.11791v1)). TaH's limitations exist only in Appendix A.6 ([TaH](https://arxiv.org/html/2511.08577v4)).

The project's two baselines sit on opposite sides of this line. CODI has a dedicated §7 Limitations. SIM-CoT has none in either version.

### Versions add and remove statements

Peer review usually added limitation text. SIM-CoT is the exception:

- **SIM-CoT.** Its arXiv v2 Appendix J listed four directions: multimodal extension; "multi-path implicit reasoning," described as "exploring branched latent structures, inspired by Tree-of-Thought"; RLHF integration; and information-theoretic foundations ([SIM-CoT arXiv v2](https://arxiv.org/pdf/2509.20317v2)). The ICLR camera-ready drops that appendix and adds a training-overhead subsection ([SIM-CoT](https://proceedings.iclr.cc/paper_files/paper/2026/file/5d087955ee13fe9a7402eedec879b9c3-Paper-Conference.pdf)).
- **LatentTTS.** Its Limitation section, which holds its runtime and hyperparameter-sensitivity statements, appears in v1 and in the ACL camera-ready but **not in arXiv v2–v4**, whose HTML also cuts the future-work sentence off mid-list ([LatentTTS](https://aclanthology.org/2026.acl-long.2069.pdf)). Its v1 conclusion had a different direction, about "adjusting compute allocation dynamically, and generalizing across tasks," which the camera-ready replaced with the RL and ensemble directions ([LatentTTS v1](https://arxiv.org/html/2510.07745v1)).
- **ITS-Continuous.** The Findings camera-ready adds a Limitation section and a geometric-regularization section that arXiv v2 lacks ([ITS-Continuous](https://aclanthology.org/2026.findings-acl.1338.pdf)).
- **KaVa.** Its ICLR v2 adds an Ethics statement that is the only source of its interpretability limitation and its safety direction ([KaVa](https://arxiv.org/pdf/2510.02312v2)).
- **Huginn.** The camera-ready adds a combined "Limitations and Conclusions" section but drops some v2 caveats about the training run. This changes no tally ([Huginn arXiv v2](https://arxiv.org/html/2502.05171v2)).

The practical lesson for anyone citing these papers is that **the latest arXiv version can be the least complete one**. Earlier versions of Coconut, CoLaR, CODI and PCCoT were not compared, and neither were the ACL Anthology, COLM, OpenReview or PMLR camera-readies of CODI, PCCoT, Coconut, Pause tokens or TaH.

## Verification, step memory and the CoT gap barely register

**The gap to explicit CoT (T3, 7 papers; 5 own-study)** is conceded almost only by the training papers written before 2025-26. ICoT-SI writes that "explicit CoT still achieves higher accuracies compared to our approach to implicit CoT" ([ICoT-SI](https://arxiv.org/pdf/2405.14838v1)). CoLaR "approximates explicit CoT reasoning without surpassing it" ([CoLaR](https://arxiv.org/pdf/2505.16552v6)). Coconut says "universally surpassing language-based CoT likely requires significant research efforts dedicated to latent space pre-training" ([Coconut](https://arxiv.org/pdf/2412.06769v4)).

Newer papers report parity as a success. CODI reports reaching 99% of CoT-SFT accuracy on GPT-2 without calling the remainder a limitation ([CODI](https://arxiv.org/pdf/2502.21074v3)). LOTUS reports staying within about 1.5 points of explicit CoT ([LOTUS](https://arxiv.org/html/2606.31779v2)). LatentTTS claims that COCONUT with test-time scaling beats explicit CoT on the same GPT-2 backbone at lower compute ([LatentTTS](https://aclanthology.org/2026.acl-long.2069.pdf)). Huginn's T3 is a missing comparison rather than a conceded gap: it lacks a comparison with "verbal CoT models targeting the same FLOP count at test time" ([Huginn](https://proceedings.neurips.cc/paper_files/paper/2025/file/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Paper-Conference.pdf)).

**Supervision dependence (T8, 6 papers)** comes mainly from the CODI lineage. PCCoT writes that its training "relies on distillation from the CoT teacher task" ([PCCoT](https://arxiv.org/pdf/2506.18582v2)). Fan et al. note that their step-dependent supervision "requires the ground-truth number of steps in the training data" ([Fan et al.](https://arxiv.org/html/2409.15647v5)). Coconut asks for strategies that work "without the supervision from language reasoning chains" ([Coconut](https://arxiv.org/pdf/2412.06769v4)). Two dependencies of the same kind go unflagged: RecurTrace distills its halting head from an oracle loss signal, and TaH trains its decider on oracle labels.

**Hybrid language-latent reasoning (T15, 6 papers)** is stated as a limitation only once. Five of the six papers state it only as a direction. Coconut proposes "generating the reasoning skeleton in language and completing the reasoning process in latent space" ([Coconut](https://arxiv.org/pdf/2412.06769v4)). RecurTrace calls latent looping and CoT "complementary rather than direct substitutes" ([RecurTrace](https://arxiv.org/html/2609.03379v3)). ICoT-SI hopes a "mixed-mode approach" could "offer the best of both worlds" ([ICoT-SI](https://arxiv.org/pdf/2405.14838v1)). Adding SIM-CoT's v2 multimodal direction gives 7, and recoding Liang & Pan's "hybrid latent-text approaches" from T16 gives 8.

**Test-time scaling and verification (T11, 5 papers)** is concentrated in the test-time family, plus Saunshi et al., who call "leveraging looping explicitly for inference-time scaling" "a very promising future direction" ([Saunshi et al.](https://arxiv.org/html/2502.17416v1)). **Verification specifically is stated only by the three papers that built verifiers or aggregators.**

- **ITS-Continuous** finds that "established text-based reward modeling methods transfer poorly to continuous reasoning," and that "continuous thoughts lack the geometric separability needed to distinguish correct from incorrect reasoning" ([ITS-Continuous](https://aclanthology.org/2026.findings-acl.1338.pdf)).
- **LatentTTS** reports that "a performance gap persists between the Best-of-N with LatentRM and the Coverage@N upper bound" ([LatentTTS](https://aclanthology.org/2026.acl-long.2069.pdf)).
- **NoisyCoconut** limits itself to "consensus by voting ... via exact match," and says "the vast space of possible aggregation strategies remains largely unexplored" ([NoisyCoconut](https://arxiv.org/html/2605.08221v1)).

**Memory or attention over steps (T14, 4 papers; 3 own-study)** is the rarest theme, and every own-study statement comes from the looped LMs at scale.

- **Ouro** finds that each recurrent step "requires maintaining its own KV cache, leading to 4× memory overhead," and that reusing caches during prefill costs ">10 points on GSM8K" ([Ouro](https://arxiv.org/pdf/2510.25741v6)).
- **Dreamer's** memory is constant in sequence length "though not wrt. depth" ([Dreamer](https://arxiv.org/html/2601.21582v2)).
- **RecurTrace's** loop memory "excludes older states and cross-position memory" ([RecurTrace](https://arxiv.org/html/2609.03379v3)).
- **Li et al.** state the field-level version: models could improve "by explicitly shaping where information is written and how it is consolidated across steps ... rather than only scaling the number of latent steps" ([Li et al.](https://arxiv.org/html/2602.08783v3)).

KaVa, a KV-cache distillation method, states no T14 limitation. AGCLR's "concept bottleneck" of overwritten hidden states appears only as motivation, so it is not counted ([AGCLR](https://arxiv.org/html/2606.07720v1)).

### New tags are lower bounds, and three of them fold cleanly

Each NEW tag was checked only by the group that coined it. Its count is therefore a floor, and statements of the same kind sit under other T-tags in the other groups' notes.

| NEW tag | Coined by | Lower-bound count | Papers | Merge or fold |
|---|---|---|---|---|
| Knowledge capacity | Looped at scale; theory (as "memorization-perplexity tradeoff") | ≥5 | Huginn, Ouro, AdaLoopMem, Dreamer (F only), Saunshi et al. | Merged here, since both name the same claim. AdaLoopMem: looping "cannot compensate for insufficient storage capacities" ([AdaLoopMem](https://arxiv.org/html/2603.08391v3)). TF-Loop reports the opposite pattern, with its gains concentrated on knowledge-heavy multiple-choice tasks. |
| Hyperparameter sensitivity | Test-time | ≥4 | LatentTTS, ITS-Continuous, NoisyCoconut, TwoScale | LatentTTS: "optimal performance depends on tuning the dropout rate (p) and noise scale (σ)" ([LatentTTS](https://aclanthology.org/2026.acl-long.2069.pdf)). On my re-reading, rows in other notes would also qualify, though I have not counted them: Ouro's untuned learning rates, Blayney's classification hyperparameters, ICoT-SI's Δ schedule, Kuo's entropy regularization and LOTUS's fixed K, c and R. |
| Safety or broader impact | Training ("safety-or-misuse"); looped at scale ("broader-impact-risk") | ≥4 | CoLaR, KaVa, Dreamer, FLT | Merged. Huginn's oversight concern sits under T4 by its group's rule. The hybrids, test-time and theory groups did not code this tag. |
| Loop placement | Looped at scale | ≥2 | Retrofit (F), TF-Loop (L) | Adding Saunshi et al.'s "best looping strategies" (middle looping, which the theory group coded as T16) gives 3. |
| Latent underuse | Hybrids | ≥1 (0 own-study) | Dilgren & Wiegreffe (field) | Two rows coded under other tags fit it: ITS-Continuous's "raises questions about continuous thoughts' actual contribution" (coded T4) and KaVa's small margins over no-CoT (coded T5). With them the count is 3. |
| Data scale or diversity | Training | ≥1 | KaVa | Folding into T8, as its coiners suggest, gives T8 7. Data-scale rows about pretraining were coded T1 (for example Huginn's 800B tokens), so the fold applies only to CoT data. |
| Latent intervention | Hybrids | ≥1 | Duraipandian et al. (F) | Folding into T11 gives 6. Chang et al.'s method is exactly this, but they state no limitation or direction about it. |
| Latent-token design; white-box access; language mixing | Hybrids; test-time; test-time | 1 each | Pause tokens; NoisyCoconut; NoisyCoconut | No natural fold |

## The project has already tested the directions authors report as stalling

The table below maps the most frequent stated directions against the project's own record. That record comes from the project's own notes and result directories, not from the reviewed papers. The feasibility column is my inference, not a count.

| Direction (papers stating it as F) | What the authors propose | Project status | GPT-2 feasibility (inference) |
|---|---|---|---|
| Task breadth, T2 (18) | Other domains beyond math and synthetic tasks | Untested; GSM8K only | Not an accuracy lever for this goal |
| Adaptive compute, T5 (18) | Per-input halting, learned stopping, variable-depth training | Tested. A depth-agreement stopping signal lost to plain token log-probability in preregistered probes. The inner loop trained with random k from 1 to 4, and its gate settled at 0.013. | Done. The authors' own results (the TaH oracle gap; Kuo's point that stability is not extrapolation) predict saved compute, not higher peak accuracy. |
| Scale, T1 (16) | Larger backbones, pretraining | Tested in the held-out and scale study, with drift on all four models | Out of scope at GPT-2 |
| New objectives or RL, T12 (16) | New distillation target, step-role supervision, RL, geometric and contrastive terms | A uniform loss from SIM-CoT's frozen step decoder was tested against matched answer-only arms over two seeds and made no difference: 209 against 201 at seed 7 and 200 against 202 at seed 8, or −3 per seed pooled (p=0.52), per `results/inner_loop/`. The earlier 199-vs-215 gap compared runs with different settings and lies within seed noise. | **The largest untested pool; feasible** |
| Past trained depth, T7 (12) | Variable-loop training, positional changes, better extrapolation | Tested. CODI drifts from 206 to 173 of 500 past six steps. Dedup and re-anchoring hold the score but never beat six steps. | Done. The result matches the "depth-safety, not conversion" consensus. |
| Architecture and loop topology, T16 (11), plus loop placement (≥2) | Multiple recurrences, middle looping, layer choice | One inner-loop window (layers 4–7), null | Other windows untested; low prior, given TF-Loop's evidence on small models |
| Interpretability-guided intervention, T4 (10), plus latent intervention (≥1) | Use decoded traces or causal hubs to steer latents | Anchoring writes, every-loop and periodic, had no measurable effect | Gradient-guided edits (Chang et al.) untested; training-free, with small reported gains |
| Hybrid language-latent, T15 (6) | Language skeleton with latent completion; mixed mode | Decode-and-re-inject was demoted to diagnostics because logit-lens decodes are unreliable at GPT-2 | Feasible, but it changes the comparison class |
| Sampling, aggregation and verification, T11 (4) | Noise or dropout sampling, voting, latent reward models, ensembles | Voting across depths was tested and never helps; noise sampling with a step-decoder verifier is queued | **Feasible; the only direction with author-reported GPT-2 evidence** |
| Memory or attention over steps, T14 (2) | Depth attention, consolidation across steps | The key-to-key bias (KK-LSE) changed 1 answer of 500, and the gate and dedup only hold the score | Done |

The overlap is striking. **The project's nulls fall on the four themes where the looped literature itself reports stability without gains: T5, T7, T14 and loop topology.** ThinkShallow's line that a repair "restores depth-safety, not new capability" ([ThinkShallow](https://arxiv.org/html/2608.18222v1)) and LOTUS's finding of "no further gain" past the trained depth ([LOTUS](https://arxiv.org/html/2606.31779v2)) say in print what the project's dedup, re-anchoring and inner-loop runs found. The inner-loop null also fits two observations from TF-Loop. Undamped looping "diverges visibly within four applications, even on a window of just four mid layers," and sub-1.7B models "lack the mid-layer redundancy the loop relies on" ([TF-Loop](https://arxiv.org/html/2605.23872v1)). Both observations come from training-free looping, not from the fine-tuned loop the project ran, so they are suggestive rather than explanatory.

**Inference, not a count:** five frequently stated directions remain untested by the project and look feasible on GPT-2 with modest compute. In my order of priority they are as follows.

**First, noise-sampled parallel latent chains with better selection (T11).** The project has already queued this. It is the only direction whose authors ran it on GPT-2-124M. LatentTTS reports that coverage rises while selection is the bottleneck ([LatentTTS](https://aclanthology.org/2026.acl-long.2069.pdf)). ITS-Continuous warns that text-style reward models will not provide that selection, because correct and incorrect latents are barely separable ([ITS-Continuous](https://aclanthology.org/2026.findings-acl.1338.pdf)). GSM8K answers are discrete, so NoisyCoconut's exact-match voting applies directly ([NoisyCoconut](https://arxiv.org/html/2605.08221v1)). The project's planned twist, SIM-CoT's step decoder used as a verifier, rests on a cross-question correlation in its own records (63% against 36% accuracy). That correlation has not yet been tested within a question.

**Second, RL over those sampled chains (T12, proposed by LatentTTS, CoLaR, TaH and Huginn).** It reuses the sampling machinery as exploration and uses exact-match answers as the reward. Ouro's failed RL was blamed on dynamic early exit and on rollout infrastructure that breaks under variable depth ([Ouro](https://arxiv.org/pdf/2510.25741v6)). CODI's fixed six steps avoid both obstacles, which is my inference rather than a tested claim.

**Third, the distillation-target changes that CODI's own authors propose.** These are distilling at the token that concludes the CoT rather than at ":" and varying the "The answer is:" prompt ([CODI](https://arxiv.org/pdf/2502.21074v3)). Each variant costs one CODI retrain, and none of the five notes records a paper trying either.

**Fourth, step-selective supervision in place of uniform supervision.** Li et al.'s advice runs opposite to the uniform step-decoder loss, which made no difference in the project. It constrains only high-leverage steps, keeps early steps multi-modal and consolidates late ([Li et al.](https://arxiv.org/html/2602.08783v3)). ITS-Continuous's isotropy, diversity and contrastive terms are a natural companion, because they also make samples more separable for the first direction ([ITS-Continuous](https://aclanthology.org/2026.findings-acl.1338.pdf)). SIM-CoT's dropped "multi-path implicit reasoning" direction points the same way ([SIM-CoT arXiv v2](https://arxiv.org/pdf/2509.20317v2)).

**Fifth, a language skeleton with latent completion (T15).** It is feasible, but it would no longer be a pure latent-versus-latent comparison. The project's planned gold-step injection diagnostic gives its ceiling cheaply before any training.

One reading caution applies to all five. The project measured a seed spread of about 9 of 500 (1.8 points) for the same recipe. Several reported gains in the corpus come from single runs and are of similar size, for example Chang et al.'s +1.3% on GSM-Hard ([Chang et al.](https://arxiv.org/html/2606.01243v1)). The preregistered thresholds the project already uses are stricter than most of the evidence behind these directions.

## Conclusion

The corpus shows that frequency and promise point in different directions. The most-stated themes, scale, breadth, rigor and generality, are mostly authors scoping their claims, while the one strong method-level consensus is negative: extra depth buys stability, not accuracy. The project has rediscovered that consensus on CODI, a model family whose authors never report it. That alone is a defensible contribution for the course report, because the latent-CoT and looped-model literatures each leave the CODI case outside their stated limitations.

The rare themes are where the open accuracy levers sit. Test-time scaling and verification is stated by 5 papers and memory by 4, yet sampling-based test-time scaling is the only direction with GPT-2 evidence from its authors. New objectives are promised by 16 papers but conceded by only 5, which means nobody has shown that the current objective is the bottleneck. The project's step-decoder null is one of the few hard data points on that question. My recommendation is to move effort away from adding depth, attention and halting, which both the literature and the project's own results now argue against, and toward selection and objective design. Measure every candidate against the 1.8-point seed spread before believing it.
