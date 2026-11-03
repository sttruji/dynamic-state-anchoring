# Looped / recurrent-depth transformers and test-time depth extrapolation (context: append-only latent CoT, CODI / SIM-CoT)

Notes compiled 2026-11-02. Reading-status tags: **[main text]** means I read the HTML/PDF body (sometimes only the first ~100k characters, noted where it matters). **[abstract]** means I saw only the arXiv landing page or abstract, so mechanism details and numbers are unverified. Unless a venue is given, every 2026 item is an **arXiv preprint (2026 preprint)** and is not peer reviewed. Several 2026 papers were found through a community reading list ([Awesome-Loop-Transformers](https://github.com/Mrkkew/Awesome-Loop-Transformers)). I checked each one cited below on arXiv. Papers I did not check are listed under Gaps.

Terminology used throughout:
- **State-replacing recurrence:** the hidden state for a position is overwritten each loop. Examples are Huginn, Ouro, Deep Thinking nets, TRM/HRM and PCCoT.
- **Append-only recurrence:** each step adds a new position or KV entry that later steps and the readout attend to. Examples are Coconut, CODI, SIM-CoT and pause tokens.

---

## Q1. Classic and 2021–2025 recurrent-depth models: what happens past the trained iteration count, and what fixes overthinking?

### Takeaway
Running more iterations than trained can improve, preserve or destroy accuracy. Which one happens depends on whether the trained operator settles toward an input-conditioned fixed point. Three things reliably make extra depth safe in state-replacing models: re-injecting the input every step (recall), training that does not let the model count iterations (progressive/incremental loss, randomized or heavy-tailed loop counts), and explicit fixed-point or stability pressure. LM-scale looped models trained at a fixed depth (Ouro, T=4) degrade past it, sometimes badly on reasoning tasks. Huginn, trained with a heavy-tailed random depth (mean 32), mostly saturates instead of degrading on its own benchmarks, although later work finds collapse on procedural probes.

### Cited Findings
**Deep Thinking / easy-to-hard line (convolutional recurrent nets, state-replacing)**
- Recurrent nets trained on easy problems with few iterations can solve harder ones by "thinking for longer," i.e. running more iterations at test time. Tasks were prefix sums, mazes and chess. Schwarzschild et al., "Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks," NeurIPS 2021 [abstract] — [arXiv:2106.04537](https://arxiv.org/abs/2106.04537)
- Bansal et al. (NeurIPS 2022) name the failure **"overthinking."** Recurrent systems run too far past their training regime "often deteriorate," and earlier DT nets extrapolated only from 9×9 to 13×13 mazes [main text] — [arXiv:2202.05826](https://arxiv.org/abs/2202.05826)
- The first fix is the **recall** architecture: the input is concatenated into the recurrent block every iteration, r_recall(φ, x) = r([φ, x]), so the net "cannot forget" the problem [main text] — [arXiv:2202.05826](https://arxiv.org/abs/2202.05826)
- The second fix is the **progressive loss** (incremental-progress training, Alg. 1). Sample n ~ U{0, m−1} and k ~ U{1, m−n}. Run n iterations without gradient, detach, run k more, and compute L = (1−α)·L_max_iters + α·L_progressive. The stated purpose is to discourage the net "from internally counting iterations and learning iteration-specific behaviors" [main text] — [arXiv:2202.05826](https://arxiv.org/abs/2202.05826)
- Bansal et al. numbers [main text] — [arXiv:2202.05826](https://arxiv.org/abs/2202.05826):
  - Prefix sums: trained on 32-bit strings with at most m=30 iterations. Vanilla DT gets 0% on 512-bit strings; DT-Recall + progressive loss gets >97% after about 200 iterations.
  - Mazes: trained on 9×9, the model solves 74% of 201×201 mazes and solves 801×801 mazes with 20,000 iterations.
  - Ablation: fixing n=0 instead of random n lowers 512-bit accuracy from 97.12% to 90.27%. The random restart point matters, not just more depth.
- Mechanism claim (Bansal et al. §5.3): the good models "converge to a fixed point," measured by ‖φ_n − φ_{n−1}‖₂ decreasing, while DT nets without recall "explode." The authors note that fixed-point models "require no learned or hand-crafted stopping rules" [main text] — [arXiv:2202.05826](https://arxiv.org/abs/2202.05826)

**Equilibrium / path independence**
- **Path independence** means converging to the same steady state from any initialization. Anil et al. find it predicts whether an equilibrium model benefits from extra test-time iterations on harder inputs. Interventions that promote it improve upward generalization, and per-example OOD path independence correlates strongly with accuracy. Anil et al., NeurIPS 2022 [abstract] — [arXiv:2211.09961](https://arxiv.org/abs/2211.09961)

**Looped transformers on algorithmic tasks**
- Universal Transformers (Dehghani et al., ICLR 2019) introduced recurrence over depth with dynamic per-position halting. They reported better generalization than standard Transformers on algorithmic tasks where test lengths exceed training lengths [abstract] — [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)
- PonderNet (Banino et al., 2021) learns the number of steps end-to-end and "succeeds at extrapolation tests where traditional neural networks fail" [abstract] — [arXiv:2107.05407](https://arxiv.org/abs/2107.05407)
- Looped Transformers for Length Generalization (Fan, Du, Ramchandran, Lee; ICLR 2025) [main text] — [arXiv:2409.15647](https://arxiv.org/html/2409.15647):
  - Design: the loop count T(n) is tied to input length, there is supervision only at the target step, and input embeddings are re-injected every step.
  - Halting: a "maximum confidence" rule picks the step with the lowest output cross-entropy.
  - Past the needed step, on Addition, Copy, Multiplication and Unique Set the model "somehow learns to converge for a certain number of steps after solving the task," without being trained to. Parity and Binary Sum do not converge, and per-sample stopping works worse on them.
- Looped Transformers are Better at Learning Learning Algorithms (Yang, Lee, Nowak, Papailiopoulos; ICLR 2024). The abstract claims parity with standard transformers on in-context data-fitting at under 10% of the parameters. I did not verify its behavior beyond trained loops (see Gaps) [abstract] — [arXiv:2311.12424](https://arxiv.org/abs/2311.12424)

**HRM / TRM (state-replacing, deep supervision)**
- HRM (Wang et al., 2025) is a two-timescale recurrent model with 27M parameters trained on 1k samples. It is near-perfect on hard Sudoku and large mazes. The abstract says nothing about scaling beyond trained segments [abstract] — [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)
- TRM (Jolicoeur-Martineau, 2025) is a 2-layer, ~7M-parameter recursive model reporting 45% on ARC-AGI-1 and 8% on ARC-AGI-2 [abstract] — [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)
- An independent 2026 measurement shows TRM overthinking on Sudoku-Extreme. TRM peaks at 62.2% at 468 effective layers and falls to 58.1% at 1,872 layers. It turns 20.9% of its correct predictions incorrect at least once and loses 19.7% of solved puzzles after 36 updates. This is reported by the InfiLoop paper, not by the TRM authors (2026 preprint) [main text] — [arXiv:2610.11570](https://arxiv.org/html/2610.11570)

**LM-scale recurrent-depth models**
- Huginn-3.5B (Geiping et al., 2025) [main text, first ~100k characters] — [arXiv:2502.05171](https://arxiv.org/html/2502.05171):
  - Training samples the recurrence count per step from a log-normal Poisson with mean r̄=32 and σ=½. The distribution "most often samples values less than r̄" with a "heavy tail." Gradients flow only through the last k=8 iterations.
  - Test-time saturation is "highly task-dependent." HellaSwag saturates at about 8 iterations. ARC-C saturates at about 8–12 iterations zero-shot and about 32 with 25–50 shots. The EMA model at r=64 gets 47.23% GSM8K (flexible match).
  - The paper reports no explicit inference-time degradation at high r.
  - Latent trajectories: path independence from random s₀; some tokens converge, some fall into "orbits" (e.g. numeric tokens in GSM8K), and some "slide" in one direction.
  - Adaptive exit: stop when the KL divergence between successive steps falls below 5×10⁻⁴. MTBench score is 5.63 without exit and 5.56 with it.
- Conflicting evidence on Huginn: Viakhirev et al. (2026 preprint) classify Huginn's per-token dynamics as "normalized-marginal" (σ_eff = 1.000 ± 0.003) [main text] — [arXiv:2608.18222](https://arxiv.org/html/2608.18222):
  - Huginn exhibits "guess-freezing": answers freeze, but freezing does not track difficulty.
  - On procedural probes, carry accuracy goes 0.31 (r=4) → 0.69 (r=8) → 0.00 (r≥32).
  - So "Huginn saturates gracefully" holds for Geiping et al.'s benchmarks, not for these probes.
- Ouro / LoopLM (Zhu et al., arXiv 2510.25741, v6 revised Oct 2026) is trained with a maximum of **T=4** recurrent steps. Its Table 10 caption says "Performance peaks at the trained depth (T = 4) and then degrades" [main text, PDF] — [arXiv:2510.25741](https://arxiv.org/abs/2510.25741)

  | Model, benchmark | T=4 (trained) | T=8 |
  |---|---|---|
  | 1.4B base, MMLU | 67.45 | 64.49 |
  | 1.4B base, ARC-C | 60.92 | 58.19 |
  | Ouro-2.6B-Thinking, AIME24 | 64.70 | 39.00 |
  | Ouro-2.6B-Thinking, OlympiadBench | 76.44 | 39.26 |
  | Ouro-2.6B-Thinking, AIME25 | 50.30 | 24.33 |
  | Ouro-1.4B-Thinking, AIME24 | 65.00 | 38.67 |

  - The authors conclude that "performance is optimized for the depth seen during training."
  - The exception is HEx-PHI safety, which keeps improving through T=5–8.
- Mixture-of-Recursions (Bae et al., 2025) reports "test-time scaling": raising the maximum recursion depth at inference can improve generation quality. Figure 5c gives only log-likelihood gains, and I did not read exact numbers [main text, partial] — [arXiv:2507.10524](https://arxiv.org/html/2507.10524)
- Relaxed Recursive Transformers (Bae et al., ICLR 2025) relax layer tying with layer-wise LoRA and pair it with early exit and depth-wise batching. The abstract makes no claim about extra test-time recursion [abstract] — [arXiv:2410.20672](https://arxiv.org/abs/2410.20672)
- Retrofitted recurrence (McLeish et al., 2025) converts pretrained LMs to depth-recurrent models with "a curriculum of recurrences" [abstract] — [arXiv:2511.07384](https://arxiv.org/abs/2511.07384). Blayney et al. ran retrofitted Llama for 128 recurrences: each layer reaches a stable state and keeps the same "stages of inference" (2026 preprint) [main text] — [arXiv:2604.11791](https://arxiv.org/html/2604.11791)
- Probing Huginn finds "limited evidence of interpretable latent CoT," and more recurrence depth gives only marginal gains on arithmetic. Lu et al., COLM 2025 workshop [abstract] — [arXiv:2507.02199](https://arxiv.org/abs/2507.02199)

### Inferences
- Every training-side fix above does the same thing: it stops the model from tying its computation to an absolute iteration index. The fixes are random restart points (Bansal), random or heavy-tailed loop counts (Huginn, Kuo et al. in Q2) and length-tied loop counts with many depths supervised (Fan et al.). CODI and SIM-CoT train at exactly 6 latent steps, the Ouro-style fixed-depth regime, which empirically degrades past the trained depth. A **randomized-latent-count or progressive-restart CODI** is therefore the obvious training-side baseline the gate must beat or complement.
- In every successful fix, depth-safety comes with convergence of a *replaced* state. The literature's guarantee is "if the state stops moving, the answer stops changing." That holds only when the readout depends on the current state, not on a growing set of past states (see Q3).

### Gaps
- I did not read Yang et al. (2311.12424) beyond the abstract. My recollection is that they report stability past trained loops when training with a loss window, but this is unverified.
- I did not read the PonderNet parity-extrapolation details or the UT ACT details beyond the abstracts.
- HRM's and TRM's own reports on test-time scaling past trained segments were not checked; only the third-party TRM numbers from InfiLoop are cited.

---

## Q2. 2026 papers: what do the named papers, and the wider 2026 cluster, claim about depth extrapolation? Do they discuss appended-KV growth vs state replacement?

### Takeaway
The five named papers are all real arXiv items (2026 preprints, one NeurIPS 2026). Only "Think Shallow, Solve Deep" and Kuo et al. directly study depth extrapolation. Think Shallow gives a dynamical-regime theory (settling / marginal / drifting) and a fixed-point "depth-safety" condition. Kuo et al. show that randomized loop counts stabilize predictions across inference loop counts. **None of the five discusses KV-cache growth, attention over appended latent entries, Coconut, or CODI.** All of them study state-replacing loops. A large October 2026 cluster adds fixed-point, Jacobian, decay and windowed-memory approaches, but I found none that addresses append-only latent CoT.

### Cited Findings
**Viakhirev, Borodin, Almutairi, Barannikov, Abramov, Mkrtchian — "Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth," arXiv 2608.18222, submitted 18 Aug 2026 (2026 preprint)** [main text, full] — [arXiv:2608.18222](https://arxiv.org/html/2608.18222)
- **Regimes:** operators are classified by the finite-time largest Lyapunov exponent λ_max, estimated with Benettin QR on the top-8 spectrum. Settle is λ_max < −0.05, marginal is |λ_max| ≤ 0.05 and drift is λ_max > 0.05. The claim is that the regime predicts whether extra depth improves, preserves or degrades the answer.
- **Depth-safety (Prop. 1a):** let δ_t = ‖z_{t+1} − z_t‖, R_t = Σ_{u≥t} δ_u, and μ(z_t) be the radius of the largest ball around z_t on which the decoder g is constant. If R_t < μ(z_t), then g(z_s) = g(z_t) for all s ≥ t. The authors stress this guarantees stability, not correctness.
- **Fix:** a "terminal fixed-point objective" penalizes ‖z_T − z_{T−1}‖ = a_T·‖f(z_{T−1}, x) − z_{T−1}‖ at the supervised horizon, where a_T is a learned update gate. They check that the gate does not satisfy the penalty by simply closing.
- **Setup:** training uses 16 recurrent steps with deep supervision over 4 unrolls; test depths are {3, 16, 32, 64, 128}.
- **Sudoku:** past the training horizon, accuracy goes 0.19 → 0.34 (±0.04) on the first unseen tier. Removing the fixed-point term gives 0.01 → 0.03 instead.
- **Huginn repair:** a rank-16 LoRA with "latent anchoring," which pulls deep states toward the base model's r=8 state, restores digit-copy from 0.00 to 1.00 at r=128. However, "No treated model exceeds the base's peak competence at any depth," and answer-agreement halting recovers 0.56 vs 0.74 for an oracle fixed depth on carry. Settling is "necessary but not sufficient" for converting depth into accuracy.
- **Four criteria:** R1 extrapolation by iteration; R2 difficulty-adaptive stabilization that converts; R3 incremental logit-lens rank descent; R4 convergent answer. R4 has a "weak form": a frozen answer while the latent moves in directions the readout ignores.
- **Scope:** no discussion of KV cache, attention over earlier iterations, token-appending recurrence, Coconut or CODI was found in either part of the text.

**Kuo, Chayti, Reizinger, Brendel, Jaggi — "Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping," arXiv 2606.29983, NeurIPS 2026** [main text, first ~100k characters] — [arXiv:2606.29983](https://arxiv.org/html/2606.29983)
- **Setup:** a Fan et al.-style looped block (3 layers, input injection H^(t) = Block(H^(t−1) + H^(0)), NoPE) on binary addition, Dyck-1, Unique Set and Copy. Training uses lengths n < 20 with T_max = 20; OOD is n ∈ {20…60}.
- **Claim:** OOD behavior is brittle, with high run-to-run variance. The authors trace this to a spurious length–loop correlation.
- **Randomized loop counts:** with K = clip(n+Δ, 1, T_max), Δ ~ U{−w…w}, results on addition (Table 1) are:

  | Training schedule | OOD accuracy | Std across runs |
  |---|---|---|
  | K = n | 22.1 | 9.9 |
  | K = n ± 5 | 30.9 | 5.9 |
  | RL-Halt | 45.0 | 2.7 |

  - With a fixed budget, the std falls from 7.4 (K=20) to 4.7 (K=20±5) while OOD accuracy stays at 34.2.
  - w = ±10 does worse than ±5: "too much randomness need not further improve extrapolation."
- **Inference loop counts:** deterministic schedules "often achieve high accuracy only over a narrow range of loop counts, and their predictions can continue changing across many iterations." Randomized training gives "a broader high-accuracy region across K."
- **RL-Halting:** a linear hazard head on pooled states, trained with REINFORCE using reward −loss.
- **Scope:** state is replaced each loop. No KV, appended-token or CODI/Coconut discussion. No fixed-point analysis.

**Fan, Svete, Lee — "Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers" (LOTUS), arXiv 2606.31779, Jun–Jul 2026 (2026 preprint)** [abstract] — [arXiv:2606.31779](https://arxiv.org/abs/2606.31779)
- A looped transformer processes K latent blocks in parallel for R iterations, with per-position CE supervision on gold CoT-step tokens. It claims to be the first latent-CoT method to match explicit CoT at 3B, with 2.5×–6.9× lower thought-phase latency.
- The abstract makes no claims about depth extrapolation or KV growth and does not name Coconut or CODI.
- The latents form a fixed parallel set refined by looping, so this is closer to state-replacing than to CODI's append.

**Blayney, Arroyo, Obando-Ceron, Castro, Courville, Bronstein, Dong — "A Mechanistic Analysis of Looped Reasoning Language Models," arXiv 2604.11791, 13 Apr 2026 (2026 preprint)** [main text, first ~100k characters] — [arXiv:2604.11791](https://arxiv.org/html/2604.11791)
- In many looped LMs "each layer in the cycle converges to a distinct fixed point" (cyclic fixed points). Huginn-0125 and retrofitted Llama converge quickly. Ouro does not, "despite small successive differences."
- Prop. 4.2 bounds the change in softmax attention by the change in the residual stream: stable states imply stable attention.
- "Models that do not reach a fixed point exhibit unstable stages of inference."
- Input injection is needed for non-degenerate fixed points. Without it, only pre-norm converges, and every layer converges to the same fixed point.
- The paper cites Ouro's degradation past training depth and Huginn's flat behavior rather than running new extrapolation experiments.
- No KV-cache or Coconut/CODI discussion was found. All models analyzed are state-replacing.

**Kaissis, Mildenberger, Gomez, Menten, Triantafillou — "Step-resolved data attribution for looped transformers," arXiv 2602.10097, 10 Feb 2026 (2026 preprint)** [abstract] — [arXiv:2602.10097](https://arxiv.org/abs/2602.10097)
- Step-Decomposed Influence splits TracIn into per-iteration influence, using TensorSketch for scale.
- No depth-extrapolation, fixed-point or KV claims. Peripheral; it could serve as a tool to attribute *which latent step* training data shapes.

**Other 2026 depth-extrapolation papers (all 2026 preprints unless noted)**
- "Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models" (STARS). Yang, Han, Zhang, Wei, Shao, Guo, Li; ICML 2026 per the comments field [abstract] — [arXiv:2605.26733](https://arxiv.org/abs/2605.26733)
  - LoopLM performance "often peaks at a certain iteration depth and then collapses with further recurrence."
  - The fix is Jacobian spectral-radius regularization plus random loop sampling, pushing states toward asymptotically stable fixed points. It "substantially mitigates performance degradation" on math.
  - This is probably the "Yang et al. (2026) spectral-radius regularization" that Think Shallow cites as the control that helps at scale. That match is my inference, not confirmed.
- "Fixed-Point Reasoners" (FPRM). Movahedi et al., 16 Jun 2026 [abstract] — [arXiv:2606.18206](https://arxiv.org/abs/2606.18206)
  - Uses fixed-point convergence as an end-to-end halting rule, with pre-norm and residual scaling for stability.
  - Evaluated on Sudoku, Maze, state tracking and ARC-AGI. No extrapolation numbers in the abstract.
- "Stability and Generalization in Looped Transformers." Labovich, Apr 2026 [abstract] — [arXiv:2604.15259](https://arxiv.org/abs/2604.15259)
  - Without recall, looped nets have countable fixed points and "cannot achieve strong input-dependence."
  - Recall plus outer normalization gives reachable, input-smooth fixed points.
  - Proposes "internal recall."
- "Parcae: Scaling Laws for Stable Looped Language Models." Prairie, Novack, Berg-Kirkpatrick, Fu, Apr 2026 [abstract] — [arXiv:2604.12946](https://arxiv.org/abs/2604.12946)
  - Attributes instability to large spectral norms of the injection parameters and constrains them with a negative-diagonal parameterization.
  - Test-time looping returns follow a "saturating exponential decay."
- "Learning to Extrapolate in Depth with Looped Transformers." Yang, Huang, Wang, Mirzasoleiman, 10 Oct 2026 [abstract] — [arXiv:2610.14456](https://arxiv.org/abs/2610.14456)
  - Extrapolation holds only within a finite depth window on graph tasks: "too few test-time loops provide insufficient propagation, while too many eventually create false connections."
  - Derives explicit window endpoints.
- "Beyond the Training Horizon: Mechanisms and Limits of Length Generalization in Looped Transformers." Liang, Jin, Pan, 27 Sep 2026 [abstract] — [arXiv:2609.33144](https://arxiv.org/abs/2609.33144)
  - Compares two schemes. MR-Loop updates a state at a fixed readout position. DR-Loop passes states across positions as a "moving computational frontier."
  - Both generalize somewhat past the horizon but become unreliable at greater depth, and local errors compound.
- "Right Direction, Wrong Step." Guo et al., 15 Sep 2026 [abstract] — [arXiv:2609.16665](https://arxiv.org/abs/2609.16665)
  - Some extra-iteration failures are overshoot: "a locally improving direction produces a harmful full update."
  - A fixed quarter step gives positive gains for 72.2–83.2% of selected failures.
- "Dynamical phase selection controls compute scaling in looped transformers." Kim, 27 Aug 2026 [abstract] — [arXiv:2608.26556](https://arxiv.org/abs/2608.26556)
  - Identically trained nets land in different phases, a saddle-node fold or a Neimark–Sacker transition, depending on initialization.
  - "Test-time compute is not determined by architecture alone."
- "Invariant-Measure Reasoners." Inui, Konishi, Kawahara, 7 Oct 2026 [abstract] — [arXiv:2610.10996](https://arxiv.org/abs/2610.10996)
  - Latents may settle into a compact region without a fixed point, so predictions vary with depth.
  - The fix is to average the head's output over the invariant measure. It can be done by fine-tuning only the prediction head.
- "Understanding Dynamic Compute Allocation in Recurrent Transformers" (ANIRA). Moosa et al., Feb 2026 [abstract] — [arXiv:2602.08864](https://arxiv.org/abs/2602.08864)
  - Learned token-wise depth aligns with complexity, but "models fail to extrapolate to unseen input sizes despite allocating additional computation."
- "Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals" (InfiLoop). Li, Muhtar, He, Su, Yin, Liu, Oct 2026 [main text] — [arXiv:2610.11570](https://arxiv.org/html/2610.11570). See Q4 for detail.
  - Defines "latent overthinking": "noisy state updates overwrite correct intermediate deductions and even undo completed solutions."
  - A learned-decay residual over past loop states keeps improving on Sudoku-Extreme to about 25k layer evaluations.

### Inferences
- Think Shallow's Prop. 1a and Blayney's Prop. 4.2 both quietly assume the readout is a function of the *current* state with a *fixed* context. Your SIM-CoT result is a counterexample to "settling ⇒ depth-safe" in the append-only setting: the latent sits at a fixed point, yet the decoded answer changes because the context grows.
  - Why a fixed point is not enough: with N identical copies of a key with logit s among other keys with logits s_j, softmax mass on the copies is N·e^s / (N·e^s + Σ_j e^{s_j}). That tends to 1 as N grows. So a fixed-point latent, appended repeatedly, *monotonically crowds out* the question and earlier-latent entries.
  - This is a readout drift with zero state drift. The depth-safety condition would need R_t to include the change in the attention output, not just ‖Δz‖.
  - This is my derivation, not a cited result, but it matches your exact-KV-copy experiment.
- Think Shallow's "weak R4" (frozen answer while the latent moves in readout-ignored directions) and ImR (averaging over an invariant measure) are both *readout-side* invariance ideas. Your gate is also readout-side: it bounds the set the readout sees. These are the conceptually nearest 2026 framings.
- Think Shallow's Huginn result parallels your anchoring experience. Anchoring deep states to a good shallow state restores stability ("depth-safety"), but "no treated model exceeds the base's peak competence." A gate that bounds the latent set should likewise be expected to *preserve* step-6 accuracy at depth 24, not to *gain* from extra steps. The R2/R3 "conversion" criteria are a separate and harder bar.

### Gaps
- None of the named 2026 papers was checked for an appendix discussion of KV caching beyond the portions read. For Kuo et al. and Blayney et al. only the first ~100k characters of the HTML were parsed.
- LOTUS was read at abstract level only. Whether it tests R > R_train is unknown.
- I could not confirm the identity of "EqR (Huang et al. 2026)," which Think Shallow cites.

---

## Q3. Does any work distinguish state-replacing recurrence from append-only recurrence with respect to depth extrapolation?

### Takeaway
I found **no paper that directly compares state-replacing vs append-only recurrence on depth extrapolation, or that identifies attention reweighting over a growing latent KV set as the cause of degradation.** The nearest material falls into three groups:
1. Theory separating fixed-size loops from growing scratchpads in *expressivity*, not extrapolation stability.
2. Latent-CoT methods that replace CODI's left-to-right append with a fixed latent set refined in parallel (PCCoT on CODI/GPT-2, LLoCoT, LOTUS). These avoid cache growth but report non-convergence or fluctuation.
3. Append-only systems with mild robustness to extra steps (pause tokens) or instability when adding more (Coconut).

### Cited Findings
- **Memory-budget separation (theory).** "Chain-of-Thought and Compressed Looped Transformers: A Memory-Budget Separation." Zhang, 29 May 2026 (2026 preprint) [abstract] — [arXiv:2605.30757](https://arxiv.org/abs/2605.30757)
  - A compressed loop is capped by its recurrent-state size. Extra iterations add compute but not persistent memory.
  - CoT's appended tokens add persistent state, so polynomial CoT can decide logspace-P-complete problems that compressed loops cannot, under standard assumptions.
  - "Full sequence-state loops" behave more like scratchpads.
- **PCCoT, state-replacing latent CoT on the CODI recipe.** Wu, Teng, Tu, "Parallel Continuous Chain-of-Thought with Jacobi Iteration," EMNLP 2025 [main text] — [arXiv:2506.18582](https://arxiv.org/html/2506.18582)
  - A fixed set of c latent positions is overwritten each Jacobi iteration rather than appended. Each iteration reruns the forward pass over question + `<bot>` + c latents.
  - Setup: GPT-2 Small and Llama-3.2-1B with LoRA, trained CODI-style, with c=24 and T=3.
  - GSM8K: GPT-2 49.48 ± 0.31 vs CODI 43.7 vs Coconut 34.1; Llama-1B 53.35 vs CODI 55.6.
  - "Increasing the number of iterations does not necessarily improve the performance." Results above about T=6 "fluctuate heavily."
  - After training, the latents "no longer converge and stays fluctuating after a certain number of iterations," even though Theorem 1 guarantees a fixed point at T ≥ c in the continuous-CoT-equivalent case.
  - Caveat: the T sweeps appear to be per-trained-T configurations, not a pure inference-only extrapolation test.
- **LLoCoT.** García et al., "From Chain-of-Thought to Loops: Non-Autoregressive Latent Reasoning via Looped Transformers," 8 Oct 2026 (2026 preprint) [abstract] — [arXiv:2610.11472](https://arxiv.org/abs/2610.11472)
  - Argues autoregressive latent methods "retain a left-to-right dependency among latent vectors." LLoCoT instead refines a fixed set of latent slots with a shared looped transformer.
  - Evaluated on HumanEval/MBPP only, with about 42× lower reasoning-phase latency vs reasoning SFT.
  - No depth-extrapolation or KV claims in the abstract.
- **LOTUS** (2026 preprint) uses K parallel latent blocks refined for R iterations, i.e. a fixed set and not an append [abstract] — [arXiv:2606.31779](https://arxiv.org/abs/2606.31779)
- **"Trading Depth for Time in Recurrent Transformers."** Huang, He, Shen, Lee, Sep 2026 (2026 preprint) [abstract] — [arXiv:2609.21605](https://arxiv.org/abs/2609.21605)
  - Explicitly frames *thought tokens* as adding positions in time and *loops* as adding depth.
  - With one latent thought token between vocabulary tokens, the model overtakes a two-loop transformer as compute grows under matched FLOPs.
  - No extrapolation or KV analysis in the abstract.
- **"Beyond the Training Horizon."** (2026 preprint) [abstract] — [arXiv:2609.33144](https://arxiv.org/abs/2609.33144)
  - MR-Loop (state at a fixed readout position) vs DR-Loop (state passed across positions) is the closest within-looped-transformer analogue of "replace in place" vs "move forward along positions."
  - Both degrade at greater depth, through different mechanisms: readout and progress-cue decay vs unreliable state propagation.
- **Pause tokens (append-only, non-state-derived).** Goyal et al., ICLR 2024 [main text] — [arXiv:2310.02226](https://arxiv.org/html/2310.02226)
  - With M_ft = 10 pauses at finetuning, models "degrade gracefully to shifts as wide as M_inf ∈ [5,25]." Zero pauses at inference causes "a drastic drop."
  - The authors note that M_inf > M_ft could in principle produce garbage, but it did not in the tested range. Appendix F reports a GSM8K drop for M_ft = 50.
- **Coconut (append-only, state-derived).** Hao et al., 2024 [main text] — [arXiv:2412.06769](https://arxiv.org/html/2412.06769)
  - Each thought's last hidden state is fed back as the next input embedding, so the sequence grows.
  - Raising thoughts per step c from 0 to 2 improved GSM8K. At c=3 there was "a slight performance drop accompanied by increased variance," attributed to a training-loss spike at stage transitions.
  - No inference-only extrapolation beyond the trained count was reported.
- **Other analyses of CODI latent steps (search snippets only, not read).** A 2026 GTS paper reports that CODI's "deterministic strength peaks at the second latent step and then transitions into a sustained lower-SNR regime." A 2026 faithfulness paper finds that ablating continuous thoughts largely preserves predictions in trained COCONUT/CODI [search snippet only] — [arXiv:2602.14077](https://arxiv.org/pdf/2602.14077); [arXiv:2607.06648](https://arxiv.org/pdf/2607.06648)

### Inferences
- The append-only failure mode is mechanistically different from classic overthinking. Classic overthinking is the *state* drifting (λ_max > 0, explosion without recall). In append-only latent CoT the state can settle while the *readout's attention set* grows and redistributes probability mass. That makes your "exact KV copies of the step-6 latent reproduce the whole drop" control a distinguishing experiment the literature lacks.
- Two existing alternatives bracket your gate:
  - **Avoid growth entirely** with a fixed-size latent set refined in place (PCCoT, LLoCoT, LOTUS). Costs: you lose the scratchpad's memory advantage (Zhang 2605.30757), and PCCoT's latents empirically fluctuate rather than converge.
  - **Keep growth but make the readout invariant** to it, which is your gate. The gate is a middle ground: persistent memory with a bounded active set.
- Pause tokens degrade gracefully under extra appended entries, while Coconut destabilizes. One plausible reason is that pause-token KVs are position-distinct fixed embeddings, while state-derived latents at a fixed point produce near-*duplicate* keys. Duplicate keys are exactly the case where softmax mass accumulates. This is speculative; it could be tested by comparing appended fixed-point latents vs appended pause tokens in CODI.

### Gaps
- I found no paper that runs CODI, SIM-CoT or Coconut at more latent steps than trained and reports the result, apart from your own unpublished data. Targeted searches returned nothing direct.
- SIM-CoT and CODI were not re-read for latent-count ablations. The KaVa paper's mention of CODI/Coconut step dependence was seen only as a search snippet — [arXiv:2510.02312](https://arxiv.org/abs/2510.02312)
- "Internalizing Chain-of-thought into Looped Transformers" (listed as arXiv 2610.25998) was not verified.

---

## Q4. Analyses of attention to earlier loop iterations, and of the KV cache across loops

### Takeaway
In mainstream looped LMs, each recurrence of a token writes its own KV entries for *other* tokens to read. Attention generally does not span the same token's earlier iterations. Several efficiency results show the readout tolerates, or even benefits from, seeing **one bounded set** of per-token states:
- Ouro: last-step-only KV at decode costs ≤0.07 GSM8K points; first-step-only collapses.
- Huginn: modular KV budget, no loss on MTBench.
- Shared-memory LPT: only the first recursion writes KV, plus a short local window; better perplexity at 76–79% less memory.

Two 2026 architectures do attend over a token's own *past loop states*, and both find that **bounding or decaying that set is essential**:
- RecurTrace: gated sliding window, W=3 better than W=5.
- InfiLoop: learned temporal decay. Removing the decay collapses Sudoku-Extreme accuracy from 82–90% to 38–40%.

These are the closest existing evidence for your gate hypothesis, though in state-replacing loops rather than appended latent CoT.

### Cited Findings
- **Huginn KV sharing (§6.2).** [main text] — [arXiv:2502.05171](https://arxiv.org/html/2502.05171)
  - Set a per-token KV budget k. At recurrence i, read and write slot i mod k, so step 17 overwrites step 1's entry with a budget of 16.
  - MTBench is 5.63 standard vs 5.86 shared. The budget is printed as "44," likely a typo for 4.
  - Remark 6.1: with early exit, tokens attend to "the deepest available KV entries" of earlier tokens, which is possible because all recurrences share K/V projections.
- **Ouro KV reuse (§5.4.2, Table 14).** [main text, PDF] — [arXiv:2510.25741](https://arxiv.org/abs/2510.25741)
  - Prefill needs all 4 per-step caches; reusing them costs ">10 points on GSM8K."
  - Decode results:

    | Decode strategy | GSM8K | MATH-500 | Memory |
    |---|---|---|---|
    | Full (4 caches) | 78.92 | 82.40 | 1× |
    | Last-step only | 78.85 | 80.40 | 4× smaller |
    | Averaged | 78.73 | 78.52 | 4× smaller |
    | First-step only | 18.73 | 8.43 | 4× smaller |

  - Authors' reading: "the final recurrent step's representations are most informative for subsequent token generation."
- **Mixture-of-Recursions.** [main text, partial] — [arXiv:2507.10524](https://arxiv.org/html/2507.10524)
  - *Recursion-wise caching*: "only tokens routed to a given recursion step store their key–value entries at that level," and attention is restricted to them.
  - *Recursive KV sharing*: KV is cached only at the first recursion and reused, so keys and values always span the full sequence.
  - Expert-choice, Nr=3: caching gives NLL 2.7925 / 42.6% few-shot vs sharing 2.7983 / 41.9%.
- **Shared memory in looped transformers (LPT).** Monea, Ramji, El-Kurdi, Lastras, Artzi, Godey, Astudillo, 1 Oct 2026 (2026 preprint) [abstract] — [arXiv:2610.02383](https://arxiv.org/abs/2610.02383)
  - Only the first recursion writes a KV cache. Later recursions read it plus "a short window of their own local memory" and "attend mostly to the shared memory."
  - With 5 recursions, FineWeb-Edu perplexity drops 1.12–1.82 vs a same-size transformer at 76–79% less context memory.
- **Looped Latent Attention.** O'Neill & Reid, Jul 2026 (2026 preprint) [abstract] — [arXiv:2607.15456](https://arxiv.org/abs/2607.15456)
  - The loop-indexed KV "trace a short low-rank trajectory across loops." The cache is "low-rank but not safely collapsible to a single state."
  - Per-head LLA beats final-loop reuse at matched budget. Note the tension with Ouro's near-lossless last-step reuse.
- **Blayney et al.** Attention patterns converge across recurrences once fixed points are reached. Each layer's attention is most similar to itself at other recurrences, and Ouro stabilizes after the first iteration. Prop. 4.2 bounds attention change by residual change. No KV-cache analysis (2026 preprint) [main text] — [arXiv:2604.11791](https://arxiv.org/html/2604.11791)
- **RecurTrace — "Adaptive Latent Reasoning with Loop-Time Memory."** Sep 2026 (2026 preprint) [main text, partial] — [arXiv:2609.03379](https://arxiv.org/html/2609.03379)
  - Each looped layer attends over "a sliding window of the layer's own outputs from up to W earlier loops," along loop time only, per position.
  - A learnable per-head recency bias −β_h(t − t_j), initialized to ALiBi, penalizes older loops.
  - A scalar gate γ and a token-wise gate g can "close the memory." The window "caps the cost of remembering at a constant."
  - Window ablation over W ∈ {1, 2, 3, 5}: W=3 is best.
  - Training samples loop counts from a clamped log-normal Poisson (mean ≈3.9, 8-loop clamp). At 16 loops accuracy "stays stable" at 47.5%, below the 54.7% two-loop peak.
  - No analysis of attention weights over past iterations (dilution or recency) is reported.
- **InfiLoop.** (2026 preprint) [main text] — [arXiv:2610.11570](https://arxiv.org/html/2610.11570)
  - Weights over past loop states z_i: α_{t,i} ∝ exp(qᵀRMSNorm(z_i) − λ(t−i)), with a streaming form at O(d) memory per token. Equivalently h_t = h_{t−1} + γ_t(z_t − h_{t−1}).
  - Decay ablation (Table 2): λ=0 ("Full Attention Residuals," no decay) gives **38.4% / 40.2%** vs **82.1% / 89.7%** with learned decay on H3L6 / H32L6.
  - Training: H=3, L=6, 12 deep-supervision segments. Test: up to 160 segments, about 24,960 layer evaluations.
  - The paper argues full attention over all past representations costs O(td) but does not frame λ=0 as "dilution."
- **Adjacent explicit-CoT KV eviction (not loops).**
  - LightThinker (EMNLP 2025) compresses thought steps into gist tokens and "discards the original reasoning chains" through attention masks [abstract] — [arXiv:2502.15589](https://arxiv.org/abs/2502.15589)
  - KaVa (ICLR 2026, per search snippet) compresses a teacher CoT's KV with redundancy-importance eviction to a latent budget and distills it into a latent-reasoning student [search snippet] — [arXiv:2510.02312](https://arxiv.org/abs/2510.02312)
  - IndexMem uses a learned gate that can fall back to attending only over retained entries [search snippet] — [arXiv:2605.25475](https://arxiv.org/pdf/2605.25475)

### Inferences
- InfiLoop's λ=0 ablation is the strongest published evidence for your mechanism. It is not standard attention, but an aggregation over an ever-growing set of past iteration states without recency/gating. It halves accuracy, and adding a learned decay (soft eviction) restores it and enables extreme depth.
- RecurTrace's W=3 beating W=5 says the same for a hard window plus gates.
- Together they **partially pre-empt the general claim** that "bounding the set of past-iteration states the model attends to fixes depth drift." Your novelty must rest on three things:
  1. The append-only latent-CoT setting, where the entries are in the main KV cache and also seen by the answer readout.
  2. The causal diagnosis via exact-KV-copy controls.
  3. A learned eviction gate on latent KV, as opposed to a decay or window over a per-position residual.
- Ouro's last-step-only decode result is a cheap, strong baseline for your gate. It amounts to "evict all but the latest latent" for the readout. Huginn's i mod k scheme is a ready-made "bounded ring buffer" baseline.

### Gaps
- RecurTrace was read only up to ~100k characters. I found no attention-weight-over-iterations analysis in that part.
- Listed but unverified (only the reading-list summary seen, not arXiv), so they should not be cited until checked:
  - "LSTM-Style Looped Transformers with Shared KV Caches" (2610.33360)
  - "BeNice! Closing Looping's Inference-Memory Gap via KV-Sharing" (2610.31633)
  - "Enabling Dynamic Computation in Looped LMs" (2610.09013)
  - "T-LoopFormer" (2609.15160)
  - "Cross-Loop Query Handoff… Ouro" (2610.34639)
  - "Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers" (2607.14427)
  - "Looped Reasoning Finishes Earlier Than You Think" (2610.18174)
  - "Do Halting Heads Really Save Compute in Looped Language Models?" (2610.32246)
  - "Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates…" (2607.20519)
  - Source for all of the above: [Awesome-Loop-Transformers](https://github.com/Mrkkew/Awesome-Loop-Transformers)
- An alphaxiv link for an "LSTM-style looped transformer" carried a malformed ID, so I treated it as unreliable.
- I found no work measuring how attention mass on question tokens changes as latent or loop entries accumulate in an appended cache.

---

## Q5. Per-paper citations, one-line relevance, and tags (supports / motivates / competes / threatens novelty). Which are the closest competitors?

### Takeaway
**Closest competitors, ranked:**
1. **InfiLoop** (2610.11570): learned-decay aggregation over past loop states; no-decay collapses; extreme depth.
2. **RecurTrace** (2609.03379): gated, recency-biased, windowed attention over past loop states.
3. **Think Shallow, Solve Deep** (2608.18222): the depth-safety theory your result extends or refutes for append-only models.
4. **PCCoT** (2506.18582): a state-replacing alternative built on CODI with GPT-2.
5. **Kuo et al.** (2606.29983) and **STARS** (2605.26733): training-side stability fixes.
6. **Ouro / Huginn / LPT** bounded-KV schemes: efficiency-motivated bounded readout sets.

None combines append-only latent CoT, attention reweighting as the diagnosed cause, and a learned latent-KV eviction gate.

### Cited Findings
| Paper (authors, year/venue) | Link | Status | One-line claim relevant to the gate idea | Tag |
|---|---|---|---|---|
| Viakhirev et al., "Think Shallow, Solve Deep," 2026 preprint | [2608.18222](https://arxiv.org/abs/2608.18222) | main text | Settling regime + terminal fixed-point loss make extra depth safe; theory assumes the readout depends on the current state only, no KV-growth discussion | motivates (your data is a counterexample in append-only); partial competes |
| Kuo, Chayti, Reizinger, Brendel, Jaggi, NeurIPS 2026 | [2606.29983](https://arxiv.org/abs/2606.29983) | main text (partial) | Randomized loop counts and RL-halting stabilize predictions across inference loop counts | competes (training-side baseline: randomize CODI latent count) |
| Fan, Svete, Lee, "LOTUS," 2026 preprint | [2606.31779](https://arxiv.org/abs/2606.31779) | abstract | Looped parallel latent blocks match explicit CoT at 3B; no extrapolation claims | competes (fixed-set latent CoT avoids growth) |
| Blayney et al., "Mechanistic Analysis of Looped Reasoning LMs," 2026 preprint | [2604.11791](https://arxiv.org/abs/2604.11791) | main text (partial) | Cyclic fixed points ⇒ stable attention (Prop. 4.2), assuming fixed context; non-converging Ouro has unstable stages | supports (stable states ⇏ stable attention once context grows) |
| Kaissis et al., "Step-resolved data attribution for looped transformers," 2026 preprint | [2602.10097](https://arxiv.org/abs/2602.10097) | abstract | Per-iteration influence (SDI); no depth claims | peripheral (tool) |
| Geiping et al., Huginn, 2025 | [2502.05171](https://arxiv.org/abs/2502.05171) | main text (partial) | Heavy-tailed random depth; task-dependent saturation; KV ring buffer (i mod k) with no MTBench loss | supports; competes (bounded-KV baseline) |
| Zhu et al., Ouro / LoopLM, 2025 (v6 2026) | [2510.25741](https://arxiv.org/abs/2510.25741) | main text | Fixed T=4 training degrades past T=4 (AIME24 64.7→39.0 at T=8); decode with last-step KV only is near-lossless | motivates; competes (last-step-only baseline) |
| Bansal et al., NeurIPS 2022 | [2202.05826](https://arxiv.org/abs/2202.05826) | main text | Overthinking; recall + progressive loss ⇒ fixed points, 32→512-bit, 9×9→201×201 | supports (iteration-agnostic training); competes (progressive-restart CODI baseline) |
| Schwarzschild et al., NeurIPS 2021 | [2106.04537](https://arxiv.org/abs/2106.04537) | abstract | Easy-to-hard by thinking longer | background |
| Anil et al., NeurIPS 2022 | [2211.09961](https://arxiv.org/abs/2211.09961) | abstract | Path independence predicts gains from extra test-time compute | supports |
| Dehghani et al., Universal Transformers, ICLR 2019 | [1807.03819](https://arxiv.org/abs/1807.03819) | abstract | Depth recurrence + per-position halting | background |
| Banino et al., PonderNet, 2021 | [2107.05407](https://arxiv.org/abs/2107.05407) | abstract | Learned halting; extrapolation where standard nets fail | background |
| Fan, Du, Ramchandran, Lee, ICLR 2025 | [2409.15647](https://arxiv.org/abs/2409.15647) | main text | Length-tied loops; models "converge" after solving on some tasks, not parity | supports |
| Yang, Lee, Nowak, Papailiopoulos, ICLR 2024 | [2311.12424](https://arxiv.org/abs/2311.12424) | abstract | Looped transformers for in-context learning | background (extrapolation details unverified) |
| Bae et al., Mixture-of-Recursions, 2025 | [2507.10524](https://arxiv.org/abs/2507.10524) | main text (partial) | Recursion-wise caching vs recursive KV sharing; test-time deeper recursion helps likelihood | competes (bounded KV design) |
| Bae et al., Relaxed Recursive Transformers, ICLR 2025 | [2410.20672](https://arxiv.org/abs/2410.20672) | abstract | Layer-wise LoRA for recursion; early exit | background |
| Wang et al., HRM, 2025 | [2506.21734](https://arxiv.org/abs/2506.21734) | abstract | Two-timescale recurrence for Sudoku/maze/ARC | background |
| Jolicoeur-Martineau, TRM, 2025 | [2510.04871](https://arxiv.org/abs/2510.04871) | abstract | Tiny recursive model; overthinks per InfiLoop (62.2→58.1%) | motivates |
| McLeish et al., Retrofitted recurrence, 2025 | [2511.07384](https://arxiv.org/abs/2511.07384) | abstract | Curriculum of recurrences converts pretrained LMs | background |
| Lu et al., Decoding the depth-recurrent transformer, COLM-W 2025 | [2507.02199](https://arxiv.org/abs/2507.02199) | abstract | Limited interpretable latent CoT in Huginn; marginal gains from more depth | motivates |
| Yang et al., STARS, ICML 2026 | [2605.26733](https://arxiv.org/abs/2605.26733) | abstract | LoopLMs peak then collapse; Jacobian spectral-radius reg + random loops fixes it | competes (training-side stability) |
| Movahedi et al., FPRM, 2026 preprint | [2606.18206](https://arxiv.org/abs/2606.18206) | abstract | Fixed-point convergence as halting | competes (weak) |
| Labovich, Stability & Generalization, 2026 preprint | [2604.15259](https://arxiv.org/abs/2604.15259) | abstract | Recall + outer norm ⇒ reachable input-dependent fixed points | background |
| Prairie et al., Parcae, 2026 preprint | [2604.12946](https://arxiv.org/abs/2604.12946) | abstract | Spectral-norm-constrained injection; saturating test-time returns | background |
| Yang, Huang, Wang, Mirzasoleiman, 2026 preprint | [2610.14456](https://arxiv.org/abs/2610.14456) | abstract | Finite safe depth window; too many loops create "false connections" | supports |
| Liang, Jin, Pan, "Beyond the Training Horizon," 2026 preprint | [2609.33144](https://arxiv.org/abs/2609.33144) | abstract | MR-Loop vs DR-Loop both degrade at depth via different mechanisms | supports / nearest "replace vs propagate" comparison |
| Guo et al., "Right Direction, Wrong Step," 2026 preprint | [2609.16665](https://arxiv.org/abs/2609.16665) | abstract | Overshoot; quarter-step recovers 72–83% of failures | background |
| Kim, "Dynamical phase selection," 2026 preprint | [2608.26556](https://arxiv.org/abs/2608.26556) | abstract | Init selects dynamical phase, which sets compute scaling | background |
| Inui et al., Invariant-Measure Reasoners, 2026 preprint | [2610.10996](https://arxiv.org/abs/2610.10996) | abstract | Readout averaged over invariant measure ⇒ depth-stable predictions; head-only fine-tune | competes (readout-side invariance) |
| Moosa et al., ANIRA, 2026 preprint | [2602.08864](https://arxiv.org/abs/2602.08864) | abstract | More allocated compute does not yield size extrapolation | background |
| Li et al., InfiLoop, 2026 preprint | [2610.11570](https://arxiv.org/abs/2610.11570) | main text | Learned-decay residual over past loop states; λ=0 (no decay) 82→38%; >20k steps | **threatens novelty (general claim)**; supports mechanism |
| RecurTrace, 2026 preprint | [2609.03379](https://arxiv.org/abs/2609.03379) | main text (partial) | Gated, recency-biased sliding window (W=3 best) over past loop states | **closest competitor (architecture)** |
| Monea et al., Shared memory in looped transformers (LPT), 2026 preprint | [2610.02383](https://arxiv.org/abs/2610.02383) | abstract | First recursion writes KV; later loops read it + short local window; better quality at less memory | competes (bounded KV) |
| O'Neill & Reid, Looped Latent Attention, 2026 preprint | [2607.15456](https://arxiv.org/abs/2607.15456) | abstract | Loop KV low-rank but "not safely collapsible to a single state" | caution vs last-step-only |
| Zhang, CoT vs Compressed Loops memory separation, 2026 preprint | [2605.30757](https://arxiv.org/abs/2605.30757) | abstract | Appended tokens add persistent memory that loops lack | motivates (a gate should be learned, not a hard cap) |
| Wu, Teng, Tu, PCCoT, EMNLP 2025 | [2506.18582](https://arxiv.org/abs/2506.18582) | main text | State-replacing latent CoT on CODI/GPT-2: 49.5% GSM8K; latents fluctuate, more iterations not better | **competes (alternative fix)** |
| García et al., LLoCoT, 2026 preprint | [2610.11472](https://arxiv.org/abs/2610.11472) | abstract | Fixed latent workspace refined by loops, not left-to-right | competes |
| Huang, He, Shen, Lee, "Trading Depth for Time," 2026 preprint | [2609.21605](https://arxiv.org/abs/2609.21605) | abstract | Thought tokens (time) vs loops (depth) under matched FLOPs | background |
| Goyal et al., Pause tokens, ICLR 2024 | [2310.02226](https://arxiv.org/abs/2310.02226) | main text | Graceful under 5–25 inference pauses when trained with 10 | supports (contrast case) |
| Hao et al., Coconut, 2024 | [2412.06769](https://arxiv.org/abs/2412.06769) | main text | More thoughts per step: c=3 drops, higher variance | motivates |
| Zhang et al., LightThinker, EMNLP 2025 | [2502.15589](https://arxiv.org/abs/2502.15589) | abstract | Compress thoughts to gist tokens, discard originals | competes (mechanism, explicit CoT) |
| Kuzina et al., KaVa, ICLR 2026 | [2510.02312](https://arxiv.org/abs/2510.02312) | search snippet | Evicted/compressed teacher KV distilled into latent reasoning | competes (weak) |
| Lin et al., MixerLoop, 2026 preprint | [2608.18230](https://arxiv.org/abs/2608.18230) | abstract | What to loop (mixers vs FFNs) | peripheral |

### Inferences
- **Novelty assessment.**
  - *Not novel:* "bounded or decayed access to past-iteration states improves depth robustness" (InfiLoop, RecurTrace), and "fixed points give depth safety in state-replacing loops" (Bansal, Anil, Think Shallow, STARS).
  - *Apparently open (no counter-example found):*
    1. Showing that in **append-only latent CoT** a **settled latent still causes readout drift** through attention-mass redistribution (the exact-KV-copy control).
    2. Fixing it with a **learned gate that evicts latent KV entries** so the answer readout sees a bounded set, evaluated on GSM8K against CODI and SIM-CoT.
- **Baselines a reviewer will expect:**
  - Ouro-style "last-latent-only" readout.
  - Huginn-style i mod k ring buffer.
  - A RecurTrace-style fixed window with recency bias.
  - InfiLoop-style learned decay.
  - Randomized latent count in training (Kuo et al.; Huginn log-normal Poisson).
  - Bansal-style progressive restart.
  - PCCoT (no growth).

### Gaps
- Several 2026 items are abstract-only, so their main-text treatment of KV caches is unknown. The most important to read next are RecurTrace's remainder, LPT and Looped Latent Attention.
- I did not check whether any 2026 paper evaluates CODI or SIM-CoT at more latent steps than trained.
