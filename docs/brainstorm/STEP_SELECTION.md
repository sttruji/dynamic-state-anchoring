# Step-level selection over sampled latent steps: quick reference

A summary from the dsa-brainstorm session, 6 November 2026. It follows the nulls of the inner loop (A) and KK-LSE (F) in
`results/inner_loop/README.md`, and it grew out of the ranking in
[Latent reasoning limitations ranked](Latent%20reasoning%20limitations%20ranked.md).

## The idea in brief

Most test-time scaling for latent reasoning samples several whole chains and picks one finished answer. This idea makes
the choice **at every latent step** instead:

1. At latent step s, propose K candidate executions of that step. Candidate 0 is the model's own step, with dropout off.
   The others rerun the step with MC dropout. A candidate is everything the step produces: its cache entries and the next
   latent.
2. A selector scores the K candidates and keeps one, and only that candidate is fed forward.
3. The selector is trained with RL, rewarded only by whether the final answer is right. The SIM-CoT generator stays
   frozen, so only the small selector learns.

Two things set it apart from prior work:
- **SIM-CoT's step decoder can read every candidate.** It turns each latent into a `<<a op b=c>>` step, so the selector can
  check the arithmetic, whether the operands are grounded in the question or earlier steps, and how confident the decoder
  is. Prior work says latents "cannot be interpreted or scored by existing PRMs" (process reward models).
- **The start is an identity.** A selector that always keeps candidate 0 is the released checkpoint, exactly.

## Why choose per step

- **Plans go wrong early.** In Stage 0, 188 of the 296 wrong answers were plan errors, not arithmetic slips. The decoder
  reads latents back well only at slots 0–2: exact steps fall from 199 at slot 0 to 21 at slot 2, then to about 0.
  Choosing among finished answers can't repair a plan that went wrong at step 1 in every sample.
- **Search is cheaper.** K candidates at each of 6 steps cost about K×6 latent steps, yet reach combinations that
  independent chains would need far more samples to find.
- **Explicit CoT shows it works.** Process advantage verifiers score each step by how much it raises the chance of a
  correct answer. Beam search with them is 8–10% more accurate, and 1.5–5× more compute-efficient, than choosing among
  finished answers with an outcome reward model (Setlur et al., ICLR 2025).

## Papers around the idea

| Paper | What it does | GPT-2 result, and what it leaves open |
|---|---|---|
| [LatentTTS](https://arxiv.org/abs/2510.07745), ACL 2026 | Samples latent chains with MC dropout or Gaussian noise. A latent reward model, trained per step on success rates from 128 rollouts, picks among chains or guides beam search over latent steps. | On Coconut, 4 samples hold a right answer 46.7% of the time against 34.0% greedy, but best-of-4 reaches only 34.5% and majority vote 33.6%. Beam search ties best-of-N on GSM8K and trails on GSM-Hard, which the authors put down, untested, to noisy early-step scores. No aggregation results on CODI; the selector is supervised, not RL. |
| [SLPO](https://arxiv.org/abs/2607.19691), same group | RL (RLOO/GRPO) on the latent generator through a Gaussian surrogate built from MC dropout, plus a learned stopping head. | CODI 42.30 → 42.76; Coconut 34.12 → 35.63. No per-step choice among candidates. |
| [CoLaR](https://arxiv.org/abs/2505.16552) | A latent head that predicts a Gaussian over the next latent, then GRPO on the generator. | Up to +5.4 points on MATH. Not GPT-2. |
| [LEPO](https://arxiv.org/abs/2604.17892) | Gumbel-softmax randomness in latent reasoning, so RL can be applied to the latents directly. | Full text not read; its numbers are unverified. |
| [LaDi-RL](https://arxiv.org/abs/2602.01705), [HRPO](https://proceedings.neurips.cc/paper_files/paper/2025/file/087f7678af2abbe0c37ecc30bd7d1adf-Paper-Conference.pdf) | RL with latent diffusion policies, and RL over mixed token and latent rollouts. | Generator-side RL, not selection. |
| [ITS-Continuous](https://arxiv.org/abs/2510.12167), Findings of ACL 2026 | Inference-time scaling for continuous reasoning. | Text reward models "transfer poorly", and correct and incorrect continuous thoughts "lack the geometric separability" a scorer needs. This is the main risk for this idea. |
| [NoisyCoconut](https://arxiv.org/abs/2605.08221) | Votes across noise-perturbed latent paths and abstains when they disagree. | Exact-match voting only; it calls aggregation "largely unexplored". |
| [Rewarding Progress](https://arxiv.org/abs/2410.08146) (Setlur et al.), ICLR 2025 | Process advantage verifiers for explicit CoT: a step's reward is its progress under a prover policy. | The explicit-CoT case for per-step selection: +8–10% over outcome reward models in search, and 5–6× better sample efficiency in RL. |
| [Coconut](https://arxiv.org/abs/2412.06769) | Reads its continuous thoughts as an implicit search over next steps. | Motivation only. |

**What is open, as far as these searches found:**
- per-step selection on CODI or SIM-CoT;
- a selector trained by RL rather than by imitating rollout success rates;
- a selector that reads candidates through SIM-CoT's decoder.

LatentTTS is the paper to beat and cite.

## The plan

**Stage 0: evaluation only, no training (`scripts/step_select/`).** It runs on the SIM-CoT GPT-2 base and the 500
validation questions, with MC dropout at p = 0.2 (LatentTTS's rate) and two sampling seeds. It measures:
1. **Identity:** the dropout-free path must reproduce Stage 0's plain answers on 500/500 questions.
2. **Room to gain from choosing per step:** at each step, rerun it 3 times with dropout and continue each rerun normally
   to an answer. A question is coverable if the greedy answer or any of its 18 branches is right. This is a lower bound
   on what a per-step selector could reach.
3. **The same room, choosing among finished answers:** 18 independent dropout chains plus greedy, so both sides offer 19
   finished answers.
4. **Whether the decoder can tell the candidates apart:**
   - Look at the (question, step) sets whose candidates include both right and wrong continuations.
   - Does a fixed decoder rule pick a right one more often than chance?
   - For each feature, what is its AUC, the chance it ranks a right candidate above a wrong one?
5. **A selector with no training:**
   - It generates 4 candidates per step, keeps one by the rule, and continues from it.
   - Its cost is about the same as 4 chains, so it is compared with majority vote over 4 chains.

**The fixed decoder rule.** It was set before any run:
- Each candidate is ranked by two checks, in this order: does its decoded step have correct arithmetic, and are its
  operands grounded? An operand is grounded if it appears in the question, in an earlier chosen step, or in a small fixed
  set of constants (1, 2, 3, 4, 7, 10, 12, 24, 30, 60, 100, 1000).
- Candidate 0 is kept whenever it ties for the best rank.
- Otherwise, among the candidates with the best rank, the one with the most confident decode is kept. Confidence is the
  decoder's mean log-probability.

**Stage 0 decision rule.** This was written before Stage 0 ran, and both sampling seeds are pooled.
- **GO** to Stage 1 if both hold:
  - the per-step branches cover at least 40 more of the 500 questions than greedy does;
  - the decoder rule picks a right candidate more often than chance in the mixed sets, at p < 0.05.
- **STOP** if either holds:
  - the branches cover fewer than 20 more than greedy;
  - the decoder rule shows no signal (p ≥ 0.05) and the untrained selector does no better than greedy.
- **AMBIGUOUS** otherwise. An RL selector reading hidden states might still learn where the decoder features fail; we
  decide together.

The comparison with the 19 finished answers is reported, but it doesn't gate.

**Stage 1: the RL selector, about 30–40 G4-minutes per seed, two seeds.**
- **Setup:** K = 4 candidates per step. The selector is a small head over two kinds of input:
  - the candidate's hidden state;
  - the decoder features.
  - Its logits start out favoring candidate 0, so it begins as the checkpoint.
- **Training:** GRPO on training questions. Each question gets G full runs of 6 choices, rewarded 0 or 1 by the answer.
  Each run's advantage is its reward minus the group mean.
- **Ablations:**
  - the untrained rule;
  - a LatentTTS-style supervised scorer;
  - RL with and without the decoder features;
  - branching only at steps 1–3;
  - majority vote at the same compute.
- **Go:** pooled over two seeds, the RL selector beats greedy by at least 10 out of 500 and the best answer-level method
  at the same compute by at least 5, at p < 0.05.

**Stage 2: confirm.** Three seeds on the 1,319-question test set. Then CODI GPT-2, which needs a step reader other than
SIM-CoT's decoder, and the comparison with LatentTTS.

## Stage 0, concretely

For each validation question:

1. **Greedy.** Run the six latent steps with dropout off and read the answer. Save the cache and the latent after every
   step, and decode the seven latents with SIM-CoT's decoder.
2. **Chains.** From the question's cache, run 18 copies of all six steps with dropout on, then read each answer with
   dropout off.
3. **Branches.** For each step s from 1 to 6:
   - take greedy's state just before step s;
   - rerun step s 3 times with dropout on;
   - finish each rerun with dropout off and read its answer;
   - decode each rerun's new latent and score it with the rule's features.

   That gives 18 branches, each differing from greedy at exactly one step.
4. **Untrained selector.** Walk the six steps.
   - At each step, make candidate 0 plus 3 dropout reruns and decode all 4.
   - Keep one by the fixed rule, then continue from it.
   - Read the answer at the end.

The report then computes the identity check, the room for each kind of selection, the per-step rescue and break counts,
the signal tests and the verdict.

## Code

| File | What it does |
|---|---|
| `scripts/step_select/stepsel.py` | MC-dropout candidates, batched chains and readout, decoder reads with confidence, the features and the fixed rule |
| `scripts/step_select/eval.py` | Stage 0 for one sampling seed: greedy, chains, branches and the untrained selector, one JSON record per question |
| `scripts/step_select/report.py` | Identity, room to gain, per-step rescues, signal tests, the untrained selector against majority vote, and the verdict pooled over seeds |
| `notebooks/DSA_Step_Select_Colab.ipynb` | Colab notebook for Stage 0 |
| `tests/test_step_select.py` | Tiny-model checks: dropout off reproduces greedy, batched readout matches one row at a time, the rule, and an end-to-end run |

## Sources

- [Parallel Test-Time Scaling for Latent Reasoning Models (LatentTTS), arXiv 2510.07745](https://arxiv.org/abs/2510.07745)
- [SLPO: Scaling Latent Reasoning via a Surrogate Policy, arXiv 2607.19691](https://arxiv.org/abs/2607.19691)
- [CoLaR, arXiv 2505.16552](https://arxiv.org/abs/2505.16552)
- [LEPO, arXiv 2604.17892](https://arxiv.org/abs/2604.17892)
- [LaDi-RL, arXiv 2602.01705](https://arxiv.org/abs/2602.01705)
- [Hybrid Latent Reasoning via RL (HRPO), NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/087f7678af2abbe0c37ecc30bd7d1adf-Paper-Conference.pdf)
- [Towards Inference-time Scaling for Continuous Space Reasoning, arXiv 2510.12167](https://arxiv.org/abs/2510.12167)
- [NoisyCoconut, arXiv 2605.08221](https://arxiv.org/abs/2605.08221)
- [Rewarding Progress (Setlur et al.), arXiv 2410.08146](https://arxiv.org/abs/2410.08146)
- [Coconut, arXiv 2412.06769](https://arxiv.org/abs/2412.06769)
- [SIM-CoT, arXiv 2509.20317](https://arxiv.org/abs/2509.20317)
