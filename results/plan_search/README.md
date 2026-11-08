# Answer-conditioned plan search (H): results

H searches the first latent z₁ of SIM-CoT GPT-2, on training questions the model gets wrong, for plans whose own latent
steps reach the gold answer. If such plans exist, Stage 1 would distill them into the loop. The idea is entry H in
`docs/brainstorm/ACCURACY_IDEAS.md`, and the code is in `scripts/plan_search/`. This is brainstorm work, kept out of the
course report.

## Stage 0 (evaluation only, commit `f782268`): KILL

**The plan.** It is frozen in `stage0/PLAN.md` and pinned by `stage0/PLAN.sha256`. Every criterion was fixed before the run.
- **Phase 1:** greedy on GSM8K-Aug training questions until 1,000 are wrong. Each wrong question gets a radius, the
  farthest of 8 dropout samples of z₁.
- **Phase 2:** 20 Adam steps on z₁ within that radius and the top 64 principal directions. One search aims at gold and
  one at a wrong integer within ±50 of gold, chosen as the closest to gold in starting log-prob. A plan is accepted if
  the model's own steps from it give exactly the target.
- **Gate:** GO only if gold is accepted at least 1.5× as often as wrong, with at least 150 gold plans.

**The run.**
- **Session and checks:** one Colab session. The identity check passed on 20 questions.
- **Phase 1** (215 seconds): 5,421 training questions, greedy right on 4,421 (81.6%), leaving 1,000 greedy-wrong.
- **Phase 2** (152 seconds): all 1,000 searched, under the cap.
- **Files:** `stage0/report.txt` is the verdict, reproduced locally to the byte from `stage0/search.jsonl`.
  `stage0/explore.txt` is the output of `scripts/plan_search/explore.py`, which is exploratory and never gating. The z₁
  matrix (17 MB) stays out of git, on Drive at `plan-search/stage0/run/z1.npy`.

| | Gold | Wrong |
|---|---|---|
| Plans accepted, of 1,000 | **676** (67.6%) | **478** (47.8%) |
| Accepted on that question only | 255 | 57 |
| Ended on the radius boundary | 858 | 875 |
| Median log-prob gain | +1.62 | +1.83 |
| Median starting log-prob | −3.03 | −3.81 |

**The verdict is KILL:** gold was accepted only 1.41× as often as wrong, under the 1.5× bar. Stage 1 does not run, and
neither does wrong-plan distillation.

**What the exploratory reads show.** None of these gate.
- **The search mostly writes answers.** Within one question's own dropout spread, nearly half of wrong integers can be
  written in: the median wrong target was 12 from gold. Most plans of both kinds pressed against the radius, so a looser
  ball would reach still more.
- **Much of gold's edge comes from loose matching.** The plan matched the wrong target to gold's starting log-prob, but
  near a large gold no integer is as likely as gold.
  - The ratio rises with answer size, from 1.13 for golds under 10 to 5.81 for golds of 1,000 or more, where wrong was
    accepted only 11% of the time.
  - Where the two targets started within 0.25 nats (483 questions), the ratio is **1.18**.
  - Where wrong started at least as likely as gold (294 questions), it is **1.07** (+39/−27, p = 0.18).
- **A small gold edge remains,** at +78/−37 among the well-matched questions, and mostly where a target starts between
  −5 and −3. These are training questions SIM-CoT was trained on, so the edge may be memorized answers rather than plans.
  Either way, most accepted gold plans would not be gold-specific, and distilling them would mostly teach the loop to
  write answers into z₁.
- **The model is near the answer on many failures.** Gold's median starting log-prob of −3.03 means the model often gives
  the gold answer a few percent of its probability even when greedy is wrong.

**What closes, per PLAN.md:** H's Stage 1 and its wrong-plan arm. The search does not separate plans from answer
injection near greedy.
