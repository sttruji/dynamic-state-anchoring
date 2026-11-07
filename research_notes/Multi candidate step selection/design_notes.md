# Design working notes (lead researcher), 2026-11-06

Not the report. Scratch reasoning for MULTI_CANDIDATE_SELECTION.md. Everything here is my inference unless cited.

## Facts from the repo used below

- Latent step = `latent_step` in `scripts/inner_loop/looped.py`: input `embeds + wpe(position)`, absolute learned
  positions, optional key mask. Several cache entries at the same position are legal for GPT-2 (attention is
  permutation-invariant over keys given absolute position embeddings baked into the inputs).
- Per-entry additive attention bias already exists: `scripts/latent_gate/batched.py` (dedup_soft, bias = −log cluster
  size at cos ≥ 0.95) and `scripts/inner_loop/kk.py` (KK-LSE).
- Measured: exact copies of latent entry 6 appended to 24 entries reproduce the whole SIM-CoT drift (172 vs plain 170);
  dedup_soft holds 207 at 24 vs 170 plain; KK-LSE at 6 steps changes 1/500 answers (no duplicates exist at 6).
- Stage 0 record per question: greedy {answer, correct, steps[0..6] decoder features}, branches[6][3] {answer, correct,
  text, logprob, parsed, valid, grounded, result}, chains[18] {prediction, correct}, rule {answer, chosen}. No latents or
  hidden states are saved.
- Srivastava et al. 2014 §7.5 (read): on MNIST, MC averaging of ~50 dropout nets matches weight scaling (= dropout off
  under inverted dropout). So averaging dropout reruns of a step should drift toward candidate 0.

## Compute accounting (token-forwards of a GPT-2-small; generator and decoder are both GPT-2 small)

c_L = 1 per latent step, c_D ≈ T_D (decoder read length, `<<3*12=36>>` ≈ 7 BPE tokens + EOS ≈ 8, cap 32),
c_A ≈ 1 + T_A (answer readout after `<eot>`, ≈ 4–8 tokens).

| method | latent steps | decoder tokens | readout tokens | total at T_D=8, T_A=6 |
|---|---|---|---|---|
| greedy | 6 | 0 | 7 | 13 |
| maj@N chains | 6N | 0 | 7N | 13N |
| single-select K=4, decoder on all steps | 24 (18 if cand 0 reused) | 24·8 = 192 | 7 | ≈ 223 (≈ maj@17) |
| single-select K=4, decoder at steps 1–3 only | 24 | 12·8 = 96 | 7 | ≈ 127 (≈ maj@10) |
| single-select K=4, hidden-state features only | 24 | 0 | 7 | 31 (≈ maj@2.4) |
| keep-alive, mean live width W̄, K=4 | 24·W̄ | 24·W̄·8 | 7·W_6 | ≈ 223·W̄ |
| merge-now, K=4 | 24 | 192 | 7 | ≈ 223 + tiny attention overhead |
| Stage 0 branches (18) | 63 | 18·8 | 18·7 | 333 |

The current plan's "about the same as 4 chains" counts generator latent steps only. The decoder reads dominate token
forwards. Matched-compute baselines must be stated in both units.

## Policy forms

- P1 primary + extras: a ~ Cat(softmax u), extras j≠a ~ Bern(σ(g_j)). log π = log softmax(u)_a + Σ log Bern.
  Action is (anchor, extras) so no marginalisation over which member is the anchor. Identity: u_0 bias, g bias −6.
- P2 size head + Plackett–Luce: n ~ Cat{1..W}, then PL/Gumbel-top-k ordered prefix. log π = log p(n) + PL prefix.
  For K = 4 the unordered-set probability is a sum over ≤ 24 permutations: exact and cheap.
- P3 relative threshold: keep k if u_k ≥ max u − τ, τ ~ logNormal(head). Beam-pruning style.
- P4 frontier allocation (REBASE-like) for keep-alive: total continuation budget B split over live nodes by softmax(u/T).

## Credit

- GRPO over G rollouts, R = 1[correct] − λ·(extra cost / single cost), λ by dual ascent to a target width.
- Difference reward for the extras gate (keep-alive): spine = path of anchors. R_extras = R(aggregate) − R(spine) −
  λ·extra cost; anchor choice trained on R(spine) as in single-select. If extras never pay, the gate closes and the
  method reduces to the single-select plan. For merge-now the spine needs one extra no-merge rollout at training time.
- Tree rollouts give node values for free: V(node) = mean correctness of its leaves (TreeRL-style).

## Merge operators (option a)

- M1 weighted mean of next latents, renormalised to the mean norm; KV of anchor only.
- M2 superposed step: keep all |S| candidates' KV entries at position q+s, add bias log w_k (Σ w = 1) in every layer for
  every later query. Near-identical keys ⇒ exactly a w-mixture of values with the step's total attention mass unchanged.
  Different keys ⇒ query-dependent choice. |S| = 1 ⇒ bias 0 ⇒ identity. Without the bias, |S| near-copies get |S|× mass
  = the measured dilution/overweighting mechanism. This is dedup_soft with known fractional counts; KK-LSE estimates the
  count from keys and is only needed when w is unknown.
- M3 learned pooling (PMA with one seed, zero-init residual on the anchor) trained by backprop through the frozen
  generator on teacher-forced gold-answer CE; RL only for the subset. Identity at init.
- M4 KV averaging per layer (CaM/KVMerger-like). Keys from different dropout masks share coordinates (same weights), so
  no permutation alignment problem, but averaged keys have smaller norm ⇒ flatter attention.

## Offline checks on Stage 0 records (no new runs)

C1 diversity and minority-correct rate per step; C2 step-local keep-all-4-and-vote; C3 pooled 19-answer votes, branches
vs chains, flat vs level-balanced, matched compute; C4 hedge vs commit under the fixed rule (top-1 vs tied-best set vote
vs all ≥ rank of cand 0); C5 "where" oracle (some step whose 4-set plurality is right) vs "which" oracle (coverage);
C6 decoded-result clusters as redundancy signal; C7 single dropout chain accuracy vs greedy (pass@1 of chains);
C8 cross-validated within-question logit on decoder features → accuracy-vs-width Pareto curve for a keep-extras
threshold; C9 (needs a short new eval) mean-of-reruns and superposed-KV continuation agreement with greedy.
