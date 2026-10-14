# Colab quickstart

For the **next test**, open [DSA_FLT_DataScale.ipynb](https://colab.research.google.com/github/sttruji/dynamic-state-anchoring/blob/main/notebooks/DSA_FLT_DataScale.ipynb), select **T4 GPU**, and **Run all**. It trains four FLT write-policy controls on 512 selected records for 1,024 updates each, evaluates all 500 validation questions, caps training diagnostics, and exports one small review ZIP. See [FLT_DATA_SCALE.md](FLT_DATA_SCALE.md) for the frozen settings, approximate budget and backup behavior.

The instructions below describe the earlier **32-record optimization pilot**.

Open [DSA_Colab.ipynb](https://colab.research.google.com/github/sttruji/dynamic-state-anchoring/blob/main/notebooks/DSA_Colab.ipynb), choose **T4 GPU**, and **Run all**. Use the current GitHub notebook, because an older Drive copy does not update automatically.

The default is a fresh **K=8, 32-question, 200-update** comparison of four conditions: standard/static, standard/learned, FLT-inspired/static and FLT-inspired/learned. It starts a separate experiment, reuses cached data/model downloads, checks matched initialization and saves a full backup after each arm. Partial arms restart; verified finished arms can be reused.

See [K8_FLT_PILOT.md](K8_FLT_PILOT.md) for the precise architecture, initialization screen, comparison design and verification limits.

Keep the **four full checkpoint ZIPs** downloaded after training each arm. The last cell exports **one small review ZIP**; hand that ZIP back for analysis. Wait for browser transfers to complete before disconnecting. Mounted Drive readback cannot guarantee cloud durability. The held-out test stays disabled.

## Evidence collected in one run

| Question | Evidence |
|---|---|
| Did both arms receive matched training? | Initialization/data/order/runtime/settings identities and actual per-update sample sequences |
| Did the models learn their training examples? | Question-only final-answer generation on all 32 training questions; deterministic teacher-forced losses on every training row |
| Does more latent depth change answers? | Same saved weights, same first eight training questions, K=1/2/4/8/16 |
| Do anchor writes and final memory matter? | Paired no-write, every-write, zero-anchor, zero-workspace and zero-memory inference interventions |
| Does the controller follow its labels? | Training gate targets, probabilities, decisions, precision/recall and positive/write rates |
| Do latent states settle or destabilize? | Per-loop anchor/workspace scales, changes and cosine similarities; comparison to question-embedding scale |
| Are teacher milestones being aligned? | Scheduled candidate/teacher normalized MSE, cosine and state norms |
| Where do large gradients arise? | Answer-only depth probes on four fixed training rows; weighted loss contributions at the training budget; actual per-loop input/output gradients |
| Do losses compete at the controller? | Gate-head and candidate-projection gradient direction cosines between loss components |
| Is the pretrained explicit reasoning route capable? | Original cached model versus each arm's trained shared backbone on the same eight train/eight validation questions |
| Does performance extend beyond the training set? | First 128 original validation questions, paired scores and raw outputs; use `VALID_LIMIT=0` for all 500 |
| Did generation end normally? | Raw generated token IDs, token counts and EOS/token-cap stop reasons in training/control diagnostics |
| What does the run cost and fit? | Original per-update timing, allocated GPU memory and loss/gradient journals; inference/control timings |
| Is enough evidence present to choose a larger run? | Hashed inventory, stage status, missing artifacts, paired validation counts and `scale_readiness.json` |

Intermediate reasoning stays in continuous vectors. Decoded text is an output
check rather than a transcript of those vectors. Alignment/state similarities
are proxies; interventions can be out of distribution. Explicit controls have a
192-token cap versus 64 for latent answer decoding, so they are not a matched
compute comparison. Depth interventions use saved weights, not separately
trained depth controls. Per-loss gradient norms cannot be added as scalars.

Probe rows are fixed in prepared source order before seeing their results.
No-event and overflow rows remain in the main losses, with the original auxiliary
masking policy. The source-data quality exclusions remain unapplied. Validation
is for development; the full test evaluation remains disabled by default.
