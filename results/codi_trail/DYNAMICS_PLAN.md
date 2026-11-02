# CODI trail dynamics: plan fixed before the run

Post hoc follow-up. It was designed after seeing the probe results, so none of this is a replication or a confirmed claim.
Published CODI weights, no training, gsm8k_valid n=500; swap and mean also on SVAMP n=1000.
Control: the 6 and 12 step texts must match gsm8k_valid.jsonl on all 500. Any mismatch stops the analysis.

Facts that motivated it, from the saved latents: the trail alternates with period 2; past step 6 same-phase cosine is above 0.98;
147/500 answers still change between 6 and 12; 306 SVAMP questions settle on nearly one shared vector; CODI has no loss on
intermediate steps.

D1, parity. Read at 2,4,5,6,7,8,9,10,12. Compare 4v5, 6v5, 6v7, 8v7, 8v9, 10v9. Parity effect if the even depth has more
correct in at least 5 of 6 and pooled gains vs losses give a sign test p<0.01.

D2, where drift comes from. At depths 12 and 24:
  pos_reset: keep the full cache, read out at the position a 6 step run would use.
  late_six: feed only the last 6 latents after the question, then read out at the normal 6 step position.
  A variant recovers if at 24 it gains at least half of the 33 question loss (at least 190 correct) with sign test p<0.05 vs plain 24.

D3, whether the trail is question-specific. Read at 6 after feeding:
  swap: the next question's z_0..z_5.
  mean: the mean trail over all questions.
  Trail carries little question-specific content if swap keeps at least 90% of 6 step correct (at least 185 on validation).
  Trail carries real content if swap falls below 100 and answers follow the donor more than their own question.
  Between the two is a mixed result, reported as such.

What each outcome means for idea 3:
  Little question-specific content: a latent anchor has little to protect; idea 3 as specified is not worth building.
  Real content, drift recovered by pos_reset or late_six: drift is position or trail length, so a much simpler fix exists.
  Real content, neither recovers: the latent values themselves carry the drift, and an anchor has a real target.
