# Baselines and feasibility for an accuracy-raising latent-CoT method (GPT-2 / LLaMA-3.2-1B, GSM8K-Aug)

Notes compiled 2026-11-05 as the shared reference for judging three candidate ideas: looped layers per latent step, decode-and-reinject, and recursive attention.

Verification tags:
- [T] = number read from the source paper's own table this session, through a summarizing fetch of the arXiv HTML. The page itself was not hand-read, so a transcription error from the fetch model is possible. Spot-check before quoting in a paper.
- [C] = the source paper shows the number only in a figure. The value comes from a later paper's table that copies it and marks it as copied (PCCoT, KaVa).
- [P] = from this project's earlier notes (`research_notes/Latent thought attention gating/latent_cot_depth_supervision.md`). Not re-fetched today.
- [U] = unverified (search snippet, or inferred).
- [code] = read directly from the repository source files, downloaded with curl.

"GSM8K" means the GSM8K test set (1,319 questions) unless noted. GSM8K-Aug is the 385k-example equation-style training set of Deng et al. GSM8K-Aug-NL is its natural-language variant. Our 500-question split matches the standard GSM8K-Aug **validation** split ("a validation set of 500 samples and a test set of 1,319 samples", PCCoT App. C.1). Our ~205/500 = 41.0% is therefore a validation number, while every published number below is a test number.

## Q1. Published GSM8K accuracies at GPT-2 scale (GPT-2 small, 124M), with paper, table and arXiv id

### Takeaway
On GPT-2 trained on GSM8K-Aug, published GSM8K-test accuracies cluster as follows. No-CoT is 13–19. Explicit CoT-SFT is 39.8–44.1, depending on who trained it. CODI is 40.9–43.7. SIM-CoT-on-CODI is 42.1–42.6, PMPS 44.35 and LOTUS 44.1±0.7. The outlier is **PCCoT at 49.48±0.31** (24 latents, 3 Jacobi iterations). PCCoT's own CODI-style "Continuous CoT" baseline with **12 latents** (no projection MLP) gets **48.24±1.61**. Every paper that reran CODI got less than CODI's own 43.7 (42.0 and 40.86). Within-paper reproduction gaps of 2–3 points are therefore normal.

### Cited Findings

**GPT-2, trained on GSM8K-Aug, evaluated on the GSM8K test set**

| Method | GSM8K acc. | Paper / table | arXiv | Tag |
|---|---|---|---|---|
| No-CoT SFT | 19.1 | CODI Table 2 (ablation text); also SIM-CoT Table 1 and LOTUS Table 1 | 2502.21074; 2509.20317; 2606.31779 | [T] — [CODI](https://arxiv.org/html/2502.21074), [SIM-CoT](https://arxiv.org/html/2509.20317) |
| No-CoT SFT | 16.5 | Coconut Table 1 | 2412.06769 | [P] — [Coconut](https://arxiv.org/html/2412.06769) |
| No-CoT SFT | 0.13 (i.e. 13%) | iCoT-SI Table 3 | 2405.14838 | [T] — [iCoT](https://arxiv.org/html/2405.14838) |
| No-CoT SFT | 15.4±0.2 | RiM Table 4 (own splits, SE over 16 split repeats) | 2605.30343 | [T] — [RiM](https://arxiv.org/html/2605.30343v1) |
| Explicit CoT SFT | 44.1 | CODI (Fig. 3 only), copied in PCCoT Table 1 | 2502.21074 / 2506.18582 | [C] — [PCCoT](https://arxiv.org/html/2506.18582v2) |
| Explicit CoT SFT | 42.9 | Coconut Table 1 | 2412.06769 | [P] — [Coconut](https://arxiv.org/html/2412.06769) |
| Explicit CoT SFT | 42.7 | SIM-CoT Table 1 (also LOTUS Table 1) | 2509.20317 | [T] — [SIM-CoT](https://arxiv.org/html/2509.20317) |
| Explicit CoT SFT | 42.50 | PMPS Table 1 | 2609.09928 | [T] — [PMPS](https://arxiv.org/html/2609.09928) |
| Explicit CoT SFT | 0.41 | iCoT-SI Table 3 | 2405.14838 | [T] — [iCoT](https://arxiv.org/html/2405.14838) |
| Explicit CoT SFT | 39.8±0.2 | RiM Table 4 (own splits) | 2605.30343 | [T] — [RiM](https://arxiv.org/html/2605.30343v1) |
| iCoT-SI | 0.30 (GPT-2 Medium 0.35); iCoT-KD 0.20 | iCoT-SI Table 3 | 2405.14838 | [T] — [iCoT](https://arxiv.org/html/2405.14838) |
| iCoT-SI | 30.0 / 30.1 | Coconut Table 1 / SIM-CoT Table 1 | 2412.06769 / 2509.20317 | [P]/[T] — [SIM-CoT](https://arxiv.org/html/2509.20317) |
| Coconut | 34.1±1.5 | Coconut Table 1 | 2412.06769 | [P] — [Coconut](https://arxiv.org/html/2412.06769) |
| Coconut (rerun) | 36.6 / 36.50 / 31.1±0.2 | SIM-CoT Table 1 / PMPS Table 1 / RiM Table 4 (w/ Stage 0) | 2509.20317 / 2609.09928 / 2605.30343 | [T] |
| Pause tokens | 16.4 (Coconut); 39.27±0.46 (PCCoT, 24 pause tokens) | Coconut Table 1; PCCoT Table 1 | 2412.06769; 2506.18582 | [P]; [T] |
| CODI (6 latents) | **43.7** | CODI Table 2 / Table A5 (text). The main comparison is Fig. 3 only. | 2502.21074 | [T] (text) — [CODI](https://arxiv.org/html/2502.21074) |
| CODI (rerun) | 42.0 | SIM-CoT Table 1 (copied by LOTUS Table 1) | 2509.20317 | [T] — [SIM-CoT](https://arxiv.org/html/2509.20317) |
| CODI (rerun) | 40.86 | PMPS Table 1 | 2609.09928 | [T] — [PMPS](https://arxiv.org/html/2609.09928) |
| SIM-CoT on CODI | **42.6** (+0.6) | SIM-CoT Table 1 | 2509.20317 (ICLR 2026 per repo title) | [T] — [SIM-CoT](https://arxiv.org/html/2509.20317), [repo](https://github.com/InternLM/SIM-CoT) |
| SIM-CoT on CODI (rerun) | 42.08 | PMPS Table 1 | 2609.09928 | [T] |
| SIM-CoT on Coconut | 44.8 (+8.2 over 36.6) | SIM-CoT Table 1 | 2509.20317 | [T] |
| CODI-style "Continuous CoT", **12 latents**, no projection MLP | **48.24±1.61** (3 seeds) | PCCoT Table 1 | 2506.18582 | [T] — [PCCoT](https://arxiv.org/html/2506.18582v2) |
| **PCCoT** (c=24 latents, T=3 Jacobi iterations) | **49.48±0.31** (3 seeds) | PCCoT Table 1 (EMNLP 2025 main) | 2506.18582 | [T] — [PCCoT](https://arxiv.org/html/2506.18582v2) |
| PCCoT's "Implicit CoT" reimplementation | 37.78±0.31 | PCCoT Table 1 | 2506.18582 | [T] |
| PMPS | **44.35** (no std) | PMPS Table 1 | 2609.09928 (2026 preprint) | [T] — [PMPS](https://arxiv.org/html/2609.09928) |
| LOTUS (looped transformer; full FT) | **44.1±0.7**; LOTUS+CODI 43.1±1.2; LOTUS-aux 35.5±4.5 | LOTUS Table 1 | 2606.31779 (2026 preprint) | [T] — [LOTUS](https://arxiv.org/html/2606.31779v1) |
| CoLaR-2 | 11.07 (PMPS's reproduction; CoLaR's authors report no GPT-2) | PMPS Table 1 | 2609.09928 | [T] |
| RiM (memory blocks) | 39.5±0.3 any-block / 33.6±0.2 final-block (own splits); 35.5 vs Coconut 34.1 on "official numbers" | RiM Tables 4 and 6 | 2605.30343 (2026 preprint) | [T] — [RiM](https://arxiv.org/html/2605.30343v1) |
| KaVa | **not reported for GPT-2** (backbones: Qwen2.5-0.5B, Llama-3.2-1B/3B) | KaVa Table 1 | 2510.02312 (ICLR 2026) | [T] — [KaVa](https://arxiv.org/html/2510.02312v1) |
| Latent-SFT | **not reported for GPT-2** (Llama-3.2-1B/3B, DeepSeek-Distill-Qwen-7B) | Latent-SFT Table 1 | 2510.15522 | [T] — [Latent-SFT](https://arxiv.org/html/2510.15522v2) |
| SemCoT | **no GPT-2** (Llama-2-7b-chat, Mistral-7B-Instruct; its GSM8K numbers of about 9.8–18.5 come from a garbled snippet) | SemCoT (NeurIPS 2025) | 2510.24940 | [U] — [SemCoT](https://arxiv.org/pdf/2510.24940) |
| SoftCoT | **no GPT-2-scale result found**. Backbones are not visible in the abstract; it is ACL 2025 main. | SoftCoT | 2502.12134 | [U] — [SoftCoT](https://arxiv.org/abs/2502.12134) |

**GPT-2 trained on GSM8K-Aug-NL (natural-language CoT), GSM8K test.** All from PCCoT Table 1 [T]. The CoT, CODI and iCoT rows are copied from the literature. CoT 34.8; CODI 35.3; iCoT 3.2; Pause 33.79±2.10; Continuous CoT (12 latents) 45.06±2.58; **PCCoT 49.23±0.80**. — [PCCoT](https://arxiv.org/html/2506.18582v2)

**Other GPT-2 results relevant to the three ideas**
- PCCoT Table 4 (GPT-2, GSM8K / GSM8K-NL): "PCCoT w/ CoT decoding" 50.42 / 39.80, i.e. decoding text after the latents. "CoT w/ gold reasoning steps" 88.21 / 87.95. "PCCoT w/ CoT decoding & gold reasoning steps" 62.24 / 55.88. This is the closest published decode-style variant. It gives about +0.9 on GSM8K and hurts on NL. [T] — [PCCoT](https://arxiv.org/html/2506.18582v2)
- PCCoT Table 3: making latent tokens untrainable gives 47.31±1.07 vs 49.48±0.31. [T] — [PCCoT](https://arxiv.org/html/2506.18582v2)
- PCCoT App. C.2: "Though CODI reports 6 latent thought tokens is the best performing setting for continuous CoT, we find that using 12 latent thought tokens is much better than using 6 in our experiments." The exact 6-vs-12 numbers are only in Figs. 2–3. [T] (quote) — [PCCoT](https://arxiv.org/html/2506.18582v2)
- CODI App. D: models fine-tuned on decoded latent patterns reach only 34.1–35.7% vs CODI 43.7%. [P] — [CODI](https://arxiv.org/html/2502.21074)

### Inferences
- The 43.7 that is usually quoted for CODI GPT-2 is the authors' own single deterministic run. Independent reruns get 42.0 (SIM-CoT) and 40.86 (PMPS). Our harness's 41.0% on the validation split is consistent with these.
- The single most important baseline fact for a new method is that **a CODI-style model trained with 12 latents and no projection MLP already reaches 48.24±1.61** (PCCoT). The new method's gain must also be compared at matched latent count and matched compute, not only against 6-step CODI. Otherwise "more compute per latent" is confounded with "more latents", which PCCoT shows already helps by about 4.5 points.
- PCCoT's "CoT w/ CoT decoding" row is the published precedent for decode-style ideas. It shows decoding after latents gives little on equation-style data, but it is decoding *after* the latents, not decode-and-reinject *between* latent steps.
- "CoLaR/LaTent variants": no method named "LaTent" was found. The closest matches are Latent-SFT (2510.15522) and CoLaR-2/CoLaR-5, which have no GPT-2 results from their own authors.

### Gaps
- CODI's main GSM8K comparison (Fig. 3) is a bar chart. The GPT-2 CoT-SFT 44.1 and iCoT 30.0 come from PCCoT's copy, not from a CODI table. GSM8K-Aug-NL CODI 35.3 is likewise from PCCoT's copy.
- Coconut's Table 1 values (34.1±1.5 etc.) are from earlier project notes and were not re-fetched today. The number of Coconut seeds behind ±1.5 was not checked.
- PCCoT's exact per-c (6/12/24) accuracies are only in figures.
- SoftCoT's and SemCoT's backbones and numbers were not read from the full texts. They are 7B-class papers and are not relevant at GPT-2 scale.

## Q2. The same methods on LLaMA-3.2-1B(-Instruct)

### Takeaway
On LLaMA-3.2-1B-Instruct trained on GSM8K-Aug, the best published latent numbers are **LOTUS+CODI 58.3±0.8** and LOTUS 57.3±0.2, then PMPS 56.56, KaVa 56.5±0.4, SIM-CoT 56.1, and CODI 55.6 (authors) or 52.7–53.9 (reruns). Explicit CoT-SFT at 1B varies by 14 points across papers (49.1–63.0), depending on the recipe. Whether a latent method "beats CoT" at 1B therefore depends on which CoT row is used. Against the stronger CoT rows (61.6 CODI, 63.0 PMPS) no latent method wins.

### Cited Findings

| Method | GSM8K acc. (Llama-3.2-1B-Instruct, train GSM8K-Aug) | Paper / table | arXiv | Tag |
|---|---|---|---|---|
| No-CoT | 30.9 (CODI, via KaVa copy); 28.8 (SIM-CoT T2); 23.9±0.2 (RiM T4) | KaVa T1 †; SIM-CoT T2; RiM T4 | 2510.02312; 2509.20317; 2605.30343 | [C]; [T]; [T] |
| CoT-SFT | 61.6 (CODI, via copies); 58.4 (SIM-CoT T2; LOTUS T1); 63.00 (PMPS T1); 49.4±0.72 (CoLaR T1); 49.4±0.6 (Latent-SFT T1); 49.1±0.4 (RiM T4) | as listed | 2502.21074 / 2509.20317 / 2609.09928 / 2505.16552 / 2510.15522 / 2605.30343 | [C]/[T] — [KaVa](https://arxiv.org/html/2510.02312v1), [SIM-CoT](https://arxiv.org/html/2509.20317), [PMPS](https://arxiv.org/html/2609.09928), [CoLaR](https://arxiv.org/html/2505.16552) |
| iCoT | 19.0 (CODI via copy; SIM-CoT T2); 19.8±0.23 (CoLaR T1) | KaVa T1 †; SIM-CoT T2; CoLaR T1 | | [C]/[T] |
| Coconut | 45.3 (CODI via copy); 33.2 (SIM-CoT T2); 40.03 (PMPS T1); 23.1±0.28 (CoLaR T1); 36.9±0.2 (RiM T4); 50.6 ("official", RiM T6) | as listed | | [C]/[T] |
| Coconut + SIM-CoT | 42.2 | SIM-CoT T2 | 2509.20317 | [T] |
| CODI | **55.6** (authors; via PCCoT/KaVa copies); 52.7 (SIM-CoT rerun); 53.90 (PMPS rerun); 13.3±0.62 ("Distill (CODI reproduction)", CoLaR T1) | | 2502.21074 etc. | [C]/[T] — [PCCoT](https://arxiv.org/html/2506.18582v2) |
| CODI-style SFT "Latent-6" | 49.73 (+RL 50.11); "Latent-6-by-1" 51.40 | Learning When to Stop | 2511.21581 | [P] — [source](https://arxiv.org/html/2511.21581v1) |
| SIM-CoT on CODI | **56.1** (+3.4); PMPS rerun 56.03 | SIM-CoT T2; PMPS T1 | 2509.20317 | [T] |
| PCCoT | 53.35±0.18 (PCCoT T1); listed as 54.2 in LOTUS T13 (**discrepancy**) | PCCoT T1; LOTUS T13 | 2506.18582; 2606.31779 | [T] — [PCCoT](https://arxiv.org/html/2506.18582v2), [LOTUS](https://arxiv.org/html/2606.31779v1) |
| PCCoT's "Continuous CoT" (12 latents) | 50.47±0.68 | PCCoT T1 | 2506.18582 | [T] |
| KaVa (24 latents, 3 Jacobi iterations) | **56.5±0.4** (3 seeds, SE). The column assignment was inferred from an ambiguous header. | KaVa T1 | 2510.02312 | [T]* — [KaVa](https://arxiv.org/html/2510.02312v1) |
| PMPS (8 latents for 1B) | **56.56** | PMPS T1 | 2609.09928 | [T] |
| CoLaR-2 / CoLaR-5 | 40.1±0.20 / 26.8±0.17 (authors, 5 seeds, 95% CI); PMPS rerun CoLaR-2 42.15 | CoLaR T1; PMPS T1 | 2505.16552 | [T] — [CoLaR](https://arxiv.org/html/2505.16552) |
| Latent-SFT (2) / (4) | 52.4±0.2 / 46.8±0.1 (authors, 5 seeds); PMPS rerun 48.82 | Latent-SFT T1; PMPS T1 | 2510.15522 | [T] — [Latent-SFT](https://arxiv.org/html/2510.15522v2) |
| LOTUS / LOTUS+CODI | **57.3±0.2 / 58.3±0.8** | LOTUS T1 | 2606.31779 | [T] — [LOTUS](https://arxiv.org/html/2606.31779v1) |
| RiM | 51.4±0.2 any-block / 42.1±0.2 final-block (own splits) | RiM T4 | 2605.30343 | [T] |

**Llama-3.2-1B trained on GSM8K-Aug-NL**: CoT 54.1 (PCCoT T1 copy) or 53.2 (KaVa T1); CODI 49.7; PCCoT 50.72±1.39; Continuous CoT 48.47±1.40; **KaVa 55.7±0.4**. [T] — [PCCoT](https://arxiv.org/html/2506.18582v2), [KaVa](https://arxiv.org/html/2510.02312v1)

**Larger models, for context**: SIM-CoT T3 reports CODI → SIM-CoT 60.8→62.3 on Llama-3.2-3B and 61.1→64.1 on Llama-3.1-8B. LOTUS reports 70.0 on Llama-3.2-3B (T13), against explicit CoT 71.5. [T] — [SIM-CoT](https://arxiv.org/html/2509.20317), [LOTUS](https://arxiv.org/html/2606.31779v1)

### Inferences
- At 1B, the CoT-SFT reference is not one number. CODI/PCCoT/KaVa copy 61.6. SIM-CoT and LOTUS use 58.4. PMPS uses 63.0. CoLaR, Latent-SFT and RiM, which all start from a CoT-SFT checkpoint trained by Coconut's recipe, get about 49. Our harness's gpt2-sft and the future LLaMA-1B SFT baseline should be trained with the CODI/SIM-CoT recipe (LoRA r128, about 10 epochs) so that comparisons are not made against a weak CoT row.
- The reruns of CODI-1B range from 13.3 (CoLaR's failed reproduction) to 55.6. Coconut-1B ranges from 23.1 to 50.6. These methods are recipe-sensitive. For the LLaMA port, use the official CODI script and checkpoint (zen-E/CODI-llama3.2-1b-Instruct), not a reimplementation.
- The current published 1B latent bar is about 56–58 (SIM-CoT, KaVa, PMPS, LOTUS). A new method that only matches CODI's 55.6 at 1B would not be news.

### Gaps
- The KaVa table header was ambiguous in extraction. The AUG vs AUG-NL column split was inferred by the fetch model. The † marks show which baselines KaVa copied.
- PCCoT-1B 53.35 (PCCoT) vs 54.2 (LOTUS T13) is unresolved. LOTUS may have rerun it or mis-copied it.
- CoLaR states its test split is "1k samples", not 1,319. This was not checked further.

## Q3. Current best GPT-2-scale latent CoT number on GSM8K (as of 2026-11-05), and whether any latent method beats explicit CoT-SFT at GPT-2 scale

### Takeaway
The best GPT-2 latent number found is **PCCoT, 49.48±0.31** (arXiv:2506.18582, EMNLP 2025, Table 1; 24 latents, 3 Jacobi iterations, 3 seeds). On GSM8K-Aug-NL it reaches 49.23±0.80. LOTUS's 2026 comparison table (T13) still lists PCCoT's 49.5 as the best GPT-2 entry. No 2026 paper found beats it at GPT-2 scale. Several latent methods do exceed the published GPT-2 CoT-SFT rows (39.8–44.1): PCCoT (+5.4 over the 44.1 it copies), its 12-latent continuous-CoT baseline (48.24), Coconut+SIM-CoT 44.8, PMPS 44.35 and LOTUS 44.1. Except for PCCoT, the margins are within about 2 points, and most of those rows have no seed variance.

### Cited Findings
- PCCoT GPT-2: 49.48±0.31 (GSM8K-Aug) and 49.23±0.80 (GSM8K-Aug-NL), vs CoT 44.1 / 34.8 (both copied from the literature). — [PCCoT](https://arxiv.org/html/2506.18582v2)
- LOTUS Table 13 (GSM8K, trained on GSM8K-Aug): GPT-2 column = Explicit CoT 42.7, Coconut 36.6, Coconut+SIM-CoT 44.8, **PCCoT 49.5 (bold, best)**, KaVa −, LOTUS 44.1. — [LOTUS](https://arxiv.org/html/2606.31779v1)
- PMPS claims that on GPT-2 it outperforms CODI and SIM-CoT by 3.49% and 2.27% on GSM8K-Aug. Its average accuracy over in-domain plus OOD (49.95) beats CoT-SFT (45.99) by 3.96 points, and it uses 47.1% of the tokens. PMPS does not compare against PCCoT. — [PMPS](https://arxiv.org/html/2609.09928); [search summary](https://arxiv.org/pdf/2609.09928)
- CODI claims to reach "99% of its accuracy" of CoT-SFT with GPT-2 (43.7 vs 44.1). — [CODI](https://arxiv.org/html/2502.21074)
- SIM-CoT on Coconut: 44.8 vs SFT-CoT 42.7 (same paper). — [SIM-CoT](https://arxiv.org/html/2509.20317)
- RiM (2026) does not beat CoT on GSM8K greedy: 39.5 vs 39.8. It does beat CoT on Pass@8 (78.1 vs 57.0). — [RiM](https://arxiv.org/html/2605.30343v1)
- Information-theoretic analysis (2606.20075, June 2026): outcome-only latent supervision underperforms. A search snippet reports Llama-3.2-1B OS-LATENT 33.81% vs explicit CoT 60.20%. Not checked in the full text. — [arXiv abs](https://arxiv.org/abs/2606.20075) [U for the numbers]
- Search turned up other 2026 latent papers with GPT-2 experiments that are not accuracy records: GTS (2602.14077, inference-time scaling of Coconut on GPT-2) and A*-Thought-V2 (2609.07821). Their numbers were not checked. — [GTS](https://www.arxiv.org/pdf/2602.14077), [A*-Thought-V2](https://arxiv.org/pdf/2609.07821) [U]

### Inferences
- The bar any new GPT-2 method must clear to claim state of the art is **about 49.5 on GSM8K test** (PCCoT). To claim a gain over its own base, it must clear the 6-step CODI/SIM-CoT baseline in our harness (about 41% on the validation split) by more than seed noise (see Q7).
- PCCoT's 49.5 uses 24 latents with 4 parallel passes. Its "Continuous CoT" uses 12 sequential latents. A fair comparison for any "more compute per latent step" idea is CODI at 12 latents, or PCCoT, at matched FLOPs.
- PCCoT's two GPT-2 checkpoints (whynlp/pccot-gpt2, whynlp/pccot-gpt2-nl) are public (see Q4). They could be added to the harness as a fifth baseline at low cost.

### Gaps
- The 2026 literature search was limited to a handful of queries. A higher GPT-2 number could exist in a paper that does not use these keywords. An OpenReview result ("Learning Multi-step Reasoning via Persistent Latent State Propagation", id Dcv4B1UCuW) could not be opened (bot-check page).

## Q4. Training recipes: CODI, SIM-CoT, PCCoT (plus KaVa, PMPS, LOTUS, CoLaR, Latent-SFT): hyperparameters, time and hardware, code, and continuing from released checkpoints

### Takeaway
**CODI GPT-2**:
- LoRA r=128, α=32 on GPT-2, plus a 2-layer MLP+LayerNorm projection with prj_dim=768.
- lr 3e-3, cosine schedule, 3% warmup, weight decay 0.1, grad-norm clip 2.0, bf16.
- Effective batch 128 (64×2), **40 epochs**, 6 latents, seed 11.
- Distillation is smooth-L1 on the hidden states of **all layers** at the token before the answer (the ":" of "The answer is:"), divided by the teacher's std and averaged over layers. Loss weights are 1/1/1 for GPT-2; the distill factor is 20 for Llama-1B.
- Training takes about **36 h on one A100-80GB** (Llama-1B: about 48 h, 10 epochs, lr 8e-4).
- The official repo has a `restore_from` flag that loads a full state dict for fine-tuning. It reads safetensors, while the HF checkpoint is `pytorch_model.bin`, so a format conversion is needed. No optimizer state is released.

**SIM-CoT GPT-2** uses the identical CODI script plus `--use_decoder True`. It trains **from base GPT-2 for 40 epochs**, not from the CODI checkpoint, and the step-loss factor defaults to 1.0.

**PCCoT GPT-2**:
- Same LoRA and lr settings, 40 epochs, no projection MLP.
- 24 latents, updated in parallel for T=3 extra Jacobi iterations.
- Trains in 13.72 h on 2×H800, vs 24.91 h for 12-latent sequential continuous CoT and 4.39 h for CoT-SFT.
- Code and GPT-2 checkpoints are released.

### Cited Findings

**CODI (Shen et al., arXiv:2502.21074; code github.com/zhenyi4/codi)**
- Paper App. A: LoRA rank 128, alpha 32, bf16. AdamW with a cosine schedule and linear warmup over the first 3% of steps. Effective batch 128. lr 3e-3 (GPT-2) and 8e-4 (Llama-3.2-1B). Epochs 40 (GPT-2) and 10 (Llama-1B). α=β=1. γ=1 for GPT-2 and γ=20 for Llama-1B. App. F calls β=20 the default for Llama-1B, which **conflicts with App. A**. — [CODI](https://arxiv.org/html/2502.21074)
- Distilled position: all layers, at the ":" token of "The answer is:". The projection is "a two-layer MLP followed by layer normalization". 6 continuous thoughts. — [CODI](https://arxiv.org/html/2502.21074)
- Compute: "a single A100 (80GB)". GPT-2 takes "approximately 36 hours" and Llama-3.2-1B "approximately 48 hours". — [CODI](https://arxiv.org/html/2502.21074)
- `scripts/train_gpt2_gsm8k-aug.sh` [code] sets:
  - `--model_name_or_path gpt2 --data_name icot --seed 11 --model_max_length 512`
  - `--per_device_train_batch_size 64 --gradient_accumulation_steps 2 --bf16 --num_train_epochs 40 --learning_rate 3e-3 --max_grad_norm 2.0`
  - `--use_lora True --lora_r 128 --lora_alpha 32 --lora_init --save_strategy "no" --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type "cosine"`
  - `--num_latent 6 --use_prj True --prj_dim 768 --prj_dropout 0.0 --distill_loss_div_std True --remove_eos True`
  - The Llama-1B script uses batch 32×4, 10 epochs, lr 8e-4, `prj_dim 2048`, `--distill_loss_factor 20` and `--max_token_num 200`. — [script](https://raw.githubusercontent.com/zhenyi4/codi/main/scripts/train_gpt2_gsm8k-aug.sh), [Llama script](https://raw.githubusercontent.com/zhenyi4/codi/main/scripts/train_llama1b_gsm8k-aug.sh)
- `src/model.py` defaults [code]: `distill_loss_type="smooth_l1"`, `distill_loss_factor=1.0`, `ref_loss_factor=1.0`, and `fix_attn_mask=False`, described as "Correct a bug about attention mask". The README says to leave it False. The distillation loop iterates `zip(outputs.hidden_states, ref_outputs.hidden_states)`, which covers every layer including the embedding output. It applies the loss to the teacher's detached state, divides by `ref_selected.std()`, and averages over layers. The total is `loss = ce_loss_total + distill_loss_total + ref_ce_loss`. — [model.py](https://raw.githubusercontent.com/zhenyi4/codi/main/src/model.py), [README](https://github.com/zhenyi4/codi)
- Continuing from a checkpoint [code]: `TrainingArguments.restore_from` is documented as "The checkpoint that should be restored from for fine-tuning". It is loaded with `safetensors.load_file(...)` followed by `self.load_state_dict(state_dict)`. `test.py` accepts either `model.safetensors` or `pytorch_model.bin` from `--ckpt_dir`. The HF repo `zen-E/CODI-gpt2` contains only `pytorch_model.bin` (about 813 MB total storage) and was trained on `zen-E/GSM8k-Aug`. — [model.py](https://raw.githubusercontent.com/zhenyi4/codi/main/src/model.py), [test.py](https://raw.githubusercontent.com/zhenyi4/codi/main/test.py), [HF API](https://huggingface.co/api/models/zen-E/CODI-gpt2)
- Data: the training script loads `zen-E/GSM8k-Aug` (or `zen-E/GSM8k-Aug-NL`). The OOD evaluation sets are `juyoung-trl/gsm-hard`, `ChilleD/MultiArith` and `ChilleD/SVAMP`. [code] — [train.py](https://raw.githubusercontent.com/zhenyi4/codi/main/train.py), [test.py](https://raw.githubusercontent.com/zhenyi4/codi/main/test.py)
- Released checkpoints are `zen-E/CODI-gpt2` and `zen-E/CODI-llama3.2-1b-Instruct`. The README gives no training time or expected accuracies. — [README](https://github.com/zhenyi4/codi)

**SIM-CoT (Wei et al., arXiv:2509.20317; code github.com/InternLM/SIM-CoT, "[ICLR 2026]")**
- Paper App. E.2 / Table 6: the GPT-2 CODI settings are lr 3e-3, 40 epochs and batch 128, following CODI. Adam (0.9, 0.999), weight decay 0.1, early-stopping patience 3 epochs. The λ values, LoRA settings, hardware and training time are **not stated**. No seeds or std are reported. — [SIM-CoT](https://arxiv.org/html/2509.20317)
- `CODI/scripts/train_gpt2_gsm8k-aug-decoder-2.sh` [code] is the CODI GPT-2 script line for line: LoRA r128/α32, lr 3e-3, 40 epochs, 64×2, 6 latents, prj_dim 768. It adds `--use_decoder True` and starts from base `gpt2`, with no `restore_from`. So the released SIM-CoT GPT-2 was **trained from scratch, not continued from CODI** (inferred from the script). `explain_loss_factor` (the step loss) defaults to 1.0. — [script](https://raw.githubusercontent.com/InternLM/SIM-CoT/main/CODI/scripts/train_gpt2_gsm8k-aug-decoder-2.sh), [model.py](https://raw.githubusercontent.com/InternLM/SIM-CoT/main/CODI/src/model.py)
- The auxiliary decoder is "architecturally identical to the LLM" and is dropped at inference. — [SIM-CoT](https://arxiv.org/html/2509.20317)
- The HF card for `internlm/SIM_COT-GPT2-CODI` is a template. Its accuracy metrics read "xx.x" and its tags (llama3, multimodal) are wrong. — [HF API](https://huggingface.co/api/models/internlm/SIM_COT-GPT2-CODI)
- The repo README says the scripts are "configuration examples", with hard-coded author paths in `train.py` and `test.py`. — [repo](https://github.com/InternLM/SIM-CoT)

**PCCoT (Wu, Teng, Tu, arXiv:2506.18582, EMNLP 2025 main; code github.com/whyNLP/PCCoT, default branch `dev-publish`)**
- Method:
  - The first pass runs over the query, `<bot>` and c dummy `<latent>` tokens.
  - Then come "T extra iterations". In each, "we update the input vectors of the latent thought tokens as the final hidden state vectors" for **all latents in parallel**, giving T+1 passes in total (inferred from Fig. 1).
  - `<eot>` follows, and the answer is generated after the final iteration.
  - Training uses the CODI teacher/student losses but **without CODI's projection MLP** ("for a fair comparison").
  - Only the `<bot>`/`<latent>`/`<eot>` embeddings and the LoRA (attn+ffn) are trainable.
  - The paper does not say whether gradients flow through all iterations. — [PCCoT](https://arxiv.org/html/2506.18582v2)
- Main setting: PCCoT uses c=24 and T=3. The continuous-CoT baseline uses 12 latents, and Pause uses 24. — [PCCoT](https://arxiv.org/html/2506.18582v2)
- Hyperparameters:
  - LoRA r128, α32, dropout 0.1 on attn+ffn. AdamW, cosine schedule, warmup 0.03, clip 1.0, batch 128.
  - GPT-2: lr 3e-3, weight decay 1e-2, 40 epochs, CODI α=β=γ=1.
  - Llama-1B: lr 8e-4, weight decay 0.1, 10 epochs, γ=20.
  - Greedy decoding. Seeds 0/1/2. — [PCCoT](https://arxiv.org/html/2506.18582v2)
- Compute (Table 2, GPT-2 Small, GSM8K-Aug, training hours / inference seconds): CoT 4.39 / 1.353; Implicit CoT 11.23 / 0.040; Pause 11.88 / 0.041; **Continuous CoT (12 latents) 24.91 / 0.443; PCCoT 13.72 / 0.199**. Training uses "two H800 GPUs" (Table 7 lists A100×2 for GSM8K-NL), and Llama runs use 4×H20. — [PCCoT](https://arxiv.org/html/2506.18582v2)
- The repo releases `whynlp/pccot-gpt2`, `whynlp/pccot-gpt2-nl` and `whynlp/pccot-llama1b`, plus datasets `whynlp/gsm8k-aug` and `whynlp/gsm8k-aug-nl`. Training uses `run_ccot.sh.template` (HF `run_clm.py`-based, `accelerate launch`, flash-attn). The config exposes `loss_alpha/beta/gamma` and `num_iterations`. There is a known bug with Llama eval at batch size >1. — [README](https://raw.githubusercontent.com/whyNLP/PCCoT/dev-publish/README.md)

**KaVa (arXiv:2510.02312, ICLR 2026)**: LoRA r128/α32/dropout 0.1. 24 latents with 3 Jacobi iterations (PCCoT-style). AdamW, cosine schedule, batch 128, clip 2. Llama-1B: α1 (CODI)=10, α2 (KV)=1, smooth-L1 KV loss, R-KV λ=0.1, lr 8e-4, 10 epochs. Three seeds for the best setting. Hardware and compute are not reported. — [KaVa](https://arxiv.org/html/2510.02312v1)

**PMPS (arXiv:2609.09928)**:
- LoRA r128/α32, AdamW, weight decay 0.1, warmup 0.03.
- GPT-2: λ_proto=0.5, batch 256, 6 latents, K=24 prototypes.
- Llama-1B: λ_proto=0.3, batch 512, 8 latents, K=32.
- "All remaining parameters are consistent with CODI."
- 8×A100. Training time and code are not given. — [PMPS](https://arxiv.org/html/2609.09928)

**LOTUS (arXiv:2606.31779; code github.com/yingfan-bot/lotus)**: **full fine-tuning** from each backbone's explicit-CoT checkpoint. Constant lr 1e-4 (GPT-2 and Llama-1B), AdamW, clip 1.0, fp32 for GPT-2, 30 epochs, batch 128, λ_step=0.05. K=6 blocks, c=13 (GPT-2) or 25 (Llama) positions, R=6 loops. Best checkpoint by validation accuracy. Training hardware and time are not reported. — [LOTUS](https://arxiv.org/html/2606.31779v1)

**CoLaR (arXiv:2505.16552; code github.com/xiaomi-research/colar)**: LoRA r128/α32, initialized from CoT-SFT. lr 1e-4, batch 256, up to 50 epochs or 12 h on 8×A100. The RL stage runs on 1 A100. — [CoLaR](https://arxiv.org/html/2505.16552)

**Latent-SFT (arXiv:2510.15522)**: LoRA r64, initialized from CoT-SFT. 8×A100. Llama-1B takes 12.5 h (latent-chain construction) + 12.0 h (Latent-Optim) = 24.5 h. — [Latent-SFT](https://arxiv.org/html/2510.15522v2)

**RiM (arXiv:2605.30343)**: LoRA r128, batch 128 ("about 3,000 update steps per epoch on GSM8K-Aug"), 8×H200. No code. — [RiM](https://arxiv.org/html/2605.30343v1)

### Inferences
- **Continuing from the released CODI GPT-2 is supported in principle** through `--restore_from`. Two changes are needed: (1) convert `pytorch_model.bin` to safetensors, or patch `load_file` to `torch.load`; (2) build the model with the same LoRA r128/α32/prj_dim 768 config so the state-dict keys match. No optimizer or scheduler state is released, so "continuing" means a fresh fine-tune from the weights. Re-warming to the original lr of 3e-3 risks knocking the model off its converged solution. A lower lr (for example 1e-4 to 3e-4) with short warmup is the safer default. This is an untested suggestion.
- The CODI and SIM-CoT scripts use `--save_strategy "no"` (save only at the end). On Colab this must be changed to periodic saving, or a preemption loses the whole run.
- The new mechanism must keep `fix_attn_mask=False` and the same `remove_eos=True` prompt format, or the released checkpoint sees a different input distribution.
- Since SIM-CoT GPT-2 was trained from scratch with an auxiliary GPT-2-sized decoder, "fine-tune SIM-CoT for one more epoch" needs the SIM-CoT fork of the code, not zhenyi4/codi. The two forks differ (`use_decoder`, `explain_loss_factor`).
- PCCoT's Table 2 gives a rough split of where training time goes (inference). CoT-SFT (essentially the teacher pass) takes 4.39 h. Continuous CoT with 12 sequential latents takes 24.91 h. That works out to about 1.7 h per sequential latent over 40 epochs on 2×H800. For 6 latents, the sequential latent loop would then be about 70% of wall-clock (≈10.3 of ≈14.7 h). Any idea that adds sequential work *inside each latent step* (looped layers, decode-and-reinject) will scale mostly with that 70%. A method that multiplies per-step cost by k should cost roughly (0.3 + 0.7k)× a CODI epoch. This is an estimate across two different implementations, not a measurement.

### Gaps
- No paper reports GPU-hours for SIM-CoT, PMPS, KaVa or LOTUS training.
- The exact SIM-CoT λ_step/λ_lm used for the released checkpoint is not stated in the paper. The code default of 1.0 is the best guess.
- Whether CODI's 36 h figure includes evaluation and logging overhead is not stated.
- PCCoT's gradient path through the Jacobi iterations was not checked in code.

## Q5. Rough Colab cost: (a) fine-tune released CODI GPT-2 for 1 epoch of GSM8K-Aug with extra per-step compute; (b) train CODI GPT-2 from scratch

### Takeaway
Using CODI's reported 36 A100-80GB-hours for 40 epochs gives **about 0.9 A100-h per epoch** (about 3,000 optimizer steps at batch 128). At the most recent third-party Colab rate (A100-80GB about 7.52 compute units/h, about $0.10/unit, measured March 2026), the estimates are:
- **(a)** One fine-tuning epoch with 2–4× per-latent-step compute costs about **1.5–2.8 A100-h ≈ 11–21 units ≈ $1–2**. It is about double that if the A100 rate is the 10–15 units/h other sources report.
- **(b)** A from-scratch CODI GPT-2 run (40 epochs) costs about **36 A100-h ≈ 270 units ≈ $27**. At the higher rates it is about $38–54. With 2–4× per-step compute it reaches about $46–84, or up to about $170.
- Three seeds triple these figures.

So fine-tuning experiments are cheap. Full from-scratch training with multiple seeds is the expensive part. On Colab it also needs checkpointing across sessions.

### Cited Findings
- CODI GPT-2: 40 epochs, about 36 h on one A100-80GB. Llama-3.2-1B: 10 epochs, about 48 h. — [CODI](https://arxiv.org/html/2502.21074)
- Batch 128 gives "about 3,000 update steps per epoch on GSM8K-Aug". — [RiM](https://arxiv.org/html/2605.30343v1). The GSM8K-Aug training set has about 385k examples. — [LOTUS](https://arxiv.org/html/2606.31779v1). iCoT lists 378k. — [iCoT](https://arxiv.org/html/2405.14838)
- PCCoT Table 2 (GPT-2, 2×H800, full run): CoT-SFT 4.39 h; continuous CoT with 12 latents 24.91 h; PCCoT 13.72 h. — [PCCoT](https://arxiv.org/html/2506.18582v2)
- Colab compute-unit rates checked 2026-03-10: T4 1.19/h, L4 1.71/h, A100-40GB 5.40/h, A100-80GB 7.52/h, G4 8.71/h. Units cost $10 per 100. — [McCormick](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/). Conflicting figures: A100 "10–15 units/hr" — [Thunder Compute](https://www.thundercompute.com/blog/runpod-vs-google-colab-pro-gpu-comparison); A100-80GB 10.59 units/h measured in 2024 — [varlog](https://varlog-info.pages.dev/colab-credit-consumption-comparison-2024/). Colab Pro costs $9.99/month for 100 units and Pro+ $49.99 for 600. — [Spheron](https://www.spheron.network/blog/google-colab-alternatives-8-gpu-clouds-compared-2026/). All are third-party sources, and the live rate is shown in the Colab resources panel.

### Inferences
- **(a) 1-epoch fine-tune from released CODI GPT-2**:
  - At baseline per-step cost: 0.9 A100-80GB-h ≈ 6.8 units ≈ $0.68 at 7.52 units/h, or ≈ $1.0–1.4 at 10.6–15 units/h.
  - With extra per-step compute, the time factor is about (0.3 + 0.7k) (see Q4). k=2 gives 1.7× ≈ 1.5 h. k=4 gives 3.1× ≈ 2.8 h. Looped layers at 2× depth per step, or decode-and-reinject with about 3 decoded tokens per step, fall in this range. That is about 11–21 units ($1.1–2.1), or $2.3–4.2 at 15 units/h.
  - Running 3 seeds × 3 ideas × 2 settings ≈ 18 runs would cost about $20–75. These are rough figures.
  - On an **L4** (1.71 units/h) there is no published GPT-2 CODI timing. If GPT-2-small training at batch 64 is latency-bound (many small sequential single-token forwards), the L4 may be only modestly slower than the A100 and cheaper per epoch. This is untested; measure the time for 50 steps first.
- **(b) CODI GPT-2 from scratch**: 40 epochs × 0.9 h = 36 A100-80GB-h ≈ 271 units ≈ $27 at 7.52 units/h. At 10.6–15 units/h it is $38–54. With the new method's 1.7–3.1× factor it is ≈ $46–84, or $65–170 at the higher rates. Three seeds roughly triple it. Long runs span several Colab sessions, so the CODI scripts' `save_strategy "no"` must be changed to periodic checkpointing.
- **Port to LLaMA-3.2-1B**: CODI reports 48 A100-h for 10 epochs (≈ 4.8 A100-h per epoch). One epoch ≈ 36 units (≈ $3.6). From scratch ≈ 361 units (≈ $36) at baseline compute, before any extra-compute factor.
- Our budget is small. The cheapest credible protocol is fine-tuning from the released CODI and SIM-CoT checkpoints, plus a **same-compute control**: the same epoch of fine-tuning with the original mechanism. Without it, the gain cannot be attributed to the mechanism rather than to one extra epoch. From-scratch training should be kept for the single best idea.

### Gaps
- No primary Google source gives units/h per GPU type. The A100 rate differs by about 2× across sources. No units/h figure for the H100 or G4 was found in an official source.
- No published per-step timing exists for CODI on an L4 or a 40GB A100. Batch 64 at max length 512 on GPT-2 fits easily on 40GB (inference, not checked).
- The (0.3 + 0.7k) cost model is extrapolated from PCCoT's implementation and hardware, not measured on CODI.

## Q6. Held-out / OOD evaluations used and typical numbers

### Takeaway
The standard OOD suite, used by CODI, SIM-CoT, PMPS, LOTUS, CoLaR and Latent-SFT, is **SVAMP, GSM-Hard and MultiArith**, evaluated zero-shot after GSM8K-Aug training. KaVa reports GSM8K-Hard and SVAMP only. RiM reports GSM-Hard only. PCCoT reports GSM8K only, with no OOD results. At GPT-2 scale the typical scores are SVAMP about 40–43 (PMPS 48.3), GSM-Hard about 9–10.5 and MultiArith about 90–97 for CODI-family models. MultiArith and SVAMP are small (the PMPS values suggest 180 and 300 questions), so differences of 1–2 points there are within noise.

### Cited Findings

**GPT-2, trained on GSM8K-Aug (SVAMP / GSM-Hard / MultiArith)**

| Method | SVAMP | GSM-Hard | MultiArith | Source |
|---|---|---|---|---|
| No-CoT-SFT | 16.4 | 4.3 | 41.1 | CODI Table 1 [T] — [CODI](https://arxiv.org/html/2502.21074) |
| CoT-SFT | 41.8 / 41.6 / 40.00 | 9.8 / 9.0 / 9.25 | 90.7 / 85.0 / 92.22 | CODI T1 / SIM-CoT T1 / PMPS T1 [T] |
| iCoT | 29.4 | 5.7 | 55.5 | CODI T1 [T] |
| Coconut | 36.4 / 36.2 / 34.67 | 7.9 / 8.1 / 7.58 | 82.2 / 83.5 / 82.20 | CODI T1 / SIM-CoT T1 / PMPS T1 [T] |
| CODI | **42.9** / 41.7 / 40.00 | **9.9** / 9.4 / 9.40 | **92.8** / 93.0 / 92.22 | CODI T1 / SIM-CoT T1 / PMPS T1 [T] |
| SIM-CoT on CODI | 42.6 / 43.00 | 9.4 / 9.40 | 92.8 / 93.89 | SIM-CoT T1 / PMPS T1 [T] |
| SIM-CoT on Coconut | 40.7 | 9.3 | 90.8 | SIM-CoT T1 [T] |
| PMPS | 48.33 | 10.46 | 96.67 | PMPS T1 [T] — [PMPS](https://arxiv.org/html/2609.09928) |
| LOTUS | 41.8±0.9 | 9.5±0.2 | 92.4±1.4 | LOTUS T1 [T] — [LOTUS](https://arxiv.org/html/2606.31779v1) |
| RiM (any block) | – | 9.4±0.0 | – | RiM T4 [T] — [RiM](https://arxiv.org/html/2605.30343v1) |

**Llama-3.2-1B, trained on GSM8K-Aug (SVAMP / GSM-Hard / MultiArith)**: CoT-SFT 66.7 / 15.6 / 99.3 (CODI T1) and 65.7 / 13.9 / 96.7 (SIM-CoT T2). CODI 61.1 / 12.8 / 96.1 (CODI T1) and 60.6 / 11.9 / 95.0 (SIM-CoT T2). SIM-CoT 61.5 / 12.7 / 96.2. PMPS 63.67 / 13.42 / 97.22. KaVa 58.9±0.5 SVAMP / 12.7±0.1 GSM-Hard. LOTUS 60.9±0.6 / 12.7±0.5 / 98.3±2.0. Latent-SFT(2) 61.3 / 12.6 / 96.8. CoLaR-2 54.9 / 9.08 / 91.3. [T] — [CODI](https://arxiv.org/html/2502.21074), [SIM-CoT](https://arxiv.org/html/2509.20317), [PMPS](https://arxiv.org/html/2609.09928), [KaVa](https://arxiv.org/html/2510.02312v1), [LOTUS](https://arxiv.org/html/2606.31779v1), [Latent-SFT](https://arxiv.org/html/2510.15522v2), [CoLaR](https://arxiv.org/html/2505.16552)

- The CODI code evaluates OOD on HF datasets `ChilleD/SVAMP`, `juyoung-trl/gsm-hard` and `ChilleD/MultiArith`, selected through `data_name` "svamp", "gsm-hard" and "multi-arith". [code] — [test.py](https://raw.githubusercontent.com/zhenyi4/codi/main/test.py), [README](https://github.com/zhenyi4/codi)
- PMPS's values (e.g. 96.67, 92.22, 48.33, 40.00) are multiples of 1/180 and 1/300. This is consistent with 180-question MultiArith and 300-question SVAMP test splits (inference from the numbers, not stated). — [PMPS](https://arxiv.org/html/2609.09928)
- CODI also evaluates commonsense (CommonsenseQA-GPT4omini, StrategyQA) through the same code path. [code] — [train.py](https://raw.githubusercontent.com/zhenyi4/codi/main/train.py)

### Inferences
- The harness should add SVAMP, GSM-Hard and MultiArith through the same HF datasets as CODI's `test.py`. It should also report the **average over the three OOD sets**, which is the SIM-CoT/PMPS/LOTUS convention ("Avg Acc"), so numbers line up with Tables 1 of those papers.
- At about 300 and 180 questions, one SVAMP question is 0.33 points and one MultiArith question is 0.56 points. Their binomial SE near p≈0.42 and p≈0.93 is about 2.8 and 1.9 points, so a single-run OOD gain under about 3 points is not evidence. GSM-Hard (1,319 questions, p≈0.1) has an SE of about 0.8 points.

### Gaps
- The exact OOD test-set sizes were not confirmed from the datasets' cards.
- No paper reports OOD numbers for PCCoT at GPT-2 scale.

## Q7. Statistical practice (seeds, variance) and typical seed-to-seed spread for CODI GPT-2

### Takeaway
Statistical practice is weak and inconsistent:
- CODI, SIM-CoT and PMPS report **single runs** with no variance.
- PCCoT (3 seeds, mean±std), KaVa (3 seeds, SE), LOTUS (seeds ± std; the caption says "33", probably 3), CoLaR and Latent-SFT (5 seeds, 95% CI) and RiM (SE over 16 split repeats) report some spread.

No paper reports a multi-seed spread for CODI GPT-2 itself. The best proxies are as follows:
- PCCoT's CODI-style continuous CoT on GPT-2 has a seed std of **1.61** (2.58 on NL).
- Coconut GPT-2 has a std of 1.5.
- LOTUS GPT-2 has a std of 0.7–1.2, rising to 4.5 for unstable variants.
- CODI's own 5-run robustness test implies a std of about 0.9.
- Independent reruns of CODI GPT-2 range from 40.86 to 43.7, a spread of about 2.8 points.

Plan for a **seed std of about 1–1.6 points** on the 1,319-question test set. On top of that comes about 2.2 points of binomial SE on a 500-question split.

### Cited Findings
- CODI: "deterministic models, and their results are reported from a single run". CoT-SFT is "sampled 10 times and their average is reported (temperature=0.1)". The only std is from "5 independent runs" in the answer-prompt robustness test, with a ±2×std band of 1.8 around a 39% baseline, which implies a std of about 0.9. — [CODI](https://arxiv.org/html/2502.21074)
- SIM-CoT: no seeds, std or CIs reported. — [SIM-CoT](https://arxiv.org/html/2509.20317)
- PMPS: no seeds or std reported. — [PMPS](https://arxiv.org/html/2609.09928)
- PCCoT: results "averaged over 3 random runs with standard deviations also shown", seeds 0/1/2.
  - GPT-2 stds: PCCoT 0.31 (NL 0.80); Continuous CoT 1.61 (NL 2.58); Pause 0.46 (NL 2.10); Implicit CoT 0.31 (NL 1.10).
  - Llama-1B stds: PCCoT 0.18 (NL 1.39); Continuous CoT 0.68 (NL 1.40).
  - The authors also report "negligible difference when using different decoding strategies". — [PCCoT](https://arxiv.org/html/2506.18582v2)
- KaVa: a hyperparameter sweep, then three random seeds for the best setting, reported as standard error (e.g. Llama-1B 56.5 (0.4)). Baselines have no error bars. — [KaVa](https://arxiv.org/html/2510.02312v1)
- LOTUS Table 1: "mean ± std across 33 seeds", probably a rendering error for 3 seeds. GPT-2: LOTUS 44.1±0.7, LOTUS+CODI 43.1±1.2, LOTUS-aux 35.5±4.5. Llama-1B: 57.3±0.2 and 58.3±0.8. — [LOTUS](https://arxiv.org/html/2606.31779v1)
- CoLaR: five runs with seeds 0–4, reported as 95% CI. Inference uses temperature 1 and top-p 0.9, so the CI mixes seed and sampling noise. — [CoLaR](https://arxiv.org/html/2505.16552). Latent-SFT: five seeds, 95% CI. — [Latent-SFT](https://arxiv.org/html/2510.15522v2)
- RiM: SE over 16 split repeats rather than over training seeds (e.g. GPT-2 CoT 39.8±0.2). — [RiM](https://arxiv.org/html/2605.30343v1)
- Coconut GPT-2: 34.1±1.5. [P] — [Coconut](https://arxiv.org/html/2412.06769)
- Cross-paper reruns of CODI GPT-2: 43.7 (authors), 42.0 (SIM-CoT), 40.86 (PMPS). — [CODI](https://arxiv.org/html/2502.21074), [SIM-CoT](https://arxiv.org/html/2509.20317), [PMPS](https://arxiv.org/html/2609.09928)

### Inferences
- On our 500-question validation split at p≈0.41, the binomial SE is √(0.41·0.59/500) ≈ 2.2 points (≈ 11 questions). On the full 1,319-question test set it is ≈ 1.35 points. Adding a seed std of about 1–1.6 points, an unpaired single-seed difference must be above about 5 points to be convincing.
- Because the eval items are shared, a **paired** test (McNemar or a paired bootstrap over questions) between the base checkpoint and the fine-tuned variant is much tighter than an unpaired one. It should be the primary statistic, together with ≥3 fine-tuning seeds. That matches PCCoT's and KaVa's 3-seed practice and beats CODI's, SIM-CoT's and PMPS's single runs.
- Reporting both the 500-question validation split (for selection) and the 1,319-question GSM8K test (for final numbers) keeps results comparable with the published tables. All the published tables are on test, and none of the papers read says it tuned on validation only (LOTUS does select checkpoints by validation accuracy).
- To claim a gain over the released 6-step CODI or SIM-CoT, a reasonable preregistered bar is: about **+3 points on GSM8K test, mean over 3 seeds, paired p<0.05**, measured against a same-compute control fine-tune. This is a suggestion derived from the spreads above, not a field standard.

### Gaps
- No direct multi-seed measurement of CODI GPT-2 trained with the official recipe was found. The 0.9 std comes from a prompt-robustness test, not from training seeds.
- How many seeds Coconut's ±1.5 uses was not checked today.
