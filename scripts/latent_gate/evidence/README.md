# Evidence scripts behind docs/LATENT_GATE_IDEA.md

Exploratory runs from 2 November 2026, before Stage 0, kept as they ran. All use `gsm8k_valid` on MPS.

| Script | What it produced | Records |
|---|---|---|
| `probe_ckpt.py` | `scripts/codi_trail/probe.py` with a `--checkpoint` option, run on SIM-CoT GPT-2 | `results/latent_gate/simcot_probe/gsm8k_valid.jsonl` (latents not committed) |
| `compare.py` | CODI against SIM-CoT: accuracy by depth, cycle, settling, content decay | `results/latent_gate/simcot_probe/compare.txt` |
| `attn_copy.py`, `copy_report.py` | Where the answer attends, and the copied-entry test (plan fixed first) | `results/latent_gate/copy_test/` |
| `decode_latents.py`, `decode_baseline.py` | Logit-lens decodes matched to gold intermediate results, with a chance baseline and accuracy by rationale length | printed only; the label check in `scripts/latent_gate/stage0_labels.py` supersedes them |

```bash
python scripts/latent_gate/evidence/probe_ckpt.py --checkpoint <SIM-CoT dir> --dataset gsm8k_valid --output <out.jsonl>
python scripts/latent_gate/evidence/compare.py results/codi_trail/probe/gsm8k_valid.jsonl <out.jsonl>   # needs both latents files
python scripts/latent_gate/evidence/attn_copy.py --repo . --dataset gsm8k_valid --output <file> [--checkpoint <dir>]
python scripts/latent_gate/evidence/copy_report.py <folder> results/codi_trail/probe/gsm8k_valid.jsonl <SIM-CoT probe records>
```

`python scripts/latent_gate/fetch_simcot.py --output <dir>` writes the SIM-CoT checkpoint directory these take.
