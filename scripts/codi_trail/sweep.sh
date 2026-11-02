#!/bin/bash
# CODI's published weights at several latent step counts on gsm8k_valid, through the shared harness.
# Rows go to results/codi_trail/sweep, not the main leaderboard. DEVICE overrides mps.
set -e
cd "$(dirname "$0")/../.."
for n in 6 12 24 3 1; do
  python scripts/evaluate.py --model codi --arg latent_iterations=$n --name codi-iter$n --datasets gsm8k_valid \
    --device "${DEVICE:-mps}" --results-dir results/codi_trail/sweep
done
