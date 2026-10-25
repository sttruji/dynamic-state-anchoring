"""Reproducible data setup and a fixed evaluation harness for DSA and baselines.

Data: ``scripts/setup_data.py`` downloads pinned sources, normalizes, tags and
splits them, and checks every output against ``provenance/bench_data_lock.json``.
Evaluation: ``scripts/evaluate.py`` runs any :class:`dsa.bench.models.BenchModel`
over named datasets and keeps a persistent leaderboard under ``results/``.
"""
