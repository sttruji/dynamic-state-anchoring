"""Report on inner-loop evaluations: Stage 0's checks and Stage 1's go/no-go (docs/brainstorm/INNER_LOOP_QUICK_REFERENCE.md).

    python scripts/inner_loop/report.py <evals dir> [--training <arms dir>]

<evals dir>/<name>/gsm8k_valid.jsonl as eval.py writes them: base-simcot and base-codi for Stage 0; ft6, ft12, loop and
loop_k1 for Stage 1; ft6_step, ft6_ce, loop_ce and kk_ce for Stage 1b. <arms dir>/<arm>/metrics.jsonl gives the final
size of the loop gate and of KK-LSE's alpha.

Stage 1, decided at the preregistered k = 3 on the 500 validation questions:
    go          loop@k3 - max(ft6 plain, ft12 plain@12) >= +10, loop k3 - k1 >= +5 in the same model, and the gate off zero
    ambiguous   a gain of +5 to +9: one more seed of loop and ft12 before deciding
    stop        a gain of +4 or less, a gate at zero, or no rise from k1 to k3

Stage 1b, at batch 128, with the same go / ambiguous / stop thresholds:
    recipe      ft6_ce (answer loss only) vs ft6_step (with the step loss), reported, no verdict
    loop        loop_ce@k3 vs ft6_ce, with k3 vs k1 as the second check and the gate off zero
    KK-LSE      kk_ce vs ft6_ce, with the bias on vs off at test (nokk) as the second check
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("evals", type=Path)
parser.add_argument("--training", type=Path)
parser.add_argument("--k", type=int, default=3, help="the preregistered loop count")
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from looped import arithmetic, parse_step  # noqa: E402


def load(name):
    path = args.evals / name / "gsm8k_valid.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else None


def sign_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def correct(records, arm):
    return [r["arms"][arm]["correct"] for r in records]


def paired(a, b):
    wins = sum(x and not y for x, y in zip(a, b))
    losses = sum(y and not x for x, y in zip(a, b))
    return sum(a) - sum(b), wins, losses, sign_p(wins, losses)


def rule(gain: int, check: int, ok: bool) -> str:
    """The preregistered pilot rule: GO at a gain of +10 with the second check at +5, AMBIGUOUS at +5 to +9 with the
    second check above 0, else STOP."""
    if gain >= 10 and check >= 5 and ok:
        return "GO"
    if 5 <= gain <= 9 and check > 0 and ok:
        return "AMBIGUOUS"
    return "STOP"


def final_metric(arm, key):
    path = args.training / arm / "metrics.jsonl" if args.training else None
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text().splitlines()[-1]).get(key)


def versus(label, a, b):
    diff, wins, losses, p = paired(a, b)
    print(f"  {label}: {sum(a)} vs {sum(b)}: {diff:+d} (+{wins}/-{losses}, p={p:.3f})")
    return diff, p


def number(text):
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def table(name, records):
    print(f"\n{name}: n={len(records)}")
    arms = list(records[0]["arms"])
    for arm in arms:
        hits = correct(records, arm)
        long = [r["arms"][arm]["correct"] for r in records if r["rationale_steps"] >= 4]
        line = f"  {arm:12} {sum(hits):4} ({sum(long)}/{len(long)})"
        if "plain" in arms and arm != "plain":
            diff, wins, losses, p = paired(hits, correct(records, "plain"))
            same = sum(r["arms"][arm]["prediction"] == r["arms"]["plain"]["prediction"] for r in records)
            drifts = [r["arms"][arm]["latent_drift"] for r in records if "latent_drift" in r["arms"][arm]]
            line += f"   vs plain {diff:+d} (+{wins}/-{losses}, p={p:.3f}), same answer {same}"
            if drifts:
                line += f", max drift {max(drifts):.1e}"
        print(line)


def identity(name, records):
    """Stage 0: with zero loop parameters every loopK arm must give plain's answer text on every question, with latents
    equal up to GPU rounding."""
    loops = [a for a in records[0]["arms"] if a.startswith("loop") and "@" not in a]
    for arm in loops:
        same = all(r["arms"][arm]["text"] == r["arms"]["plain"]["text"] for r in records)
        drift = max(r["arms"][arm].get("latent_drift", 0.0) for r in records)
        print(f"  identity {name} {arm}: {'PASS' if same and drift <= 1e-5 else 'FAIL'} (max drift {drift:.1e})")


def decoder_reliability(records, arm="plain"):
    """Per slot: how often the decode parses, is self-consistent, and matches the gold step."""
    if "steps" not in records[0]["arms"][arm]:
        return
    print(f"  step decoder on {arm} latents (slot: parsed, consistent of parsed, exact of gold-scored)")
    for s in range(len(records[0]["arms"][arm]["steps"])):
        steps = [r["arms"][arm]["steps"][s] for r in records]
        parsed = sum(x["parsed"] for x in steps)
        consistent = sum(bool(x["consistent"]) for x in steps)
        scored = [x["exact"] for x in steps if x["exact"] is not None]
        print(f"    {s}: {parsed:3} parsed, {consistent:3} consistent, {sum(scored):3} of {len(scored)} exact")


def errors(records, arm="plain"):
    """Wrong answers split by what the decoded latents held: the answer (readout), a bad sum (arithmetic) or neither."""
    if "steps" not in records[0]["arms"][arm]:
        return
    kinds = {"readout": 0, "arithmetic": 0, "plan": 0}
    for r in records:
        if r["arms"][arm]["correct"]:
            continue
        parses = [parse_step(x["text"]) for x in r["arms"][arm]["steps"]]
        gold = number(r["gold"])
        if any(p and number(p[1]) is not None and gold is not None and abs(number(p[1]) - gold) < 1e-6 for p in parses):
            kinds["readout"] += 1
        elif any(p and arithmetic(p[0]) is not None and number(p[1]) is not None
                 and abs(arithmetic(p[0]) - number(p[1])) > 1e-6 for p in parses):
            kinds["arithmetic"] += 1
        else:
            kinds["plan"] += 1
    print(f"  wrong answers on {arm}: {kinds['readout']} readout (a decoded step holds the answer), "
          f"{kinds['arithmetic']} arithmetic (a decoded step miscalculates), {kinds['plan']} plan (neither)")


for name in ("base-simcot", "base-codi"):
    records = load(name)
    if records:
        table(name, records)
        identity(name, records)
        decoder_reliability(records)
        errors(records)

stage1 = {name: load(name) for name in ("ft6", "ft12", "loop", "loop_k1")}
for name, records in stage1.items():
    if records:
        table(name, records)
        decoder_reliability(records, "loop1" if name.startswith("loop") else "plain")
if stage1["ft6"] and stage1["ft12"] and stage1["loop"]:
    k3, k1 = f"loop{args.k}", "loop1"
    controls = {"ft6@6": correct(stage1["ft6"], "plain"), "ft12@12": correct(stage1["ft12"], "plain@12")}
    best = max(controls, key=lambda c: sum(controls[c]))
    gain, wins, losses, p = paired(correct(stage1["loop"], k3), controls[best])
    dose, dose_wins, dose_losses, dose_p = paired(correct(stage1["loop"], k3), correct(stage1["loop"], k1))
    gate = final_metric("loop", "gate")
    print(f"\nStage 1 at k={args.k}: loop {sum(correct(stage1['loop'], k3))} vs best control {best} {sum(controls[best])}: "
          f"{gain:+d} (+{wins}/-{losses}, p={p:.3f}); k{args.k} vs k1 {dose:+d} (+{dose_wins}/-{dose_losses}, "
          f"p={dose_p:.3f}); mean |gate| {gate}")
    if stage1["loop_k1"]:
        matched, w, l, mp = paired(correct(stage1["loop"], k3), correct(stage1["loop_k1"], "loop1"))
        print(f"  vs the parameter-matched loop_k1: {matched:+d} (+{w}/-{l}, p={mp:.3f})")
    verdict = {"GO": "GO: confirm with three seeds on the 1,319-question test set",
               "AMBIGUOUS": "AMBIGUOUS: one more seed of loop and ft12 before deciding",
               "STOP": "STOP: test the duo-causal loop once, then write up the null"}[
        rule(gain, dose, gate is not None and gate > 1e-3)]
    print(f"  {verdict}" + ("" if gate is not None else " (gate size unknown: pass --training)"))

stage1b = {name: load(name) for name in ("ft6_step", "ft6_ce", "loop_ce", "kk_ce")}
for name, records in stage1b.items():
    if records:
        table(name, records)
        decoder_reliability(records, "loop1" if name.startswith("loop") else "plain")
if stage1b["ft6_ce"]:
    print("\nStage 1b (batch 128): the recipe check, the loop on the answer loss alone, and KK-LSE at 6 steps")
    control = correct(stage1b["ft6_ce"], "plain")
    base = load("base-simcot")
    if base:
        versus("ft6_ce vs the SIM-CoT base", control, correct(base, "plain"))
    if stage1b["ft6_step"]:
        diff, p = versus("recipe: answer loss only (ft6_ce) vs with the step loss (ft6_step)", control,
                         correct(stage1b["ft6_step"], "plain"))
        if p < 0.05:
            print(f"    the step loss {'costs' if diff > 0 else 'adds'} accuracy at this budget")
        else:
            print("    no clear difference between the recipes")
    if stage1b["loop_ce"]:
        loop, k3 = stage1b["loop_ce"], f"loop{args.k}"
        gain, _ = versus(f"loop_ce k={args.k} vs ft6_ce", correct(loop, k3), control)
        dose, _ = versus(f"loop_ce k={args.k} vs k=1", correct(loop, k3), correct(loop, "loop1"))
        gate = final_metric("loop_ce", "gate")
        print(f"    mean |gate| {gate}; loop on the answer loss alone: {rule(gain, dose, gate is not None and gate > 1e-3)}")
    if stage1b["kk_ce"]:
        kk = stage1b["kk_ce"]
        gain, _ = versus("kk_ce vs ft6_ce", correct(kk, "plain"), control)
        own, _ = versus("kk_ce with the bias vs without it at test (nokk)", correct(kk, "plain"), correct(kk, "nokk"))
        print(f"    mean |alpha| {final_metric('kk_ce', 'alpha')}, max {final_metric('kk_ce', 'alpha_max')}; "
              f"KK-LSE at 6 steps: {rule(gain, own, True)}")
