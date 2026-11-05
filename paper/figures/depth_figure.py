"""Writes paper/figures/depth.tex: primary-pool accuracy against latent steps for the four held-out models.

    python3 paper/figures/depth_figure.py

Reads results/latent_gate/heldout_scale/<model>/ and <model>-anchor/ (gsm8k_test, gsm_hard and multiarith). Every policy
coincides with `plain` at 6 steps, since no bias or move applies until there are more than 6 latent entries. The shaded band
is the plan's 2-point noninferiority margin below `plain`@6. Colors are the first three slots of the dataviz reference
palette, validated all-pairs in light mode; markers differ too, so the lines also read in grayscale.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "results/latent_gate/heldout_scale"
PRIMARY = ("gsm8k_test", "gsm_hard", "multiarith")
MODELS = [("simcot-gpt2", "SIM-CoT GPT-2"), ("codi-gpt2", "CODI GPT-2"), ("codi-llama1b", "CODI-1B"),
          ("simcot-llama1b", "SIM-CoT-1B")]
SERIES = [("plain", "plain", "series1", "*"), ("dedup_soft", "soft dedup", "series2", "square*"),
          ("dedup_anchor", "soft dedup + re-anchoring", "series3", "triangle*")]
SPAN = 6.5  # points shown on every panel's y axis, so slopes compare across panels


def load(folder):
    if not folder.exists():
        return None
    return [json.loads(line) for dataset in PRIMARY for line in (folder / f"{dataset}.jsonl").read_text().splitlines()]


def accuracy(rows, policy, depth):
    return 100 * sum(r[policy][str(depth)]["correct"] for r in rows) / len(rows)


panels = []
for model, title in MODELS:
    base, anchored = load(FOLDER / model), load(FOLDER / f"{model}-anchor")
    six = accuracy(base, "plain", 6)
    lines = {"plain": [(d, accuracy(base, "plain", d)) for d in (6, 12, 24)],
             "dedup_soft": [(6, six)] + [(d, accuracy(base, "dedup_soft", d)) for d in (12, 24)]}
    if anchored is not None:
        lines["dedup_anchor"] = [(6, six)] + [(d, accuracy(anchored, "dedup_anchor", d)) for d in (12, 24)]
    top = six + 0.5
    panels.append((title, six, top - SPAN, top, lines))

out = [r"% Written by paper/figures/depth_figure.py; do not edit by hand.",
       r"\definecolor{series1}{HTML}{2A78D6}\definecolor{series2}{HTML}{EB6834}\definecolor{series3}{HTML}{1BAF7A}",
       r"\definecolor{inkmuted}{HTML}{52514E}",
       r"\begin{tikzpicture}",
       r"\begin{groupplot}[group style={group size=4 by 1, horizontal sep=1.0cm},",
       r"  width=0.28\textwidth, height=0.21\textwidth, xmode=log, log basis x=2, xmin=5.2, xmax=27.5,",
       r"  xtick={6,12,24}, xticklabels={6,12,24}, xlabel={latent steps}, tick label style={font=\scriptsize},",
       r"  label style={font=\scriptsize, text=inkmuted}, title style={font=\small, yshift=-0.6ex},",
       r"  axis line style={inkmuted!60}, tick style={inkmuted!60}, ymajorgrids, grid style={inkmuted!15},",
       r"  yticklabel={\pgfmathprintnumber[fixed, precision=0]{\tick}}, ytick distance=2,",
       r"  every axis plot/.append style={line width=1pt, mark size=1.8pt}]"]
for i, (title, six, low, high, lines) in enumerate(panels):
    options = [f"title={{{title}}}", f"ymin={low:.2f}", f"ymax={high:.2f}"]
    if i == 0:
        options += [r"ylabel={accuracy (\%)}", r"legend to name=depthlegend",
                    r"legend style={font=\scriptsize, draw=none, legend columns=4, column sep=0.8em}"]
    out.append(rf"\nextgroupplot[{', '.join(options)}]")
    out.append(rf"\addplot[draw=none, fill=inkmuted!12, forget plot] coordinates {{(5.2,{six - 2:.2f}) (27.5,{six - 2:.2f}) "
               rf"(27.5,{six:.2f}) (5.2,{six:.2f})}} --cycle;")
    for policy, label, color, mark in SERIES:
        if policy in lines:
            points = " ".join(f"({d},{a:.2f})" for d, a in lines[policy])
            out.append(rf"\addplot[color={color}, mark={mark}] coordinates {{{points}}};")
            if i == 0:
                out.append(rf"\addlegendentry{{{label}}}")
        elif i == 0:
            out.append(rf"\addlegendimage{{color={color}, mark={mark}, line width=1pt, mark size=1.8pt}}")
            out.append(rf"\addlegendentry{{{label}}}")
    if i == 0:
        out.append(r"\addlegendimage{only marks, mark=square*, mark size=3pt, color=inkmuted!18}")
        out.append(r"\addlegendentry{within 2 points of 6 steps}")
out += [r"\end{groupplot}", r"\node[anchor=south] at ($(group c1r1.north west)!0.5!(group c4r1.north east) + (0,0.55cm)$)",
        r"  {\pgfplotslegendfromname{depthlegend}};", r"\end{tikzpicture}"]
(Path(__file__).parent / "depth.tex").write_text("\n".join(out) + "\n")
for title, six, _, _, lines in panels:
    print(f"{title}: " + "; ".join(f"{p} " + " ".join(f"{d}:{a:.1f}" for d, a in v) for p, v in lines.items()))
