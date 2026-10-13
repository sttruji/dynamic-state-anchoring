"""Question-only evaluation primitives matching pinned LOTUS answer scoring.

Full corpus evaluation must first verify the original split manifest and ordered
IDs. These helpers do not select thresholds, loops, or checkpoints from test data.
"""
import math
import re
import time

import torch


def extract_answer(text):
    """Same extraction rules as LOTUS eb77e2f scripts/eval.py:69-82 (MIT)."""
    parts = text.split("###")
    if len(parts) > 1:
        answer = parts[-1].replace(",", "").strip()
        if answer:
            return answer
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return text.split("#")[-1].replace(",", "").strip()


def score_answer(prediction, answer):
    """Pinned LOTUS numeric comparison, including its string fallback."""
    try:
        return float(prediction) == float(answer)
    except (ValueError, ZeroDivisionError):
        return prediction.strip() == answer.strip()


def predict_question(model, tokenizer, question, *, loop_budget=None, max_new_tokens=64):
    """Accept only question text and predeclared compute/decoding settings."""
    question_ids = tokenizer.encode(question.strip() + "\n", add_special_tokens=False)
    inputs = torch.tensor([question_ids], dtype=torch.long, device=model.initial_anchor.device)
    prefix = tokenizer.encode("\n### ", add_special_tokens=False)
    started = time.perf_counter()
    generated = model.generate(inputs, torch.ones_like(inputs), answer_prefix_ids=prefix,
                               max_new_tokens=max_new_tokens, loop_budget=loop_budget)
    elapsed = time.perf_counter() - started
    text = tokenizer.decode(generated.token_ids[0].tolist(), skip_special_tokens=True)
    prediction = extract_answer(text)
    try:
        valid = math.isfinite(float(prediction))
    except ValueError:
        valid = False
    return {"generated_text": text, "prediction": prediction, "valid_numeric_answer": valid,
            "latency_seconds": elapsed,
            "recurrent_forwards": generated.loop.gate_logits.shape[1],
            "gate_probabilities": generated.loop.gate_logits[0].sigmoid().tolist(),
            "writes": generated.loop.writes[0].tolist()}
