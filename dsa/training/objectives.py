"""Matched same-backbone answer, explicit, gate, candidate and answer-distillation objectives."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dsa.models.backbone import position_ids


@dataclass(frozen=True)
class LossConfig:
    cot: float = 1.0
    gate: float = 0.1
    anchor: float = 0.1
    gate_positive_weight: float = 1.0
    answer_distill: float = 0.0


def normalized_alignment(candidates, targets):
    """Detach only the target route; normalize without learned affine parameters."""
    width = candidates.shape[-1]
    return F.mse_loss(F.layer_norm(candidates, (width,)),
                      F.layer_norm(targets.detach(), (width,)))


def answer_distillation(student_layers, teacher_layers):
    """CODI's answer-position loss: per layer, smooth L1 over the teacher's std; layer mean.

    Each layer gives one position's [width] state. The teacher is detached, as in
    CODI. CODI divides by the std over its whole batch; this uses each record's
    own std, so a batch gives the same loss as its records one at a time.
    """
    losses = []
    for student, teacher in zip(student_layers, teacher_layers, strict=True):
        teacher = teacher.detach()
        losses.append(F.smooth_l1_loss(student, teacher) / teacher.std())
    return torch.stack(losses).mean()


def _padded(rows, fill, device):
    width = max(len(row) for row in rows)
    values = torch.tensor([row + [fill] * (width - len(row)) for row in rows], device=device, dtype=torch.long)
    mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows],
                        device=device, dtype=torch.long)
    return values, mask


def _mean(values):
    # A single record keeps its exact value and graph.
    return values[0] if len(values) == 1 else torch.stack(values).mean()


def joint_objective(model, records, answer_prefix_ids, loss_config=LossConfig()):
    """One record or a right-padded batch; shared backbone, no forced milestone writes.

    Every loss is a per-record mean averaged over the records, so a batch gives
    the same objective as accumulating its records one at a time. Answer
    distillation needs each record's ``answer_alignment``
    (``dsa.data.tokenize.answer_alignment``).
    """
    records = [records] if isinstance(records, dict) else list(records)
    if not records:
        raise ValueError("At least one training record is required")
    device = model.initial_anchor.device
    prefix = list(answer_prefix_ids)
    for record in records:
        if record["latent_answer_prefix_ids"] != prefix:
            raise ValueError("Training answer prompt must match question-only generation")
        if (record["latent_suffix_input_ids"][:len(prefix)] != prefix
                or any(label != -100 for label in record["latent_suffix_labels"][:len(prefix)])):
            raise ValueError("Latent answer prefix must remain fixed and masked")
    distill = loss_config.answer_distill > 0
    if distill and not all("answer_alignment" in record for record in records):
        raise ValueError("Answer distillation needs each record's answer_alignment")

    def batch(key, fill=model.config.pad_token_id):
        return _padded([record[key] for record in records], fill, device)

    def shifted_cross_entropy(logits, labels, key):
        return _mean([F.cross_entropy(logits[index, :len(record[key]) - 1],
                                      labels[index, 1:len(record[key])], ignore_index=-100)
                      for index, record in enumerate(records)])

    explicit_ids, explicit_mask = batch("explicit_input_ids")
    teacher = model.backbone.forward_embeds(
        model.backbone.embeddings(explicit_ids), explicit_mask, position_ids(explicit_mask),
        all_layers=distill)
    explicit_hidden = teacher[-1] if distill else teacher
    explicit_logits = model.backbone.logits(explicit_hidden)
    explicit_labels, _ = batch("explicit_labels", -100)
    cot_loss = shifted_cross_entropy(explicit_logits, explicit_labels, "explicit_input_ids")
    question_ids, question_mask = batch("question_input_ids")
    loop = model(question_ids, question_mask)
    suffix, suffix_mask = batch("latent_suffix_input_ids")
    suffix_labels, _ = batch("latent_suffix_labels", -100)
    readout = model.answer_logits(loop, suffix, suffix_mask, all_layers=distill)
    answer_logits, student = readout if distill else (readout, None)
    answer_loss = shifted_cross_entropy(answer_logits, suffix_labels, "latent_suffix_input_ids")
    gate_terms, anchor_terms = [], []
    for index, record in enumerate(records):
        gate_logits = loop.gate_logits[index:index + 1]
        candidates = loop.candidates[index:index + 1]
        if not record["controller_supervised"]:
            gate_terms.append(gate_logits.sum() * 0.0)
            anchor_terms.append(candidates.sum() * 0.0)
            continue
        if len(record["gate_targets"]) != gate_logits.shape[1]:
            raise ValueError("Training labels must match the configured fixed loop budget")
        labels = torch.tensor(record["gate_targets"], device=device, dtype=gate_logits.dtype)[None]
        weight = labels.new_tensor(loss_config.gate_positive_weight)
        gate_terms.append(F.binary_cross_entropy_with_logits(gate_logits, labels, pos_weight=weight))
        target_positions = record["checkpoint_positions"]
        event_loops = [value - 1 for value in record["event_loops"]]
        if len(target_positions) != len(event_loops) or not event_loops:
            raise ValueError("Checkpoint targets and scheduled events must agree")
        targets = explicit_hidden[index:index + 1, target_positions, :].detach()
        anchor_terms.append(normalized_alignment(candidates[:, event_loops, :], targets))
    gate_loss, anchor_loss = _mean(gate_terms), _mean(anchor_terms)
    total = answer_loss + loss_config.cot * cot_loss + loss_config.gate * gate_loss + loss_config.anchor * anchor_loss
    distill_loss = answer_loss.new_zeros(())
    if distill:
        # The student reads the answer after the question, anchor and workspace.
        start = question_ids.shape[1] + 1 + model.config.workspace_size
        terms = []
        for index, record in enumerate(records):
            teacher_index, prefix_index = record["answer_alignment"]
            if record["explicit_input_ids"][teacher_index] != record["latent_suffix_input_ids"][prefix_index]:
                raise ValueError("Answer alignment must pair the same token in both routes")
            terms.append(answer_distillation([layer[index, start + prefix_index] for layer in student],
                                             [layer[index, teacher_index] for layer in teacher]))
        distill_loss = _mean(terms)
        total = total + loss_config.answer_distill * distill_loss
    return {"total": total, "answer": answer_loss, "cot": cot_loss,
            "gate": gate_loss, "anchor": anchor_loss, "distill": distill_loss, "loop": loop}
