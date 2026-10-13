"""Matched same-backbone answer, explicit, gate and candidate objectives."""
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


def normalized_alignment(candidates, targets):
    """Detach only the target route; normalize without learned affine parameters."""
    width = candidates.shape[-1]
    return F.mse_loss(F.layer_norm(candidates, (width,)),
                      F.layer_norm(targets.detach(), (width,)))


def joint_objective(model, record, answer_prefix_ids, loss_config=LossConfig()):
    """One variable-length record, shared backbone, no forced milestone writes."""
    device = model.initial_anchor.device
    if record["latent_answer_prefix_ids"] != list(answer_prefix_ids):
        raise ValueError("Training answer prompt must match question-only generation")
    if (record["latent_suffix_input_ids"][:len(answer_prefix_ids)] != list(answer_prefix_ids)
            or any(label != -100 for label in record["latent_suffix_labels"][:len(answer_prefix_ids)])):
        raise ValueError("Latent answer prefix must remain fixed and masked")
    def ids(values):
        return torch.tensor(values, device=device, dtype=torch.long)[None]
    explicit_ids = ids(record["explicit_input_ids"])
    explicit_mask = torch.ones_like(explicit_ids)
    explicit_hidden = model.backbone.forward_embeds(
        model.backbone.embeddings(explicit_ids), explicit_mask, position_ids(explicit_mask)
    )
    explicit_logits = model.backbone.logits(explicit_hidden)
    explicit_labels = ids(record["explicit_labels"])
    cot_loss = F.cross_entropy(explicit_logits[:, :-1].reshape(-1, explicit_logits.shape[-1]),
                               explicit_labels[:, 1:].reshape(-1), ignore_index=-100)
    question_ids = ids(record["question_input_ids"])
    loop = model(question_ids, torch.ones_like(question_ids))
    suffix = ids(record["latent_suffix_input_ids"])
    suffix_labels = ids(record["latent_suffix_labels"])
    answer_logits = model.answer_logits(loop, suffix, torch.ones_like(suffix))
    answer_loss = F.cross_entropy(answer_logits[:, :-1].reshape(-1, answer_logits.shape[-1]),
                                  suffix_labels[:, 1:].reshape(-1), ignore_index=-100)
    gate_loss = loop.gate_logits.sum() * 0.0
    anchor_loss = loop.candidates.sum() * 0.0
    if record["controller_supervised"]:
        if len(record["gate_targets"]) != loop.gate_logits.shape[1]:
            raise ValueError("Training labels must match the configured fixed loop budget")
        labels = torch.tensor(record["gate_targets"], device=device, dtype=loop.gate_logits.dtype)[None]
        weight = labels.new_tensor(loss_config.gate_positive_weight)
        gate_loss = F.binary_cross_entropy_with_logits(loop.gate_logits, labels, pos_weight=weight)
        target_positions = record["checkpoint_positions"]
        event_loops = [value - 1 for value in record["event_loops"]]
        if len(target_positions) != len(event_loops) or not event_loops:
            raise ValueError("Checkpoint targets and scheduled events must agree")
        targets = explicit_hidden[:, target_positions, :].detach()
        anchor_loss = normalized_alignment(loop.candidates[:, event_loops, :], targets)
    total = answer_loss + loss_config.cot * cot_loss + loss_config.gate * gate_loss + loss_config.anchor * anchor_loss
    return {"total": total, "answer": answer_loss, "cot": cot_loss,
            "gate": gate_loss, "anchor": anchor_loss, "loop": loop}
