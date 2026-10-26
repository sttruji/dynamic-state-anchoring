"""Question-only, fixed-budget recurrent latent memory with matched write controls."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import torch
from torch import nn

from .backbone import BackboneAdapter, position_ids
from .lora import has_lora, merged_state_dict


@dataclass(frozen=True)
class AnchorConfig:
    workspace_size: int = 4
    loop_budget: int = 16
    gate_threshold: float = 0.5
    write_mode: str = "learned"
    periodic_interval: int = 4
    pad_token_id: int = 0
    eos_token_id: int = 1
    marker_token_id: int | None = None
    recurrence_mode: str = "standard"

    def __post_init__(self):
        if self.workspace_size < 1 or self.loop_budget < 1 or self.periodic_interval < 1:
            raise ValueError("Workspace, loop budget and periodic interval must be positive")
        if not 0 <= self.gate_threshold <= 1:
            raise ValueError("Gate threshold must be in [0, 1]")
        if self.write_mode not in {"static", "every_loop", "periodic", "learned"}:
            raise ValueError("Unknown write policy")
        if self.recurrence_mode not in {"standard", "flt_memory"}:
            raise ValueError("Unknown recurrence mode")


@dataclass
class LoopOutput:
    anchor: torch.Tensor
    workspace: torch.Tensor
    gate_logits: torch.Tensor
    writes: torch.Tensor
    candidates: torch.Tensor
    anchors: torch.Tensor
    question_embeds: torch.Tensor
    question_mask: torch.Tensor


@dataclass
class GenerationOutput:
    token_ids: torch.Tensor
    loop: LoopOutput


class AnchoredLoop(nn.Module):
    def __init__(self, backbone: BackboneAdapter, config: AnchorConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config
        if config.recurrence_mode == "flt_memory" and backbone.kind != "gpt2":
            raise ValueError("FLT-inspired memory injection currently supports GPT-2 only")
        width = backbone.hidden_size
        self.initial_anchor = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.initial_workspace = nn.Parameter(torch.randn(1, config.workspace_size, width) * 0.02)
        self.workspace_norm = nn.LayerNorm(width)
        self.gate_head = nn.Linear(width, 1)
        self.anchor_projection = nn.Linear(width, width)

    def forward(self, question_input_ids, attention_mask, *, loop_budget=None):
        """Accept questions and a configured compute budget only. No gold inputs."""
        if question_input_ids.ndim != 2 or attention_mask.shape != question_input_ids.shape:
            raise ValueError("Question IDs and masks must have matching [batch, tokens] shape")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("Attention mask must be binary")
        if torch.any(attention_mask.sum(dim=-1) == 0):
            raise ValueError("Each question must contain at least one token")
        if self.config.marker_token_id is not None and torch.any(
            (question_input_ids == self.config.marker_token_id) & attention_mask.bool()
        ):
            raise ValueError("Training markers cannot enter question-only inference")
        budget = self.config.loop_budget if loop_budget is None else loop_budget
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            raise ValueError("Loop budget must be a positive integer")
        question = self.backbone.embeddings(question_input_ids)
        batch, question_width, _ = question.shape
        anchor = self.initial_anchor.expand(batch, -1, -1)
        workspace = self.initial_workspace.expand(batch, -1, -1)
        base_embeds = torch.cat((question, anchor, workspace), dim=1)
        memory_mask = attention_mask.new_ones(batch, 1 + self.config.workspace_size)
        mask = torch.cat((attention_mask, memory_mask), dim=1)
        positions = position_ids(mask)
        logits, decisions, candidates, anchors = [], [], [], [anchor]
        for step in range(1, budget + 1):
            inputs = torch.cat((question, anchor, workspace), dim=1)
            if self.config.recurrence_mode == "flt_memory" and step > 1:
                hidden = self.backbone.forward_memory_embeds(inputs, mask, positions, base_embeds, question_width)
            else:
                hidden = self.backbone.forward_embeds(inputs, mask, positions)
            workspace = self.workspace_norm(hidden[:, question_width + 1:])
            pooled = workspace.mean(dim=1)
            gate_logit = self.gate_head(pooled).squeeze(-1)
            probability = torch.sigmoid(gate_logit)
            candidate = self.anchor_projection(pooled)
            if self.config.write_mode == "learned":
                hard = (probability >= self.config.gate_threshold).to(probability.dtype)
                write = hard + (probability - probability.detach()) if self.training else hard
            elif self.config.write_mode == "static":
                write = torch.zeros_like(probability)
            elif self.config.write_mode == "every_loop":
                write = torch.ones_like(probability)
            else:
                write = torch.full_like(probability, float(step % self.config.periodic_interval == 0))
            gate = write[:, None, None]
            anchor = (1 - gate) * anchor + gate * candidate[:, None, :]
            logits.append(gate_logit)
            decisions.append(write)
            candidates.append(candidate)
            anchors.append(anchor)
        return LoopOutput(
            anchor, workspace, torch.stack(logits, 1), torch.stack(decisions, 1),
            torch.stack(candidates, 1), torch.stack(anchors, 1), question, attention_mask,
        )

    def answer_logits(self, loop, suffix_ids, suffix_mask, *, all_layers=False):
        """Rebuild from latest memory and a teacher-forced answer suffix.

        ``all_layers`` also returns every hidden state of this readout pass.
        """
        if suffix_ids.shape != suffix_mask.shape or suffix_ids.ndim != 2:
            raise ValueError("Suffix IDs/mask must have matching batch shape")
        memory_mask = loop.question_mask.new_ones(suffix_ids.shape[0], 1 + self.config.workspace_size)
        mask = torch.cat((loop.question_mask, memory_mask, suffix_mask), 1)
        embeds = torch.cat((loop.question_embeds, loop.anchor, loop.workspace,
                            self.backbone.embeddings(suffix_ids)), 1)
        hidden = self.backbone.forward_embeds(embeds, mask, position_ids(mask), all_layers=all_layers)
        last = hidden[-1] if all_layers else hidden
        # Only the suffix is vocabulary-projected, and no recurrent cache survives.
        logits = self.backbone.logits(last[:, -suffix_ids.shape[1]:])
        return (logits, hidden) if all_layers else logits

    @torch.no_grad()
    def generate(self, question_input_ids, attention_mask, *, answer_prefix_ids,
                 max_new_tokens=64, loop_budget=None):
        if max_new_tokens < 1 or not answer_prefix_ids:
            raise ValueError("Answer prefix and positive token cap required")
        was_training = self.training
        self.eval()
        try:
            loop = self(question_input_ids, attention_mask, loop_budget=loop_budget)
            suffix = question_input_ids.new_tensor(answer_prefix_ids)[None].expand(question_input_ids.shape[0], -1)
            suffix_mask = torch.ones_like(suffix)
            finished = torch.zeros(suffix.shape[0], dtype=torch.bool, device=suffix.device)
            generated = []
            for _ in range(max_new_tokens):
                logits = self.answer_logits(loop, suffix, suffix_mask)
                token = logits[:, -1].argmax(-1)
                token = torch.where(finished, self.config.pad_token_id, token)
                generated.append(token)
                suffix = torch.cat((suffix, token[:, None]), 1)
                suffix_mask = torch.cat((suffix_mask, (~finished).long()[:, None]), 1)
                finished = finished | (token == self.config.eos_token_id)
                if bool(finished.all()):
                    break
            return GenerationOutput(torch.stack(generated, 1), loop)
        finally:
            self.train(was_training)

    def save_pretrained(self, directory, tokenizer):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        backbone = self.backbone.model
        # LoRA runs save merged weights, so the backbone reloads as plain GPT-2.
        merged = {"state_dict": merged_state_dict(backbone)} if has_lora(backbone) else {}
        backbone.save_pretrained(directory / "backbone", safe_serialization=True, **merged)
        tokenizer.save_pretrained(directory / "tokenizer")
        (directory / "anchor_config.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n")
        torch.save({key: value for key, value in self.state_dict().items()
                    if not key.startswith("backbone.")}, directory / "anchor.pt")

    @classmethod
    def from_pretrained(cls, directory):
        from transformers import AutoModelForCausalLM
        directory = Path(directory)
        backbone = BackboneAdapter(AutoModelForCausalLM.from_pretrained(directory / "backbone", attn_implementation="eager"))
        model = cls(backbone, AnchorConfig(**json.loads((directory / "anchor_config.json").read_text())))
        memory = torch.load(directory / "anchor.pt", map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(memory, strict=False)
        if unexpected or any(not key.startswith("backbone.") for key in missing):
            raise ValueError("Incomplete or incompatible anchor checkpoint")
        return model
