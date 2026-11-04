"""CODI-style LLaMA-3.2 checkpoints (CODI 1B, SIM-CoT 1B and 3B), loaded without peft.

Same inference as CODIGPT2, i.e. CODI's test.py with --remove_eos True --greedy True (commit 2c2314662c63): question
(with LLaMA's <|begin_of_text|>) + <bot>, six projected latent steps fed back through the KV cache, then <eot> and greedy
decoding with only <eot> masked. The checkpoints hold the LLaMA weights in BF16 and LoRA adapters (r=128, alpha=32 on
q/k/v/o/gate/up/down, per CODI's and SIM-CoT's LLaMA scripts) in FP32; the adapters are merged at FP32 at load time.

Neither repo ships a usable config (SIM-CoT's config.json is empty), so the LLaMA-3.2 architecture is written out here and
checked against the tensor shapes. The tokenizer is SIM-CoT 1B's: the Llama-3.2 Instruct tokenizer with CODI's [PAD] at
128256, which avoids the gated meta-llama repo. Weights and tokenizer files are pinned by revision and sha256.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from dsa.bench.models.base import BenchModel
from dsa.devices import resolve_device

LORA_ALPHA = 32
_PREFIX = "codi.base_model.model."
TOKENIZER = {"repo": "internlm/SIM_COT-LLaMA3-CODI-1B", "revision": "6e16fe4215025a9c48bc90d214c57349f247b017",
             "files": {"tokenizer.json": "e8485b8900f2ef3a891091dbe5ef0edc56f6922c0a23c7dfbb4baf64854f7de4",
                       "tokenizer_config.json": None, "special_tokens_map.json": None}}
CHECKPOINTS = {
    "codi-llama1b": {"repo": "zen-E/CODI-llama3.2-1b-Instruct", "revision": "b2c88ba224b06b12b52ef39b87f794b98a6eb1c8",
                     "size": "1b", "published_gsm8k": 55.6,  # CODI paper
                     "files": {"pytorch_model.bin": "c6d4b4b096f03fafca18409a4a7535ac9539f558ed741b31878fecadb40ed23b"}},
    "simcot-llama1b": {"repo": "internlm/SIM_COT-LLaMA3-CODI-1B", "revision": "6e16fe4215025a9c48bc90d214c57349f247b017",
                       "size": "1b", "published_gsm8k": 56.1,  # SIM-CoT paper, Table 2
                       "files": {"model.safetensors": "fc7f2d9285e4134a0c5bfdbfc19985c322e6fa27024b5fdd41edff6bf8c183aa"}},
    "simcot-llama3b": {"repo": "internlm/SIM_COT-LLaMA3-CODI-3B", "revision": "a2db0e50d53bdbf5f42d508899ab62c7ced5dbe7",
                       "size": "3b", "published_gsm8k": 62.3,  # SIM-CoT paper, Table 3
                       "files": {"model-00001-of-00003.safetensors": "b3e8d46f97161fe98f31fd6fa4416840d72e3ea2eb9178d40c4bbf75cff9f370",
                                 "model-00002-of-00003.safetensors": "ad30eb004e9c618fe25d2422e0fb7e489fcf90f80cc1e2bf81cd8b245675aa4b",
                                 "model-00003-of-00003.safetensors": "4eae72f5f5c9fe27742d1b1991383e4c1a41a45dc9c1a43d75204440dffc68fd"}},
}
# meta-llama/Llama-3.2-{1B,3B}-Instruct, as published; the vocabulary is set from the checkpoint (CODI adds 3 tokens).
ARCHITECTURES = {"1b": {"hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 16, "num_attention_heads": 32,
                        "num_key_value_heads": 8, "head_dim": 64},
                 "3b": {"hidden_size": 3072, "intermediate_size": 8192, "num_hidden_layers": 28, "num_attention_heads": 24,
                        "num_key_value_heads": 8, "head_dim": 128}}
COMMON = {"max_position_embeddings": 131072, "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "hidden_act": "silu",
          "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                           "original_max_position_embeddings": 8192, "rope_type": "llama3"},
          "tie_word_embeddings": True, "attention_bias": False, "mlp_bias": False, "bos_token_id": 128000,
          "eos_token_id": 128009}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(spec: dict) -> dict[str, Path]:
    """Download a pinned repo's files into the Hugging Face cache and check their sha256 where one is pinned."""
    from huggingface_hub import hf_hub_download

    paths = {}
    for name, expected in spec["files"].items():
        path = Path(hf_hub_download(spec["repo"], name, revision=spec["revision"]))
        if expected is not None and sha256_file(path) != expected:
            raise ValueError(f"{spec['repo']}/{name} does not match its pinned sha256")
        paths[name] = path
    return paths


def read_state(paths: list[Path]) -> dict[str, torch.Tensor]:
    """The CODI model and projection tensors of a checkpoint; SIM-CoT's step decoder and CODI's dynamic_cls are dropped."""
    keep = lambda key: key.startswith((_PREFIX, "prj."))
    state = {}
    for path in paths:
        if path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(path), framework="pt") as handle:
                state.update({key: handle.get_tensor(key) for key in handle.keys() if keep(key)})
        else:
            loaded = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            state.update({key: value for key, value in loaded.items() if keep(key)})
    return state


def merge_llama_state(state: dict[str, torch.Tensor], lora_alpha: int = LORA_ALPHA) -> tuple[dict, dict]:
    """Split a CODI LLaMA checkpoint into merged FP32 LLaMA weights and projection weights."""
    backbone, projection = {}, {}
    for key, value in state.items():
        if key.startswith("prj."):
            projection[key.removeprefix("prj.")] = value.float()
        elif key.startswith(_PREFIX) and ".lora_" not in key:
            backbone[key.removeprefix(_PREFIX).replace(".base_layer.", ".")] = value.float()
    for key in state:
        if key.endswith(".lora_A.default.weight"):
            module = key.removeprefix(_PREFIX).removesuffix(".lora_A.default.weight")
            lora_a = state[key].float()
            lora_b = state[key.replace(".lora_A.", ".lora_B.")].float()
            # nn.Linear stores [out, in]; LoRA adds (alpha / r) * B @ A. Out of place, as in merge_codi_state.
            backbone[f"{module}.weight"] = backbone[f"{module}.weight"] + (lora_alpha / lora_a.shape[0]) * (lora_b @ lora_a)
    head = backbone.pop("lm_head.weight", None)
    if head is not None and not torch.equal(head, backbone["model.embed_tokens.weight"]):
        raise ValueError("the checkpoint's lm_head differs from its embeddings; LLaMA-3.2 ties them")
    return backbone, projection


def llama_config(size: str, backbone: dict):
    from transformers import LlamaConfig

    arch = ARCHITECTURES[size]
    config = LlamaConfig(**arch, **COMMON, vocab_size=backbone["model.embed_tokens.weight"].shape[0])
    config._attn_implementation = "eager"  # the latent bias rides on the attention mask; attention shares need eager
    width, heads, kv, head = arch["hidden_size"], arch["num_attention_heads"], arch["num_key_value_heads"], arch["head_dim"]
    expected = {"model.layers.0.self_attn.q_proj.weight": (heads * head, width),
                "model.layers.0.self_attn.k_proj.weight": (kv * head, width),
                "model.layers.0.mlp.gate_proj.weight": (arch["intermediate_size"], width)}
    for key, shape in expected.items():
        if tuple(backbone[key].shape) != shape:
            raise ValueError(f"{key} is {tuple(backbone[key].shape)}, not {shape}: wrong architecture for {size}")
    if f"model.layers.{arch['num_hidden_layers'] - 1}.input_layernorm.weight" not in backbone or \
            f"model.layers.{arch['num_hidden_layers']}.input_layernorm.weight" in backbone:
        raise ValueError(f"the checkpoint does not have {arch['num_hidden_layers']} layers")
    return config


def build_projection(width: int, state: dict) -> nn.Sequential:
    projection = nn.Sequential(nn.Dropout(0.0), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
    projection.add_module("ln", nn.LayerNorm(width))
    projection.load_state_dict(state)
    return projection


class CODILlama(BenchModel):
    family = "llama"

    def __init__(self, *, checkpoint: str = "codi-llama1b", latent_iterations: int = 6, max_new_tokens: int = 256,
                 device: str = "auto", name: str | None = None):
        from transformers import AutoTokenizer, LlamaForCausalLM
        from transformers.modeling_utils import no_init_weights

        spec = CHECKPOINTS[checkpoint]
        self.name, self.device = name or checkpoint, resolve_device(device)
        self.latent_iterations, self.max_new_tokens = int(latent_iterations), int(max_new_tokens)
        backbone, projection = merge_llama_state(read_state(list(fetch(spec).values())))
        config = llama_config(spec["size"], backbone)
        with no_init_weights():
            self.model = LlamaForCausalLM(config)
        missing, unexpected = self.model.load_state_dict(backbone, strict=False, assign=True)  # no second FP32 copy
        if unexpected or set(missing) - {"lm_head.weight"}:
            raise ValueError(f"checkpoint does not match LLaMA: missing={missing} unexpected={unexpected}")
        self.model.tie_weights()
        del backbone
        self.projection = build_projection(config.hidden_size, projection)
        self.model.to(self.device).eval()
        self.projection.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(fetch(TOKENIZER)["tokenizer.json"].parent))
        self.vocab_size = config.vocab_size
        original = self.vocab_size - 3  # CODI appends pad, <bot>, <eot> after the LLaMA vocabulary
        if self.tokenizer.pad_token_id != original:
            raise ValueError(f"tokenizer pad id {self.tokenizer.pad_token_id} is not CODI's {original}")
        self.bot_id, self.eot_id = original + 1, original + 2
        self._identity = {"weights": f"{spec['repo']}@{spec['revision']}", "lora_merged_fp32": True,
                          "latent_iterations": self.latent_iterations, "max_new_tokens": self.max_new_tokens,
                          "decoding": "greedy", "dtype": "float32"}

    def identity(self) -> dict:
        return dict(self._identity)

    def encode(self, question: str) -> list[int]:
        return self.tokenizer.encode(question.strip().replace("  ", " ")) + [self.bot_id]

    @torch.no_grad()
    def _answer(self, question: str) -> str:
        output = self.model(input_ids=torch.tensor([self.encode(question)], device=self.device), use_cache=True,
                            output_hidden_states=True)
        cache = output.past_key_values
        latent = self.projection(output.hidden_states[-1][:, -1:, :])
        for _ in range(self.latent_iterations):
            output = self.model(inputs_embeds=latent, past_key_values=cache, use_cache=True, output_hidden_states=True)
            cache = output.past_key_values
            latent = self.projection(output.hidden_states[-1][:, -1:, :])
        embed = self.model.get_input_embeddings()
        step = embed(torch.tensor([[self.eot_id]], device=self.device))
        tokens = []
        for _ in range(self.max_new_tokens):
            output = self.model(inputs_embeds=step, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            token = int(output.logits[0, -1, :self.vocab_size - 1].argmax())
            tokens.append(token)
            if token == self.tokenizer.eos_token_id:
                break
            step = embed(torch.tensor([[token]], device=self.device))
        return self.tokenizer.decode([t for t in tokens if t < len(self.tokenizer)], skip_special_tokens=True)

    def generate(self, questions: list[str]) -> list[str]:
        return [self._answer(question) for question in questions]
