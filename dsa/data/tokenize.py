"""Marker registration, exact text tokenization, and training-only loop labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TinyTokenizer:
    """A bounded, download-free UTF-8 byte tokenizer for correctness tests.

    IDs are fixed across records and runs: PAD=0, EOS=1, marker=2, and byte
    values plus 3. Offsets are Python Unicode character spans; all bytes of a
    multi-byte character share that character's span. This keeps arbitrary
    Unicode reversible while limiting a tiny model's vocabulary to 259 rows.
    """

    pad_token_id = 0
    eos_token_id = 1
    marker_token_id = 2
    pad_token = "<PAD>"
    eos_token = "<EOS>"
    vocab_size = 259
    is_fast = True

    def __init__(self, marker_token: str = "<SA>") -> None:
        if not isinstance(marker_token, str) or not marker_token or marker_token.isspace():
            raise ValueError("marker_token must be a nonempty, non-whitespace string")
        if marker_token in {self.pad_token, self.eos_token}:
            raise ValueError("marker collides with a tokenizer control token")
        self.marker_token = marker_token
        self.additional_special_tokens = [marker_token]
        self.all_special_tokens = [self.pad_token, self.eos_token, marker_token]
        self.all_special_ids = [0, 1, 2]

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return {
            self.pad_token: 0, self.eos_token: 1, self.marker_token: 2,
            **{f"<byte:{byte:02x}>": byte + 3 for byte in range(256)},
        }

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.get_vocab().get(token)

    def __call__(self, text: str, *, return_offsets_mapping: bool = False,
                 add_special_tokens: bool = False, **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            raise TypeError(f"unsupported TinyTokenizer options: {sorted(kwargs)}")
        if not isinstance(text, str):
            raise TypeError("TinyTokenizer expects one string")
        ids, offsets = [], []
        index = 0
        while index < len(text):
            if text.startswith(self.marker_token, index):
                ids.append(self.marker_token_id)
                offsets.append((index, index + len(self.marker_token)))
                index += len(self.marker_token)
            else:
                encoded = text[index].encode("utf-8")
                ids.extend(byte + 3 for byte in encoded)
                offsets.extend([(index, index + 1)] * len(encoded))
                index += 1
        result = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(self, token_ids: Any, skip_special_tokens: bool = False, **kwargs: Any) -> str:
        if kwargs:
            raise TypeError(f"unsupported TinyTokenizer decode options: {sorted(kwargs)}")
        pieces, pending = [], bytearray()

        def flush() -> None:
            if pending:
                pieces.append(pending.decode("utf-8", errors="replace"))
                pending.clear()

        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in (0, 1, 2):
                flush()
                if not skip_special_tokens:
                    pieces.append({0: self.pad_token, 1: self.eos_token, 2: self.marker_token}[token_id])
            elif 3 <= token_id < self.vocab_size:
                pending.append(token_id - 3)
            else:
                raise ValueError(f"invalid TinyTokenizer token ID: {token_id}")
        flush()
        return "".join(pieces)

    def save_pretrained(self, directory: str | Path) -> tuple[str]:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        config_path = path / "tiny_tokenizer.json"
        config_path.write_text(json.dumps({"type": "dsa_tiny_utf8_v1", "marker_token": self.marker_token}), encoding="utf-8")
        return (str(config_path),)

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "TinyTokenizer":
        config = json.loads((Path(directory) / "tiny_tokenizer.json").read_text(encoding="utf-8"))
        if config.get("type") != "dsa_tiny_utf8_v1":
            raise ValueError("unsupported TinyTokenizer format")
        return cls(marker_token=config["marker_token"])


def _same_weights(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return left.weight is right.weight or left.weight.data_ptr() == right.weight.data_ptr()


def register_marker_token(tokenizer: Any, model: Any = None, marker_token: str = "<SA>") -> int:
    """Register a whitespace-preserving special token and preserve LM tying.

    A pre-existing ordinary vocabulary entry is a collision, not permission
    to repurpose a token. An existing additional special token is accepted
    only when its AddedToken metadata has safe normalization/strip settings.
    """
    if not isinstance(marker_token, str) or not marker_token or marker_token.isspace():
        raise ValueError("marker_token must be a nonempty, non-whitespace string")
    vocab = tokenizer.get_vocab()
    controls = set(getattr(tokenizer, "all_special_tokens", [])) - {marker_token}
    control_ids = {vocab[token] for token in controls if token in vocab}
    if marker_token in vocab:
        if marker_token not in getattr(tokenizer, "additional_special_tokens", []):
            raise ValueError(f"marker token {marker_token!r} collides with an existing vocabulary meaning")
        metadata = getattr(tokenizer, "added_tokens_decoder", {}).get(vocab[marker_token])
        if metadata is not None and any(getattr(metadata, key, False) for key in ("normalized", "lstrip", "rstrip")):
            raise ValueError("existing marker token has unsafe normalization or whitespace stripping")
    else:
        try:
            from tokenizers import AddedToken
        except ImportError as exc:
            raise ImportError("install tokenizers to register a Hugging Face marker token") from exc
        marker = AddedToken(marker_token, normalized=False, lstrip=False, rstrip=False, single_word=False, special=True)
        tokenizer.add_special_tokens({"additional_special_tokens": [marker]}, replace_additional_special_tokens=False)
    ids = tokenizer.encode(marker_token, add_special_tokens=False)
    if len(ids) != 1 or ids[0] in control_ids:
        raise ValueError("marker must encode as one distinct token, separate from existing controls")
    marker_id = int(ids[0])
    if tokenizer.get_vocab().get(marker_token) != marker_id:
        raise ValueError("marker encoding does not match vocabulary registration")
    if model is not None:
        input_layer = model.get_input_embeddings()
        output_layer = model.get_output_embeddings()
        was_tied = _same_weights(input_layer, output_layer)
        model.resize_token_embeddings(len(tokenizer))
        if was_tied:
            model.tie_weights()
            if not _same_weights(model.get_input_embeddings(), model.get_output_embeddings()):
                raise ValueError("embedding resize failed to preserve input/output weight tying")
        if not model.get_input_embeddings().weight.requires_grad:
            raise ValueError("marker embedding is frozen; keep the new row trainable")
    return marker_id


def _with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer with Unicode character offsets is required")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    if len(ids) != len(offsets):
        raise ValueError("token IDs and offsets have different lengths")
    if any(not 0 <= start <= end <= len(text) for start, end in offsets):
        raise ValueError("tokenizer returned invalid character offsets")
    return ids, offsets


def tokenize_record(annotated_record: dict[str, Any], tokenizer: Any, loop_budget: int) -> dict[str, Any]:
    """Create teacher-forced labels; these fields never enter inference APIs."""
    from dsa.data.milestones import map_events_to_loops

    record = annotated_record
    marker = record.get("policy", {}).get("marker_token", "<SA>")
    question, raw, answer = (record[key] for key in ("question", "rationale_raw", "final_answer"))
    if not all(isinstance(value, str) for value in (question, raw, answer)):
        raise ValueError("question, rationale_raw, and final_answer must be strings")
    for field, text in (("question", question), ("rationale_raw", raw), ("final_answer", answer)):
        if marker in text or "<SA>" in text:
            raise ValueError(f"literal annotation marker in original {field}")
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    if len(marker_ids) != 1 or tokenizer.get_vocab().get(marker) != marker_ids[0]:
        raise ValueError("register the marker as a distinct single token before tokenization")
    marker_id = int(marker_ids[0])
    if marker_id in {getattr(tokenizer, "pad_token_id", None), getattr(tokenizer, "eos_token_id", None)}:
        raise ValueError("marker collides with a padding or EOS control token")
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise ValueError("tokenizer must define an EOS token")
    events = record.get("events", [])
    char_ends = [event["end"] for event in events]
    if any(isinstance(end, bool) or not isinstance(end, int) or not 0 < end <= len(raw) for end in char_ends):
        raise ValueError("event ends must be 1-based character boundaries within the raw rationale")
    if any(left >= right for left, right in zip(char_ends, char_ends[1:])):
        raise ValueError("event character ends must be strictly increasing")
    if record.get("event_count", len(events)) != len(events):
        raise ValueError("event_count disagrees with accepted events")
    chunks, previous = [], 0
    for end in char_ends:
        chunks.extend((raw[previous:end], marker))
        previous = end
    chunks.append(raw[previous:])
    expected_marked = "".join(chunks)
    marked = record.get("rationale_marked")
    if marked != expected_marked:
        raise ValueError("marked rationale does not exactly preserve raw text and event boundaries")
    explicit_text = question + "\n" + marked + "\n### " + answer
    explicit_ids, explicit_offsets = _with_offsets(tokenizer, explicit_text)
    marker_positions = [index for index, token_id in enumerate(explicit_ids) if token_id == marker_id]
    if len(marker_positions) != len(events) or any(position == 0 for position in marker_positions):
        raise ValueError("marker token positions disagree with accepted events")
    checkpoint_positions = [position - 1 for position in marker_positions]
    labels = [(-100 if start < len(question) and end > 0 else token_id)
              for token_id, (start, end) in zip(explicit_ids, explicit_offsets)]
    explicit_ids.append(int(eos))
    labels.append(int(eos))
    raw_ids, raw_offsets = _with_offsets(tokenizer, raw)
    raw_ends = []
    for end in char_ends:
        covering = [index for index, (start, stop) in enumerate(raw_offsets) if start < end <= stop]
        if not covering:
            raise ValueError(f"event character end {end} has no raw-rationale token mapping")
        raw_ends.append(covering[-1] + 1)
    if any(left > right for left, right in zip(raw_ends, raw_ends[1:])):
        raise ValueError("raw event token ends must be sorted")
    if any(raw_ids[end - 1] == getattr(tokenizer, "pad_token_id", None) for end in raw_ends):
        raise ValueError("event boundary maps to padding")
    schedule = map_events_to_loops(raw_ends, len(raw_ids), loop_budget)
    answer_prefix = "\n### "
    # Latent decoding has a fixed inference prefix. Preserve that token boundary:
    # full suffix BPE could merge prefix and answer into the first suffix token,
    # causing shifted answer loss to discard the answer and differ at inference.
    # The explicit teacher-forced route above still tokenizes its text only once.
    answer_prefix_ids = tokenizer.encode(answer_prefix, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [int(eos)]
    suffix_ids = answer_prefix_ids + answer_ids
    suffix_labels = [-100] * len(answer_prefix_ids) + answer_ids
    return {
        "sample_id": record.get("sample_id"),
        "source_problem_id": record.get("source_problem_id"),
        "question_input_ids": tokenizer.encode(question + "\n", add_special_tokens=False),
        "answer_input_ids": answer_ids,
        "latent_answer_prefix_ids": answer_prefix_ids,
        "latent_suffix_input_ids": suffix_ids,
        "latent_suffix_labels": suffix_labels,
        "explicit_input_ids": explicit_ids,
        "explicit_labels": labels,
        "checkpoint_positions": checkpoint_positions,
        "raw_event_token_ends": raw_ends,
        "raw_rationale_token_length": len(raw_ids),
        "gate_targets": schedule["gate_targets"],
        "event_loops": schedule["event_loops"],
        "controller_supervised": schedule["supervised"],
        "controller_mask_reason": schedule.get("reason"),
        "marker_token_id": marker_id,
        "loop_budget": loop_budget,
    }


def answer_alignment(record: dict[str, Any], tokenizer: Any) -> list[int]:
    """Explicit and latent-suffix indices for CODI's answer-position distillation.

    CODI aligns the token just before the answer in both routes. The explicit
    text is tokenized once, so BPE can merge the prefix's trailing space into
    the answer (GPT-2: ``###`` then `` 18``), while the latent suffix keeps the
    whole prefix (``###``, `` ``, ``18``). Use the explicit token just before
    the answer and the last copy of that same token in the latent prefix.
    """
    explicit = record["explicit_input_ids"][:-1]  # without EOS
    prefix = record["latent_answer_prefix_ids"]
    options = {} if isinstance(tokenizer, TinyTokenizer) else {"clean_up_tokenization_spaces": False}
    text = tokenizer.decode(explicit, **options)
    answer = tokenizer.decode(record["answer_input_ids"][:-1], **options)
    ids, offsets = _with_offsets(tokenizer, text)
    if ids != explicit or not answer or not text.endswith(tokenizer.decode(prefix, **options) + answer):
        raise ValueError("explicit sequence does not end with the latent answer prefix and answer")
    start = len(text) - len(answer)
    first = next(index for index, (_, end) in enumerate(offsets) if end > start)
    shared = [index for index, token_id in enumerate(prefix) if first and token_id == explicit[first - 1]]
    if not shared:
        raise ValueError("no answer-prefix token is shared by the explicit and latent routes")
    return [first - 1, shared[-1]]


def collate_records(records: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    """Pad dense model fields and retain ragged boundary metadata as lists."""
    if not records:
        raise ValueError("cannot collate an empty batch")
    result = {key: [record[key] for record in records] for key in records[0]}
    for key in ("question_input_ids", "answer_input_ids", "latent_answer_prefix_ids", "latent_suffix_input_ids", "latent_suffix_labels", "explicit_input_ids", "explicit_labels"):
        width = max(len(record[key]) for record in records)
        fill = -100 if key.endswith("labels") else pad_token_id
        result[key] = [record[key] + [fill] * (width - len(record[key])) for record in records]
    result["question_attention_mask"] = [
        [1] * len(record["question_input_ids"]) + [0] * (len(result["question_input_ids"][0]) - len(record["question_input_ids"]))
        for record in records
    ]
    result["explicit_attention_mask"] = [
        [1] * len(record["explicit_input_ids"]) + [0] * (len(result["explicit_input_ids"][0]) - len(record["explicit_input_ids"]))
        for record in records
    ]
    return result
