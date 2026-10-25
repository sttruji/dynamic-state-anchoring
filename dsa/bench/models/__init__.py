"""Model registry for the evaluation harness.

Add a model by writing a BenchModel subclass anywhere and passing
``--model package.module:ClassName``; register it here only once it is a
team baseline. The harness itself never needs to change.
"""
from __future__ import annotations

import importlib

from dsa.bench.models.base import BenchModel

REGISTRY = {
    "gpt2-vanilla": "dsa.bench.models.gpt2_vanilla:VanillaGPT2",
    "gpt2-sft": "dsa.bench.models.gpt2_sft:SFTGPT2",
    "codi": "dsa.bench.models.codi:CODIGPT2",
    "lotus": "dsa.bench.models.lotus:LOTUSGPT2",
    "dsa": "dsa.bench.models.dsa_loop:DSAModel",
}


def build_model(spec: str, **kwargs) -> BenchModel:
    target = REGISTRY.get(spec, spec)
    if ":" not in target:
        raise ValueError(f"unknown model {spec!r}; use one of {', '.join(REGISTRY)} or module:Class")
    module, attribute = target.split(":", 1)
    cls = getattr(importlib.import_module(module), attribute)
    if not (isinstance(cls, type) and issubclass(cls, BenchModel)):
        raise TypeError(f"{target} must be a subclass of dsa.bench.models.BenchModel")
    return cls(**kwargs)


__all__ = ["BenchModel", "REGISTRY", "build_model"]
