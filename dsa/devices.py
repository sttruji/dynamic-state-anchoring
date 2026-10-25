"""Device selection and synchronized measurements for local and CUDA runs."""
from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = ("cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu")
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError("Device must be auto, cpu, cuda, or mps")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return torch.device(name)


def synchronize(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def mps_memory_snapshot(device: str | torch.device) -> dict:
    """Current allocations, not a peak or a measurement of free system RAM."""
    if torch.device(device).type != "mps":
        return {}
    return {
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "mps_recommended_max_bytes": torch.mps.recommended_max_memory(),
    }
