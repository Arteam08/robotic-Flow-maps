"""Parameter EMA for generative-model training.

Standard decay-averaging EMA (e.g. DDPM / DiT / SiT convention): on every
optimizer step, push the shadow parameters toward the live ones at rate
``1 - decay``. We keep the shadow on the same device + dtype as the model
and apply it via a context manager when we want to sample with EMA weights.

Default decay ``0.9999`` follows SiT.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must lie in (0, 1); got {decay}.")
        self.decay = decay
        # Float32 shadow regardless of training dtype -- avoids EMA drift under bf16.
        self.shadow: Dict[str, torch.Tensor] = {
            name: p.detach().clone().to(dtype=torch.float32)
            for name, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        # shadow <- decay*shadow + (1-decay)*p for every trainable param, as a
        # handful of multi-tensor kernels instead of ~2 launches per tensor.
        # Non-fp32 params (none in practice) fall back to the per-tensor path
        # so the fp32 shadow is never silently downcast.
        shadows, params, slow = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dtype == torch.float32:
                shadows.append(self.shadow[name]); params.append(p.detach())
            else:
                slow.append((name, p))
        if shadows:
            torch._foreach_mul_(shadows, self.decay)
            torch._foreach_add_(shadows, params, alpha=1.0 - self.decay)
        for name, p in slow:
            self.shadow[name].mul_(self.decay).add_(
                p.detach().to(dtype=torch.float32), alpha=1.0 - self.decay
            )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"decay": torch.tensor(self.decay), **self.shadow}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """Restore the shadow from ``state``, tolerating schema drift.

        Missing-from-state keys keep whatever the shadow was initialised with
        (typically the values cloned from the live model at ``__init__``
        time), and unexpected-in-state keys are skipped. Both cases are
        warned about so a structural mismatch is noticed.

        The motivating case: training with ``--time-conditioning strict``
        freezes the SiT t_embedder, so the saved EMA has no t_embedder.*
        keys. At sample time we may rebuild EMA without re-freezing, which
        previously crashed on the missing keys. Keeping the shadow's
        already-zeroed values for t_embedder is fine -- the frozen weights
        in the loaded model.state_dict are also zero.
        """
        if "decay" in state:
            decay = state["decay"]
            self.decay = float(decay.item()) if torch.is_tensor(decay) else float(decay)

        missing: list[str] = []
        for name, shadow_p in self.shadow.items():
            tensor = state.get(name)
            if tensor is None:
                missing.append(name)
                continue
            shadow_p.copy_(tensor)

        unexpected = [k for k in state.keys() if k != "decay" and k not in self.shadow]

        if missing:
            preview = ", ".join(missing[:3]) + (" ..." if len(missing) > 3 else "")
            print(f"[ema] WARN {len(missing)} key(s) missing from EMA state; "
                  f"shadow keeps initial values: {preview}")
        if unexpected:
            preview = ", ".join(unexpected[:3]) + (" ..." if len(unexpected) > 3 else "")
            print(f"[ema] WARN {len(unexpected)} unexpected key(s) in EMA state "
                  f"(no matching shadow param): {preview}")

    @contextmanager
    def apply_to(self, model: nn.Module):
        """Temporarily swap model parameters with EMA shadow.

        Use this around sampling / eval so the live weights are unchanged
        on exit:

            with ema.apply_to(model):
                samples = sample_from(model)
        """
        backup: Dict[str, torch.Tensor] = {}
        try:
            for name, p in model.named_parameters():
                if name in self.shadow:
                    backup[name] = p.detach().clone()
                    p.data.copy_(self.shadow[name].to(dtype=p.dtype))
            yield
        finally:
            for name, p in model.named_parameters():
                if name in backup:
                    p.data.copy_(backup[name])
