"""Vendored SiT (Ma et al. 2024) model code.

Upstream: https://github.com/willisma/SiT
License: MIT (Meta Platforms, Inc.) — preserved in ./LICENSE.

We vendor `models.py` and `download.py` verbatim so we can instantiate
SiT-XL/2 and load the pretrained ImageNet 256 checkpoint without taking
the full SiT training stack as a dependency.
"""
from .models import SiT, SiT_models, SiT_XL_2
from .download import find_model, select_weights

__all__ = ["SiT", "SiT_models", "SiT_XL_2", "find_model", "select_weights"]
