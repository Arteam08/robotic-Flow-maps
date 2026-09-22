"""ImageNet-1k synset ↔ class index mapping.

The training shards under ``gs://cmu-gpucloud-jerryhua/imagenet/wds/train``
store the class label as a WordNet synset ID (e.g. ``n01440764``) in the
``.cls`` file rather than as an integer. WebDataset's default ``.cls``
decoder calls ``int(bytes)`` and raises ``DecodingError`` on synset strings,
which silently kills every sample under ``warn_and_continue``.

We pull the canonical sorted-by-WNID synset list from
``timm.data.imagenet_info.ImageNetInfo("imagenet-1k")`` so we cannot
silently drift from the SiT-XL/2 checkpoint's expected ordering -- that
checkpoint, like virtually every torchvision-style ImageNet weight, uses
the same sorted-WNID ordering (``n01440764`` → 0, ``n15075141`` → 999).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=1)
def synset_to_index() -> Dict[str, int]:
    """Return ``{synset_id: class_index}`` for all 1000 ImageNet-1k classes."""
    try:
        from timm.data.imagenet_info import ImageNetInfo
    except ImportError as e:
        raise RuntimeError(
            "Could not import timm.data.imagenet_info. "
            "Install or upgrade timm (it is pinned in requirements.txt at >=0.9)."
        ) from e

    info = ImageNetInfo("imagenet-1k")
    synsets = info.label_names()  # canonical sorted-WNID order, length 1000
    if len(synsets) != 1000:
        raise RuntimeError(
            f"Expected 1000 ImageNet-1k synsets, got {len(synsets)}. "
            "Has the timm dataset info changed?"
        )
    return {s: i for i, s in enumerate(synsets)}


@lru_cache(maxsize=1)
def index_to_synset() -> Dict[int, str]:
    """Return ``{class_index: synset_id}`` (inverse of ``synset_to_index``)."""
    return {i: s for s, i in synset_to_index().items()}
