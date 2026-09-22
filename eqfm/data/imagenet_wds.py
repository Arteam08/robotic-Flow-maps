"""ImageNet streamed from WebDataset tar shards on GCS.

The training data lives at::

    gs://cmu-gpucloud-jerryhua/imagenet/wds/train/train-{000000..001280}.tar

We support two URL forms:

  * ``pipe:gcloud storage cat gs://...`` (the current default).
    The gcloud CLI handles auth and reads. Works on hosts where you have
    ``gcloud auth login`` configured *or* ADC at the standard path.
    Doesn't need a GCP project as the quota target, which matters when
    the bucket is owned by an org you have no project-create rights in.

  * ``gs://...`` (faster, future default).
    Streams via ``google-cloud-storage``'s ``Blob.open("rb")``. No
    subprocess startup per shard, reuses a single HTTPS session across
    shards. Requires either a service-account JSON *or* a user-OAuth
    ADC JSON with ``quota_project_id`` set -- which is the auth
    situation we're still resolving with the bucket owner.

The ``gs://`` URL handler is registered unconditionally at import time;
switching back to it is a one-line change to ``DEFAULT_TRAIN_SHARDS``.

The loader yields ``(image, class_label)`` pairs where the image is a
3xHxW float tensor normalized to ``[-1, 1]`` -- the input range expected
by the Stable Diffusion VAE that we will use to encode latents before
feeding them to SiT.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torchvision import transforms

try:
    import webdataset as wds
except ImportError as e:  # pragma: no cover - environment misconfiguration
    raise ImportError(
        "webdataset is required for the ImageNet loader. "
        "Install it via `pip install -r requirements.txt`."
    ) from e

from .imagenet_synsets import synset_to_index
from .gcs_opener import register as _register_gcs_opener

# Idempotent: makes gs:// URLs work for any caller that imports this module.
_register_gcs_opener()


DEFAULT_TRAIN_SHARDS = (
    "pipe:gcloud storage cat "
    "gs://cmu-gpucloud-jerryhua/imagenet/wds/train/train-{000000..001280}.tar"
)


def _build_train_transform(image_size: int, hflip: bool = False) -> Callable:
    """Standard SD-VAE preprocessing: resize -> center crop -> [-1, 1].

    ``hflip`` adds the random horizontal flip used by the DiT/SiT/EqM
    training recipes (off by default to keep earlier runs reproducible).
    """
    return transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        *([transforms.RandomHorizontalFlip()] if hflip else []),
        transforms.ToTensor(),                                    # [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),                # [-1, 1]
    ])


def _decode_cls_as_string(key: str, data: bytes):
    """Override webdataset's default ``.cls`` decoder.

    The bucket stores ImageNet synset IDs (``n01440764``) in ``.cls`` files.
    The default decoder calls ``int(bytes)`` and raises ``DecodingError``;
    under ``warn_and_continue`` that silently drops every sample. Returning
    the value as a stripped string lets us map synset → class index in the
    downstream ``map_tuple``.

    Note: ``webdataset.autodecode.Decoder.decode1`` prepends ``"."`` to the
    extension before dispatching to handlers, so the key we match against is
    ``".cls"`` (with the leading dot), not ``"cls"``.
    """
    if key == ".cls":
        if isinstance(data, (bytes, bytearray)):
            return data.decode("utf-8").strip()
        return str(data).strip()
    return None


def _parse_label(raw) -> int:
    """Map a ``.cls`` payload to an integer class index in ``[0, 999]``.

    Accepts ``int`` (e.g. webdataset's default int decode, if it ever fired),
    ``bytes``/``str`` containing an integer (``"42"``), or an ImageNet synset
    string (``"n01440764"``). Synsets are looked up against
    :func:`imagenet_synsets.synset_to_index`, which is built from timm and
    matches the canonical sorted-WNID class ordering used by torchvision and
    the SiT-XL/2 checkpoint.
    """
    if isinstance(raw, int):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    s = str(raw).strip()
    if len(s) >= 9 and s[0] == "n" and s[1:9].isdigit():
        return synset_to_index()[s]
    return int(s)


def build_imagenet_loader(
    shards: str = DEFAULT_TRAIN_SHARDS,
    *,
    image_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_buffer: int = 1000,
    shardshuffle: bool | int = 100,
    seed: int = 0,
    distributed: bool = False,
    persistent_workers: Optional[bool] = None,
    hflip: bool = False,
) -> Iterable:
    """Build a WebDataset-backed ImageNet training loader.

    Args:
        shards: Brace-expandable shard pattern. The ``pipe:`` prefix is
            supported and recommended for GCS reads.
        image_size: Target side length after resize+center-crop.
        batch_size: Per-worker batch size.
        num_workers: PyTorch DataLoader workers; 0 keeps everything on the
            main process (useful for debugging).
        shuffle_buffer: In-stream shuffle buffer. Larger values mix across
            shards more thoroughly at the cost of RAM in each worker.
        shardshuffle: Pass-through to ``wds.WebDataset``. Set to ``False``
            for deterministic eval; an int sets the shuffle pool size.
            WebDataset 0.2.x emits a deprecation warning for boolean
            ``True``; pass an integer (default 100 shards of buffer) or
            ``False`` to silence it.
        seed: Seed for the per-epoch shard shuffle.
        distributed: If ``True``, split shards across DDP ranks via
            ``wds.split_by_node``. Always combine with split_by_worker
            (handled inside WebDataset by default).
        persistent_workers: Forwarded to ``WebLoader``. Defaults to
            ``num_workers > 0``.

    Returns:
        A ``wds.WebLoader`` yielding ``(images, labels)`` batches where
        ``images`` is float32 in ``[-1, 1]`` of shape ``(B, 3, H, H)`` and
        ``labels`` is int64 of shape ``(B,)``.
    """
    transform = _build_train_transform(image_size, hflip=hflip)
    nodesplitter = wds.split_by_node if distributed else wds.single_node_only

    dataset = (
        wds.WebDataset(
            shards,
            shardshuffle=shardshuffle,
            handler=wds.warn_and_continue,
            nodesplitter=nodesplitter,
            seed=seed,
            # When num_workers > num_shards (e.g. single-shard smoke runs)
            # split_by_worker leaves some workers with zero shards. The
            # default empty-check raises ValueError on those workers; we
            # disable it so partial-shard scenarios are non-fatal.
            empty_check=False,
        )
        .shuffle(shuffle_buffer)
        .decode(_decode_cls_as_string, "pil", handler=wds.warn_and_continue)
        .to_tuple(
            "jpg;jpeg;png;webp",
            "cls;txt",
            handler=wds.warn_and_continue,
        )
        .map_tuple(transform, _parse_label)
        .batched(batch_size, partial=False)
    )

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    loader = wds.WebLoader(
        dataset,
        batch_size=None,                # batching is done inside the pipeline
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    return loader
