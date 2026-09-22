"""Data loaders. The ImageNet WebDataset loader needs the optional
``webdataset`` dependency; it is imported lazily so that lightweight loaders
(e.g. :mod:`eqfm.data.mnist`) work without it.

``build_imagenet_loader`` dispatches on the shard spec: a ``parquet:`` prefix
selects the local parquet reader (:mod:`eqfm.data.imagenet_parquet`), anything
else goes to the WebDataset/GCS reader."""
from .imagenet_latents import LATENTS_PREFIX, build_imagenet_latent_loader
from .imagenet_parquet import PARQUET_PREFIX, build_imagenet_parquet_loader

try:
    from .imagenet_wds import build_imagenet_loader as _build_wds_loader, DEFAULT_TRAIN_SHARDS
except ImportError:  # webdataset not installed
    _build_wds_loader = None
    DEFAULT_TRAIN_SHARDS = None


def is_latent_spec(shards) -> bool:
    """True when ``shards`` names a pre-encoded latent cache (``latents:`` prefix)."""
    return isinstance(shards, str) and shards.startswith(LATENTS_PREFIX)


def build_imagenet_loader(shards=DEFAULT_TRAIN_SHARDS, *, shardshuffle=100, **kw):
    if is_latent_spec(shards):
        kw.pop("image_size", None)  # pixels are gone; latents are 32x32x4
        return build_imagenet_latent_loader(shards, **kw)
    if isinstance(shards, str) and shards.startswith(PARQUET_PREFIX):
        return build_imagenet_parquet_loader(shards, **kw)
    if _build_wds_loader is None:
        raise ImportError("webdataset is required for non-parquet ImageNet shards")
    return _build_wds_loader(shards, shardshuffle=shardshuffle, **kw)
