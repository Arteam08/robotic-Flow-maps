"""Custom WebDataset URL opener for ``gs://`` URLs via the Python SDK.

Replaces ``pipe:gcloud storage cat gs://...`` which costs ~1-3s of
subprocess startup per shard. ``Blob.open("rb")`` streams the object
through a persistent HTTPS connection from the SDK's session pool, so
auth is paid once and shards just keep arriving.

Authentication uses ``google.auth.default()``, which honours
``GOOGLE_APPLICATION_CREDENTIALS`` and accepts **either** a service
account JSON **or** a user OAuth refresh-token JSON (the format
``gcloud auth application-default login`` writes). That's the whole
reason we made this switch -- the bucket owner does not need to mint a
service account for us; the user's own ADC creds work as-is.

The registration is idempotent and happens at import time of
``eqfm.data`` (via ``eqfm.data.imagenet_wds``), so any caller that
already uses our loader picks up ``gs://`` URL support automatically.
"""
from __future__ import annotations

from typing import BinaryIO


def gopen_gs(url: str, mode: str = "rb", bufsize: int = 8192, **kwargs) -> BinaryIO:
    """Open a ``gs://bucket/path/to/blob`` URL as a streaming binary file.

    Args:
        url: ``gs://`` URL to a single GCS object.
        mode: Only ``"rb"`` is supported (WebDataset's path).
        bufsize: Pass-through to ``Blob.open`` as ``chunk_size``. Larger
            buffers cut HTTPS round-trips at the cost of memory; the
            SDK default works well for tar shards.

    Returns:
        A ``google.cloud.storage.fileio.BlobReader`` (file-like, ``.read``,
        ``.close``, context-manager) ready for ``tarfile.open(fileobj=...)``.
    """
    if mode != "rb":
        raise ValueError(f"gs:// opener only supports rb mode; got {mode!r}.")
    if not url.startswith("gs://"):
        raise ValueError(f"Not a gs:// URL: {url!r}.")

    from google.cloud import storage

    bucket_name, _, blob_name = url[len("gs://"):].partition("/")
    if not bucket_name or not blob_name:
        raise ValueError(
            f"Malformed gs:// URL (need bucket + blob): {url!r}."
        )

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    return blob.open("rb", chunk_size=bufsize)


def register() -> None:
    """Register the ``gs`` scheme with WebDataset's gopen registry.

    Idempotent: safe to call multiple times. ``webdataset/__init__.py``
    re-exports both the ``gopen`` function and the ``gopen_schemes`` dict
    at the package top level (``from .gopen import gopen, gopen_schemes``).
    Because of that shadowing, ``wds.gopen`` is the *function*, not the
    submodule -- so we use ``wds.gopen_schemes`` directly.
    """
    import webdataset as wds

    wds.gopen_schemes["gs"] = gopen_gs
