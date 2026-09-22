# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Functions for downloading pre-trained SiT models.

Adds a configurable cache directory on top of the upstream version so the
checkpoint can be written to a disk other than $HOME. Resolution order:

    1. The ``cache_dir`` argument, if provided.
    2. The ``EQFM_CACHE_DIR`` environment variable, if set.
    3. ``./pretrained_models`` (the upstream default).

The ``model_name`` argument may still be a local path to a ``.pt`` file, in
which case ``cache_dir`` is ignored.
"""
import os
from pathlib import Path

import torch
from torchvision.datasets.utils import download_url


pretrained_models = {'SiT-XL-2-256x256.pt'}


_PRETRAINED_URLS = {
    'SiT-XL-2-256x256.pt': (
        'https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/'
        'SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0'
    ),
}


def _resolve_cache_dir(cache_dir):
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    env = os.environ.get('EQFM_CACHE_DIR')
    if env:
        return Path(env).expanduser()
    return Path('pretrained_models')


def select_weights(checkpoint, key="auto"):
    """Pick the parameter dict out of a loaded checkpoint.

    ``checkpoint`` is either a flat state_dict or a trainer dict with
    ``"model"`` and/or ``"ema"`` entries. ``key`` is ``"model"``, ``"ema"`` or
    ``"auto"``. ``auto`` prefers ``"ema"`` only when it covers at least as many
    tensors as ``"model"``: a run that froze most parameters (e.g. head-only
    fine-tuning) stores an EMA shadow of the trainable subset only, and using
    it as the full model would leave the frozen trunk randomly initialised.
    The EMA's ``"decay"`` scalar is dropped.
    """
    if not isinstance(checkpoint, dict) or not ({"model", "ema"} & set(checkpoint)):
        return checkpoint
    model = checkpoint.get("model")
    ema = checkpoint.get("ema")
    if ema is not None:
        ema = {k: v for k, v in ema.items() if k != "decay"}
    if key == "model":
        chosen = model
    elif key == "ema":
        chosen = ema
    elif key == "auto":
        if ema is not None and (model is None or len(ema) >= len(model)):
            chosen = ema
        else:
            chosen = model
    else:
        raise ValueError(f"unknown checkpoint key {key!r}; use model|ema|auto")
    if chosen is None:
        raise KeyError(f"checkpoint has no {key!r} weights (keys: {sorted(checkpoint)[:8]})")
    return chosen


def find_model(model_name, cache_dir=None, key="auto"):
    """Find a pre-trained SiT model, downloading it if necessary.

    If ``model_name`` is one of the known pretrained tags (e.g.
    ``"SiT-XL-2-256x256.pt"``), the checkpoint is fetched into ``cache_dir``
    (see resolution order in the module docstring). Otherwise ``model_name``
    is treated as a path to an existing ``.pt`` file. ``key`` selects the
    weights inside a trainer checkpoint (see :func:`select_weights`).
    """
    if model_name in pretrained_models:
        return download_model(model_name, cache_dir=cache_dir)

    assert os.path.isfile(model_name), f'Could not find SiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage,
                            weights_only=False)
    return select_weights(checkpoint, key=key)


def download_model(model_name, cache_dir=None):
    """Download a pre-trained SiT model into ``cache_dir`` if absent, then load it."""
    assert model_name in pretrained_models
    cache = _resolve_cache_dir(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    local_path = cache / model_name
    if not local_path.is_file():
        download_url(_PRETRAINED_URLS[model_name], str(cache), filename=model_name)
    return torch.load(str(local_path), map_location=lambda storage, loc: storage)
