"""Unit tests for ImageNet label parsing.

Runs offline -- no GCS, no timm-data downloads beyond the synset list shipped
with timm. Catches the failure mode that was bricking the loader: WebDataset's
default ``.cls`` decoder calls ``int(bytes)`` on synset strings and raises
``DecodingError``, which under ``warn_and_continue`` silently drops every
sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eqfm.data.imagenet_synsets import index_to_synset, synset_to_index  # noqa: E402
from eqfm.data.imagenet_wds import _decode_cls_as_string, _parse_label  # noqa: E402


def test_synset_table_size_and_ends():
    s2i = synset_to_index()
    assert len(s2i) == 1000
    # Canonical sorted-WNID ordering used by torchvision and SiT-XL/2.
    assert s2i["n01440764"] == 0           # tench
    assert s2i["n15075141"] == 999         # toilet tissue


def test_synset_table_round_trip():
    s2i = synset_to_index()
    i2s = index_to_synset()
    for i in (0, 1, 42, 500, 999):
        assert s2i[i2s[i]] == i


def test_parse_label_accepts_synset_string():
    # str input, the path the live loader exercises.
    assert _parse_label("n01440764") == 0
    assert _parse_label("n15075141") == 999


def test_parse_label_accepts_synset_bytes():
    # WebDataset hands us bytes if the decoder isn't applied first.
    assert _parse_label(b"n01440764") == 0
    assert _parse_label(b"n01440764\n") == 0  # tolerate trailing whitespace


def test_parse_label_accepts_int_and_int_string():
    # Belt-and-suspenders: also handle the case where the bucket layout ever
    # switches back to integer class indices.
    assert _parse_label(0) == 0
    assert _parse_label("0") == 0
    assert _parse_label(b"42") == 42


def test_decode_cls_handler_matches_dotted_key():
    # webdataset's Decoder prepends a "." to the extension before calling
    # handlers; the previous "cls" string match silently failed and let the
    # default int(bytes) handler raise on synset payloads.
    assert _decode_cls_as_string(".cls", b"n01440764") == "n01440764"
    assert _decode_cls_as_string(".cls", b"n01440764\n") == "n01440764"
    # Non-cls keys must opt out so other handlers (e.g. PIL on .jpg) can run.
    assert _decode_cls_as_string(".jpg", b"\xff\xd8\xff") is None
    assert _decode_cls_as_string(".txt", b"hello") is None
    # Also guard against the off-by-dot regression: bare "cls" must NOT match
    # the handler -- if it did, we'd see a false-positive test pass while the
    # live loader silently failed.
    assert _decode_cls_as_string("cls", b"n01440764") is None


if __name__ == "__main__":
    # Allow plain `python tests/test_imagenet_labels.py` without pytest.
    import traceback
    failures = 0
    for name in list(globals()):
        if name.startswith("test_"):
            try:
                globals()[name]()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
