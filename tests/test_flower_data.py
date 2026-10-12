"""Check Oxford's category decoder and local split isolation when data is present."""
import json
import struct
import zlib
from collections import defaultdict

import pytest

from conftest import ROOT
from prepare_flowers import mat_labels


def test_oxford_uint8_matlab_labels_are_not_shifted():
    labels = bytes(i % 102 + 1 for i in range(8189))
    vector = struct.pack("<II", 2, len(labels)) + labels
    matrix = struct.pack("<II", 14, len(vector)) + vector
    compressed = zlib.compress(matrix)
    raw = bytes(126) + b"IM" + struct.pack("<II", 15, len(compressed)) + compressed
    assert mat_labels(raw) == list(labels)
    with pytest.raises(ValueError):
        mat_labels(raw[:126] + b"MI" + raw[128:])


def test_prepared_flower_manifest_has_no_cross_partition_copies():
    path = ROOT / "data/flowers102/manifest.json"
    if not path.exists():
        pytest.skip("Oxford training data is not stored in the repository")
    manifest = json.loads(path.read_text())
    originals, fingerprints = defaultdict(set), defaultdict(set)
    for row in manifest["images"]:
        originals[row["sha256"]].add(row["split"])
        if row["source"] == "oxford102":
            fingerprints[row["photo_hash"]].add(row["split"])
    assert all(len(splits) == 1 for splits in originals.values())
    assert all(len(splits) == 1 for splits in fingerprints.values())
    assert {"daffodil", "petunia", "hibiscus", "rose", "acer_rubrum"} <= set(manifest["labels"])
