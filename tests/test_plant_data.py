"""Verify split isolation and expanded browser-model coverage."""
import json
from collections import defaultdict

from conftest import ROOT
from prepare_plants import split_groups


def test_related_views_and_duplicates_stay_in_one_partition():
    rows = [{"id": sid, "source": source, "group": f"leaf-{i}", "view": view}
            for sid in ("acer_rubrum", "rose") for source in ("lab", "field")
            for i in range(20) for view in range(4)]
    result = split_groups(rows)
    partitions = defaultdict(set)
    for row in result:
        partitions[(row["id"], row["source"], row["group"])].add(row["split"])
    assert all(len(splits) == 1 for splits in partitions.values())
    for sid in ("acer_rubrum", "rose"):
        for source in ("lab", "field"):
            assert {r["split"] for r in result if r["id"] == sid and r["source"] == source} == {
                "train", "validation", "test"}
    assert [r["split"] for r in split_groups([dict(r) for r in rows])] == [r["split"] for r in result]


def test_expanded_model_and_photo_attribution():
    meta = json.loads((ROOT / "site/model/metadata.json").read_text())
    assert len(meta["labels"]) == 35
    assert {"daisy", "dandelion", "rose", "sunflower", "tulip", "acer_rubrum", "ilex_opaca"} <= set(meta["labels"])
    assert meta["architecture"] == "MobileNetV2"
    assert set(meta["test_by_source"]) == {"tf_flowers", "leafsnap_lab", "leafsnap_field"}
    assert sum(v["images"] for v in meta["test_by_source"].values()) == meta["test_images"]
    credits = json.loads((ROOT / "site/samples/credits.json").read_text())
    samples = {p.name for p in (ROOT / "site/samples").glob("*.jpg")}
    assert set(credits) == samples
    assert all(c["attribution"] and c["original"] and c["license"] for c in credits.values())
    assert sum(p.stat().st_size for p in (ROOT / "site/model").glob("*")) < 5_000_000
