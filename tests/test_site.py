"""Checks that the committed GitHub Pages build in site/ is complete and in step with app/static/."""

import json
import re

import pytest

from conftest import LABELS, ROOT

SITE = ROOT / "site"
STATIC = ROOT / "app" / "static"


def test_required_files():
    for name in ("index.html", "app.js", "styles.css", "favicon.svg", "species.json",
                 ".nojekyll", "model/model.json", "model/metadata.json"):
        assert (SITE / name).is_file(), f"site/{name} is missing; run training/build_pages.py"


@pytest.mark.parametrize("name", ["app.js", "styles.css", "favicon.svg"])
def test_front_end_matches_source(name):
    assert (SITE / name).read_text() == (STATIC / name).read_text(), \
        f"site/{name} is out of date; run training/build_pages.py"


def test_index_matches_source_in_static_mode():
    site = (SITE / "index.html").read_text()
    assert '<html lang="en" data-mode="static">' in site
    assert site.replace(' data-mode="static"', "", 1) == (STATIC / "index.html").read_text()


def test_index_uses_relative_urls():
    # GitHub Pages serves a project site from /<repo>/, so root-relative URLs would 404.
    html = (SITE / "index.html").read_text()
    for url in re.findall(r'(?:src|href)="([^"#]+)"', html):
        assert url.startswith("https://") or not url.startswith("/"), url


def test_model_weights_present():
    manifest = json.loads((SITE / "model" / "model.json").read_text())
    assert manifest["format"] == "graph-model"
    assert manifest["signature"]["inputs"]["image"]["tensorShape"]["dim"][1:] == \
        [{"size": "180"}, {"size": "180"}, {"size": "3"}]
    paths = [p for group in manifest["weightsManifest"] for p in group["paths"]]
    assert paths
    for p in paths:
        shard = SITE / "model" / p
        assert shard.is_file(), p
        assert shard.stat().st_size < 50_000_000  # GitHub warns above 50 MB per file
    on_disk = {p.name for p in (SITE / "model").glob("*.bin")}
    assert on_disk == set(paths), "stray or missing weight shards"


def test_weights_are_quantized():
    manifest = json.loads((SITE / "model" / "model.json").read_text())
    weights = [w for group in manifest["weightsManifest"] for w in group["weights"]]
    floats = [w for w in weights if w["dtype"] == "float32"]
    assert floats and all(w.get("quantization", {}).get("dtype") == "uint8" for w in floats)


def test_metadata_matches_species():
    meta = json.loads((SITE / "model" / "metadata.json").read_text())
    species = json.loads((SITE / "species.json").read_text())
    assert meta["labels"] == LABELS
    assert meta["image_size"] == [180, 180]
    assert sorted(species) == meta["labels"]


def test_species_json_matches_source():
    source = json.loads((ROOT / "app" / "species.json").read_text())
    site = json.loads((SITE / "species.json").read_text())
    for sid, info in site.items():
        assert info["id"] == sid
        assert {k: v for k, v in info.items() if k not in ("id", "samples")} == source[sid]
        assert info["samples"], f"no sample photos for {sid}"
        for s in info["samples"]:
            assert re.fullmatch(rf"samples/{sid}-\d+\.jpg", s)
            assert (SITE / s).is_file(), s


def test_metadata_matches_trained_model():
    trained = ROOT / "models" / "metadata.json"
    if not trained.exists():
        pytest.skip("no trained model in models/")
    assert json.loads((SITE / "model" / "metadata.json").read_text()) == json.loads(trained.read_text())
