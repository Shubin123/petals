"""Expand the prepared plant manifest with Oxford Flowers 102.

Streams the official archive, retaining resized images locally. The split is a
custom source-stratified 70/15/15 split, not Oxford's published benchmark split.
Oxford photos are used locally for training/evaluation and aren't bundled into
the public site: the original release doesn't specify a redistribution license.
"""
import hashlib
import html
import io
import json
import pathlib
import re
import struct
import tarfile
import urllib.request
import urllib.parse
import zlib
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps

from prepare_plants import ROOT, split_groups

BASE = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
ALIASES = {"common dandelion": "dandelion", "sunflower": "sunflower", "rose": "rose", "oxeye daisy": "daisy"}


def mat_labels(raw):
    """Read the official MATLAB v5 uint8 labels vector without a SciPy dependency."""
    if raw[126:128] != b"IM":
        raise ValueError("Expected little-endian MATLAB v5 labels")
    kind, size = struct.unpack_from("<II", raw, 128)
    if kind != 15:
        raise ValueError("Expected compressed MATLAB matrix")
    matrix = zlib.decompress(raw[136:136 + size])
    offset = 8  # miMATRIX header
    while offset + 8 <= len(matrix):
        kind, size = struct.unpack_from("<II", matrix, offset)
        if kind >> 16:
            offset += 8
            continue
        value = matrix[offset + 8:offset + 8 + size]
        if kind == 2 and size == 8189:
            labels = list(value)
            if min(labels) != 1 or max(labels) != 102:
                raise ValueError("Unexpected category range")
            return labels
        offset += 8 + ((size + 7) // 8) * 8
    raise ValueError("Missing Oxford labels vector")


def photo_hash(image):
    """A conservative exact perceptual fingerprint to catch resized copies."""
    small = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    basis = np.cos(np.pi / 32 * (np.arange(32)[None, :] + .5) * np.arange(8)[:, None])
    spectrum = basis @ small @ basis.T
    bits = spectrum > np.median(spectrum.flatten()[1:])
    return np.packbits(bits).tobytes().hex()


def main():
    dest = ROOT / "data/flowers102"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("imagelabels.mat", "categories.html", "README.txt"):
        path = dest / name
        if not path.exists():
            urllib.request.urlretrieve(BASE + name, path)
    labels = mat_labels((dest / "imagelabels.mat").read_bytes())
    page = (dest / "categories.html").read_text()
    categories = {}
    thumbnails = {}
    for number, name in re.findall(r'<img src="thumbs/thumbim_(\d+)\.jpg"[^>]*alt="([^"]+)"', page):
        cid = labels[int(number) - 1]
        categories[cid] = html.unescape(name).strip()
        thumbnails[cid] = BASE + f"thumbs/thumbim_{number}.jpg"
    if set(categories) != set(range(1, 103)):
        raise ValueError("Oxford category names do not cover the label vector")
    base = json.loads((ROOT / "data/plants/manifest.json").read_text())
    rows = [dict(r) for r in base["images"]]
    fingerprints = set()
    for row in rows:
        # Only flowers can overlap the web-sourced flower dataset.
        if row["source"] == "tf_flowers":
            fingerprints.add(photo_hash(Image.open(ROOT / row["path"])))
    cache = dest / "images.json"
    if cache.exists():
        new_rows = json.loads(cache.read_text())
    else:
        new_rows = []
        with urllib.request.urlopen(BASE + "102flowers.tgz") as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                match = re.fullmatch(r"jpg/image_(\d+)\.jpg", member.name)
                if not match:
                    continue
                number = int(match[1])
                cid = labels[number - 1]
                name = categories[cid]
                sid = ALIASES.get(name, re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"))
                raw = tar.extractfile(member).read()
                image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
                fingerprint = photo_hash(image)
                target = dest / pathlib.PurePosixPath(member.name).name
                image.thumbnail((180, 180))
                image.save(target, quality=88)
                new_rows.append({"id": sid, "path": str(target.relative_to(ROOT)), "source": "oxford102",
                                 "original": member.name, "original_category": name, "category_id": cid,
                                 "sha256": hashlib.sha256(raw).hexdigest(), "photo_hash": fingerprint,
                                 "group": fingerprint})
                if number % 1000 == 0:
                    print(f"Prepared Oxford image {number}/8189", flush=True)
        if len(new_rows) != 8189:
            raise ValueError(f"Expected 8189 Oxford images, got {len(new_rows)}")
        cache.write_text(json.dumps(new_rows))
    # Do not introduce copies of existing flower photos into another partition.
    old_digests = {r["sha256"] for r in rows}
    eligible = [r for r in new_rows if r["sha256"] not in old_digests and r["photo_hash"] not in fingerprints]
    by_digest = defaultdict(set)
    for row in eligible:
        by_digest[row["sha256"]].add(row["id"])
    by_fingerprint = defaultdict(set)
    for row in eligible:
        by_fingerprint[row["photo_hash"]].add(row["id"])
    eligible = [r for r in eligible if len(by_digest[r["sha256"]]) == 1
                and len(by_fingerprint[r["photo_hash"]]) == 1]
    split_groups(eligible)
    rows.extend(eligible)
    final_labels = sorted({r["id"] for r in rows})
    for row in rows:
        row["label"] = final_labels.index(row["id"])
    source = {"name": "Oxford Flowers 102", "url": BASE + "index.html",
              "license": "Not specified in original release; original image rights retained. Photos not redistributed by Petals.",
              "citation": "Maria-Elena Nilsback and Andrew Zisserman, Automated flower classification over a large number of classes, ICVGIP 2008.",
              "labels_sha256": hashlib.sha256((dest / "imagelabels.mat").read_bytes()).hexdigest(),
              "categories_sha256": hashlib.sha256((dest / "categories.html").read_bytes()).hexdigest()}
    manifest = {**base, "labels": final_labels, "images": rows, "sources": base["sources"] + [source],
                "excluded_oxford_duplicates": len(new_rows) - len(eligible),
                "split_method": base["split_method"] + " Oxford uses a custom 70/15/15 split grouped by perceptual fingerprint, not the official benchmark split; matching copies of existing flower photos are excluded."}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    guides = json.loads((ROOT / "app/species.json").read_text())
    for cid, name in sorted(categories.items()):
        sid = ALIASES.get(name, re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"))
        if sid in guides:
            continue
        common = name.rstrip("?").capitalize()
        guides[sid] = {"common": common, "scientific": common + " (flower category)",
                       "family": "Not specified by the source dataset", "also": "Oxford Flowers 102 category; may include cultivars and related plants.",
                       "description": f"{common} is a flower category in Oxford Flowers 102. This match identifies the dataset category, not a verified botanical species or cultivar.",
                       "look_for": ["Use a close-up of a well-lit flower head.", "Include the petals and flower centre, and compare with the source dataset example."],
                       "flowering": "Varies by species, cultivar and location.",
                       "wiki": "https://en.wikipedia.org/w/index.php?search=" + urllib.parse.quote(common),
                       "source": BASE + "index.html", "example_url": thumbnails[cid]}
    (dest / "species.json").write_text(json.dumps(dict(sorted(guides.items())), indent=2))
    print(f"Prepared {len(rows)} photos, {len(final_labels)} classes; excluded {len(new_rows)-len(eligible)} Oxford copies", flush=True)


if __name__ == "__main__":
    main()
