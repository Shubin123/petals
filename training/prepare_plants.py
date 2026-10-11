"""Download/check LeafSnap30 and prepare a reproducible, source-stratified split.

Only indexed cropped lab images and original field images are used. Lab views
with the same specimen filename stem are kept together. Exact duplicate image
content is also grouped. 70% train, 15% validation, 15% test in each class/source.
"""
import csv
import hashlib
import io
import json
import pathlib
import random
import re
import tarfile
import urllib.request
import zipfile
from collections import defaultdict

from PIL import Image, ImageOps

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL = "https://zenodo.org/records/5061353/files/leafsnap-dataset-30subset.zip?download=1"
MD5 = "cf24bee185d2c2e42b8a18effc5ef554"
TAXON_CORRECTIONS = {"Aesculus pavi": "Aesculus pavia",
                     "Broussonettia papyrifera": "Broussonetia papyrifera"}
SOURCES = [
    {"name": "TensorFlow Flowers", "url": "https://www.tensorflow.org/datasets/catalog/tf_flowers",
     "license": "CC BY 2.0", "citation": "TensorFlow Flowers; original Flickr photographers (see LICENSE.txt)"},
    {"name": "LeafSnap30", "url": "https://doi.org/10.5281/zenodo.5061353", "version": "1.0",
     "license": "CC BY 4.0", "archive_md5": MD5,
     "citation": "Ranguelova, Meijer, Oostrum, Liu and Bos (2021), LeafSnap30. Derived from Kumar et al., Leafsnap: A Computer Vision System for Automatic Plant Species Identification, ECCV 2012."},
]


def split_groups(rows):
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        buckets[(row["id"], row["source"])][row["group"]].append(row)
    for key, groups in sorted(buckets.items()):
        keys = sorted(groups)
        random.Random(f"1337:{key}").shuffle(keys)
        n = len(keys)
        val_count = max(1, round(n * 0.15))
        test_count = max(1, round(n * 0.15))
        if n < 4:
            raise ValueError(f"Not enough independent groups for {key}")
        for i, group in enumerate(keys):
            split = "test" if i < test_count else "validation" if i < test_count + val_count else "train"
            for row in groups[group]:
                row["split"] = split
    return rows


def main():
    dest = ROOT / "data/plants"
    dest.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "data/leafsnap30.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != MD5:
        raise ValueError("LeafSnap30 archive checksum mismatch")
    rows = []
    with zipfile.ZipFile(archive) as z:
        for name in ("LICENSE.txt", "leafsnap-dataset-30subset-readme.txt"):
            (dest / ("leafsnap30-" + name)).write_bytes(z.read("leafsnap-dataset-30subset/" + name))
        indices = [n for n in z.namelist() if n.endswith(".txt") and "images" in n]
        if len(indices) != 1:
            raise ValueError(f"Expected one image index, found {indices}")
        index = indices[0]
        reader = csv.DictReader(io.StringIO(z.read(index).decode()), delimiter="\t")
        names = set(z.namelist())
        for row in reader:
            original_taxon = row["species"].strip()
            scientific = TAXON_CORRECTIONS.get(original_taxon, original_taxon)
            sid = re.sub(r"[^a-z0-9]+", "_", scientific.lower()).strip("_")
            path = row["image_path"].strip()
            if path not in names:
                path = str(pathlib.PurePosixPath(index).parent / path)
            raw = z.read(path)
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            image.thumbnail((320, 320))
            target = dest / sid / f"{row['file_id']}.jpg"
            target.parent.mkdir(exist_ok=True)
            image.save(target, quality=95)
            stem = pathlib.PurePosixPath(path).stem
            # Front/back lab views share the numeric specimen prefix.
            group = (re.sub(r"-\d+$", "", stem) if row["source"] == "lab"
                     else stem)
            rows.append({"id": sid, "scientific": scientific, "original_taxon": original_taxon,
                         "path": str(target.relative_to(ROOT)),
                         "source": f"leafsnap_{row['source']}", "original": row["image_path"].strip(),
                         "group": f"{sid}:{row['source']}:{group}", "sha256": hashlib.sha256(raw).hexdigest()})
    aliases = {"roses": "rose", "sunflowers": "sunflower", "tulips": "tulip"}
    flowers = ROOT / "data/flower_photos"
    if not flowers.exists():
        from train import download_tf_flowers
        download_tf_flowers(ROOT / "data")
    if not (flowers / "LICENSE.txt").exists():
        # Older Petals runs discarded this file. Restore attribution without
        # storing or extracting another copy of the image archive.
        url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        with urllib.request.urlopen(url) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                if member.name.endswith("LICENSE.txt"):
                    (flowers / "LICENSE.txt").write_bytes(tar.extractfile(member).read())
                    break
    for folder in sorted(p for p in flowers.iterdir() if p.is_dir()):
        for photo in sorted(folder.glob("*.jpg")):
            digest = hashlib.sha256(photo.read_bytes()).hexdigest()
            rows.append({"id": aliases.get(folder.name, folder.name), "path": str(photo.relative_to(ROOT)),
                         "source": "tf_flowers", "original": str(photo.relative_to(flowers)),
                         "group": digest, "sha256": digest})
    # Identical files with conflicting species labels are unusable supervision.
    digest_labels = defaultdict(set)
    for row in rows:
        digest_labels[row["sha256"]].add(row["id"])
    conflicts = {digest for digest, ids in digest_labels.items() if len(ids) > 1}
    excluded = [r for r in rows if r["sha256"] in conflicts]
    rows = [r for r in rows if r["sha256"] not in conflicts]
    # Merge duplicate bytes within each class/source before assigning partitions.
    seen = {}
    for row in rows:
        key = (row["id"], row["source"], row["sha256"])
        row["group"] = seen.setdefault(key, row["group"])
    labels = sorted({r["id"] for r in rows})
    for row in rows:
        row["label"] = labels.index(row["id"])
    split_groups(rows)
    by_digest = defaultdict(set)
    for row in rows:
        by_digest[row["sha256"]].add(row["split"])
    if any(len(splits) > 1 for splits in by_digest.values()):
        raise ValueError("Duplicate content crosses data partitions")
    manifest = {"labels": labels, "sources": SOURCES,
                "split_method": "Seed 1337; 70/15/15 grouped by class and source; lab specimen views and exact duplicates kept together. No tree/individual IDs are available for field images.",
                "excluded_conflicting_labels": len(excluded), "images": rows}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(rows)} images, {len(labels)} classes", flush=True)
    print({s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")})


if __name__ == "__main__":
    main()
