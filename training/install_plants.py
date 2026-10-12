"""Promote a trained plant model and select credited validation samples.

Usage: python training/install_plants.py --model-dir models/plants
The previous model is preserved under models/previous/ on first promotion.
"""
import argparse
import io
import json
import pathlib
import shutil

import keras
import numpy as np
from PIL import Image, ImageOps

from build_pages import inference_model, quantize_like_tfjs
from train import ROOT, IMAGE_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=pathlib.Path, default=ROOT / "models/plants")
    parser.add_argument("--manifest", type=pathlib.Path, default=ROOT / "data/plants/manifest.json")
    parser.add_argument("--species", type=pathlib.Path, default=ROOT / "app/species.json")
    args = parser.parse_args()
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    if meta["labels"] != manifest["labels"]:
        raise ValueError("Model and data labels differ")
    if set(meta["labels"]) != set(json.loads(args.species.read_text())):
        raise ValueError("Species guide coverage differs from model")
    trained = keras.models.load_model(args.model_dir / "petals.keras")
    model = quantize_like_tfjs(inference_model(trained), "float16")
    scores = np.load(args.model_dir / "validation_scores.npy")
    rows = [r for r in manifest["images"] if r["split"] == "validation"]
    license_file = ROOT / "data/flower_photos/LICENSE.txt"
    if not license_file.exists():
        raise ValueError("Restore TF Flowers LICENSE.txt before distributing photographs")
    attributions = {line.split(" CC-BY by ")[0]: line for line in license_file.read_text().splitlines()
                    if " CC-BY by " in line}
    samples = args.model_dir / "samples"
    if samples.exists():
        shutil.rmtree(samples)
    samples.mkdir()
    credits = {}
    for label, sid in enumerate(meta["labels"]):
        candidates = [i for i, row in enumerate(rows) if row["label"] == label
                      and scores[i].argmax() == label and row["source"] != "oxford102"]
        # Oxford's original release has no explicit redistribution license.
        # New categories link to dataset examples instead of bundling its photos.
        if not any(row["id"] == sid and row["source"] != "oxford102" for row in rows):
            continue
        # Prefer field samples that are confident, then the strongest held-out examples.
        candidates.sort(key=lambda i: (rows[i]["source"] == "leafsnap_field" and scores[i, label] > .95,
                                       scores[i, label]), reverse=True)
        count = 0
        for i in candidates:
            row = rows[i]
            image = ImageOps.exif_transpose(Image.open(ROOT / row["path"])).convert("RGB")
            image.thumbnail((640, 640))
            blob = io.BytesIO()
            image.save(blob, "JPEG", quality=95)
            decoded = Image.open(io.BytesIO(blob.getvalue())).resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            probs = model(np.asarray(decoded, np.float32)[None], training=False).numpy()[0]
            if probs.argmax() != label or probs[label] < .95:
                continue
            name = f"{sid}-{count}.jpg"
            (samples / name).write_bytes(blob.getvalue())
            credits[name] = {"dataset": row["source"], "original": row["original"],
                             "split": "validation", "modifications": "Resized and JPEG encoded",
                             "attribution": attributions[row["original"]] if row["source"] == "tf_flowers"
                             else manifest["sources"][1]["citation"],
                             "license": "CC BY 2.0" if row["source"] == "tf_flowers" else "CC BY 4.0"}
            count += 1
            if count == 2:
                break
        if count < 2:
            raise ValueError(f"Insufficient correctly classified high-confidence samples for {sid}: {count}")
    (samples / "credits.json").write_text(json.dumps(credits, indent=2))
    for name in ("LICENSE.txt", "leafsnap-dataset-30subset-readme.txt"):
        shutil.copy2(ROOT / "data/plants" / ("leafsnap30-" + name), samples / ("leafsnap30-" + name))
    target = ROOT / "app/static/samples"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(samples, target)
    if args.species.resolve() != (ROOT / "app/species.json").resolve():
        shutil.copy2(args.species, ROOT / "app/species.json")
    previous = ROOT / "models/previous"
    previous.mkdir(exist_ok=True)
    for name in ("petals.keras", "metadata.json"):
        old = ROOT / "models" / name
        if old.exists() and not (previous / name).exists():
            old.rename(previous / name)
        shutil.copy2(args.model_dir / name, old)
    print(f"Installed {len(meta['labels'])} classes and {len(credits)} credited validation samples")


if __name__ == "__main__":
    main()
