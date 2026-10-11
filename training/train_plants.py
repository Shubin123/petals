"""Train a compact browser plant classifier on TF Flowers and LeafSnap30.

Run prepare_plants.py first. Frozen MobileNetV2 features are cached for an
unaltered and an augmented copy; only the softmax head is trained.
"""
import argparse
import hashlib
import json
import pathlib

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from train import ROOT, IMAGE_SIZE


def build_plant_model(classes, weights="imagenet"):
    base = keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,), include_top=False, weights=weights)
    base.trainable = False
    return keras.Sequential([
        keras.Input(shape=IMAGE_SIZE + (3,)),
        layers.Rescaling(1 / 127.5, offset=-1), base,
        layers.GlobalAveragePooling2D(), layers.Dropout(0.2),
        layers.Dense(classes, activation="softmax"),
    ], name="petals_plants")


def dataset(rows, batch_size):
    paths = [str(ROOT / r["path"]) for r in rows]
    labels = [r["label"] for r in rows]
    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(image, IMAGE_SIZE), label
    return (tf.data.Dataset.from_tensor_slices((paths, labels))
            .map(load, num_parallel_calls=4).batch(batch_size).prefetch(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=ROOT / "data/plants/manifest.json")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "models/plants")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    keras.utils.set_random_seed(1337)
    manifest = json.loads(args.manifest.read_text())
    labels = manifest["labels"]
    rows = manifest["images"]
    args.out.mkdir(parents=True, exist_ok=True)
    model = build_plant_model(len(labels))
    extractor = keras.Model(model.inputs, model.layers[-3].output)
    augment = keras.Sequential([layers.RandomContrast(0.1), layers.RandomFlip("horizontal"),
                                layers.RandomRotation(0.1)])
    cache = args.out / "features.npz"
    if cache.exists():
        cached = np.load(cache)
        if str(cached["manifest_sha256"]) != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
            raise ValueError("Feature cache belongs to a different manifest; remove it first")
        features = {key: cached[key] for key in ("train", "validation", "test")}
    else:
        features = {}
        for split in ("train", "validation", "test"):
            selected = [r for r in rows if r["split"] == split]
            ds = dataset(selected, args.batch_size)
            copies = []
            for copy in range(2 if split == "train" else 1):
                batches = []
                for i, (images, _) in enumerate(ds):
                    if copy:
                        images = augment(images, training=True)
                    batches.append(extractor(images, training=False).numpy())
                    if i % 50 == 0:
                        print(f"Features {split} copy {copy}: {i * args.batch_size}/{len(selected)}", flush=True)
                copies.append(np.concatenate(batches))
            features[split] = np.concatenate(copies)
        np.savez_compressed(cache, **{k: v.astype(np.float16) for k, v in features.items()},
                            manifest_sha256=np.array(hashlib.sha256(args.manifest.read_bytes()).hexdigest()))
    features = {k: v.astype(np.float16, copy=False) for k, v in features.items()}
    targets = {s: np.array([r["label"] for r in rows if r["split"] == s])
               for s in features}
    y_train = np.tile(targets["train"], 2)
    keras.utils.set_random_seed(1337)
    # Train the identical head on pooled features without repeatedly running the backbone.
    head = keras.Sequential([keras.Input((1280,)), layers.Dropout(0.2),
                             layers.Dense(len(labels), activation="softmax")])
    head.compile(keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = head.fit(features["train"], y_train, batch_size=64, epochs=args.epochs,
                       validation_data=(features["validation"], targets["validation"]),
                       callbacks=[keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
                                  keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5)], verbose=2)
    model.layers[-1].set_weights(head.layers[-1].get_weights())
    meta = {"labels": labels, "image_size": list(IMAGE_SIZE), "architecture": "MobileNetV2",
            "frozen_base": True, "train_images": len(targets["train"]),
            "val_images": len(targets["validation"]), "test_images": len(targets["test"]),
            "sources": manifest["sources"], "split_method": manifest["split_method"],
            "excluded_conflicting_labels": manifest.get("excluded_conflicting_labels", 0),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "hyperparameters": {"seed": 1337, "lr": 0.001, "augmented_copies": 1},
            "history": {k: [float(x) for x in v] for k, v in history.history.items()}}
    for split in ("validation", "test"):
        loss, accuracy = head.evaluate(features[split], targets[split], verbose=0)
        probs = head.predict(features[split], verbose=0)
        prefix = "val" if split == "validation" else "test"
        meta[f"{prefix}_loss"], meta[f"{prefix}_accuracy"] = float(loss), float(accuracy)
        selected = [r for r in rows if r["split"] == split]
        groups = sorted({r["source"] for r in selected})
        meta[f"{prefix}_by_source"] = {}
        for source in groups:
            idx = np.array([i for i, r in enumerate(selected) if r["source"] == source])
            meta[f"{prefix}_by_source"][source] = {
                "images": len(idx), "accuracy": float(np.mean(probs[idx].argmax(1) == targets[split][idx]))}
        meta[f"{prefix}_by_class"] = {
            sid: {"images": int(np.sum(targets[split] == i)),
                  "accuracy": float(np.mean(probs[targets[split] == i].argmax(1) == i))}
            for i, sid in enumerate(labels)}
        if split == "validation":
            np.save(args.out / "validation_scores.npy", probs)
    model.save(args.out / "petals.keras")
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k.startswith(("val_", "test_"))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
