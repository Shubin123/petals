"""Build a static copy of Petals for GitHub Pages.

The model runs in the browser with TensorFlow.js and observations are kept in the
browser's IndexedDB, so no server is needed.

Usage:
    python training/build_pages.py            # writes site/
    python training/build_pages.py --check    # also measures quantized accuracy on the validation set
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile
import types

import keras
import numpy as np
import tensorflow as tf
from keras import layers

# The tfjs converter imports the JAX and decision-forest backends at module load even
# though a Keras/SavedModel conversion never uses them; register empty stand-ins.
for name in ("tensorflow_decision_forests", "tensorflowjs.converters.jax_conversion"):
    stub = types.ModuleType(name)
    stub.convert_jax = None
    sys.modules.setdefault(name, stub)
from tensorflowjs.converters import tf_saved_model_conversion_v2 as tfjs  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train import IMAGE_SIZE, ROOT  # noqa: E402

STATIC = ROOT / "app" / "static"


def inference_model(trained: keras.Model) -> keras.Model:
    """Rebuild the trained model without augmentation or dropout, which do nothing at inference."""
    base = next(l for l in trained.layers if isinstance(l, keras.Model) and l.name != "augmentation")
    head = trained.layers[-1]
    rescale = next(l for l in trained.layers if isinstance(l, layers.Rescaling))
    inputs = keras.Input(shape=IMAGE_SIZE + (3,), name="image")
    x = layers.Rescaling(rescale.scale, offset=rescale.offset)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(head.units, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.layers[-1].set_weights(head.get_weights())
    return model


def quantize_like_tfjs(model: keras.Model, dtype="uint8") -> keras.Model:
    """Measure float16 or per-tensor uint8 weight storage error before browser export."""
    q = keras.models.clone_model(model)
    q.set_weights(model.get_weights())
    for var in q.weights:
        w = var.numpy()
        if dtype == "float16":
            var.assign(w.astype(np.float16).astype(w.dtype))
            continue
        if w.dtype != np.float32 or w.size < 2:
            continue
        lo, hi = float(w.min()), float(w.max())
        scale = (hi - lo) / 255 or 1.0
        var.assign((np.round((w - lo) / scale) * scale + lo).astype(np.float32))
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "site")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model-dir", type=pathlib.Path, default=ROOT / "models")
    parser.add_argument("--manifest", type=pathlib.Path,
                        help="prepared plant manifest for validation of the expanded model")
    parser.add_argument("--quantization", choices=("uint8", "float16"),
                        help="default: float16 for MobileNetV2, uint8 for the original model")
    args = parser.parse_args()

    trained = keras.models.load_model(args.model_dir / "petals.keras")
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    model = inference_model(trained)
    quantization = args.quantization or ("float16" if meta.get("architecture") == "MobileNetV2" else "uint8")
    meta["quantization"] = quantization

    if args.check:
        if meta.get("architecture") == "MobileNetV2" and not args.manifest:
            parser.error("Expanded plant validation requires --manifest data/plants/manifest.json")
        if args.manifest:
            from train_plants import dataset
            manifest = json.loads(args.manifest.read_text())
            if manifest["labels"] != meta["labels"]:
                raise ValueError("Validation labels do not match the model")
            val = dataset([r for r in manifest["images"] if r["split"] == "validation"], 32)
        else:
            val = keras.utils.image_dataset_from_directory(
                ROOT / "data" / "flower_photos", validation_split=0.2, subset="validation",
                seed=1337, image_size=IMAGE_SIZE, batch_size=32)
        for name, m in (("float32", model), (quantization, quantize_like_tfjs(model, quantization))):
            m.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            accuracy = float(m.evaluate(val, verbose=0)[1])
            print(f"{name} validation accuracy: {accuracy:.4f}")
            meta[f"{name}_val_accuracy"] = accuracy
            if name != "float32" and accuracy < meta["float32_val_accuracy"] - 0.02:
                raise ValueError(f"{name} compression loses more than two accuracy points; refusing export")

    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(STATIC, args.out)

    with tempfile.TemporaryDirectory() as tmp:
        model.export(tmp, format="tf_saved_model", verbose=False)
        tfjs.convert_tf_saved_model(tmp, str(args.out / "model"), quantization_dtype_map={quantization: True})

    # Immutable weight names prevent cached shards from an older class count
    # being combined with a newly downloaded graph.
    graph_path = args.out / "model/model.json"
    graph = json.loads(graph_path.read_text())
    for group in graph["weightsManifest"]:
        paths = []
        for name in group["paths"]:
            shard = args.out / "model" / name
            digest = hashlib.sha256(shard.read_bytes()).hexdigest()[:12]
            renamed = shard.with_name(f"{shard.stem}-{digest}{shard.suffix}")
            shard.rename(renamed)
            paths.append(renamed.name)
        group["paths"] = paths
    graph_path.write_text(json.dumps(graph))
    meta["model_version"] = hashlib.sha256(graph_path.read_bytes()).hexdigest()

    (args.out / "model" / "metadata.json").write_text(json.dumps(meta))
    (args.model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    species = json.loads((ROOT / "app" / "species.json").read_text())
    species = {sid: species[sid] for sid in meta["labels"]}
    for sid, info in species.items():
        info["id"] = sid
        info["samples"] = sorted(f"samples/{p.name}" for p in (STATIC / "samples").glob(f"{sid}-*.jpg"))
    (args.out / "species.json").write_text(json.dumps(species))

    # Tells the front end to use the in-browser model instead of the API.
    index = args.out / "index.html"
    index.write_text(index.read_text().replace('<html lang="en">', '<html lang="en" data-mode="static">', 1))
    (args.out / ".nojekyll").touch()

    size = sum(p.stat().st_size for p in (args.out / "model").glob("*")) / 1e6
    print(f"Built {args.out} (model {size:.0f} MB)")


if __name__ == "__main__":
    main()
