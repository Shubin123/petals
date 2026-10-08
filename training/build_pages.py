"""Build a static copy of Petals for GitHub Pages.

The model runs in the browser with TensorFlow.js and observations are kept in the
browser's IndexedDB, so no server is needed.

Usage:
    python training/build_pages.py            # writes site/
    python training/build_pages.py --check    # also measures quantized accuracy on the validation set
"""

import argparse
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


def quantize_like_tfjs(model: keras.Model) -> keras.Model:
    """Apply the same per-tensor uint8 affine quantization tfjs uses, to measure its accuracy cost."""
    q = keras.models.clone_model(model)
    q.set_weights(model.get_weights())
    for var in q.weights:
        w = var.numpy()
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
    args = parser.parse_args()

    trained = keras.models.load_model(ROOT / "models" / "petals.keras")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    model = inference_model(trained)

    if args.check:
        val = keras.utils.image_dataset_from_directory(
            ROOT / "data" / "flower_photos", validation_split=0.2, subset="validation",
            seed=1337, image_size=IMAGE_SIZE, batch_size=32)
        for name, m in (("float32", model), ("uint8", quantize_like_tfjs(model))):
            m.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            print(f"{name} validation accuracy: {m.evaluate(val, verbose=0)[1]:.4f}")

    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(STATIC, args.out)

    with tempfile.TemporaryDirectory() as tmp:
        model.export(tmp, format="tf_saved_model", verbose=False)
        tfjs.convert_tf_saved_model(tmp, str(args.out / "model"), quantization_dtype_map={"uint8": True})

    (args.out / "model" / "metadata.json").write_text(json.dumps(meta))

    species = json.loads((ROOT / "app" / "species.json").read_text())
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
