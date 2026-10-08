"""Train the Petals flower classifier.

Model and schedule follow the Kaggle "Flowers Notebook - CNN" (nikhilmishra21):
InceptionResNetV2 fine-tuned on 180x180 images, five classes, warmup /
exponential-decay learning rate and early stopping.

Augmentation follows the Kaggle Learn "Data Augmentation" exercise
(ryanholbrook): RandomContrast(0.10), RandomFlip('horizontal'),
RandomRotation(0.10) as layers inside the model, so they are active during
training and switched off automatically at inference.

Usage:
    python training/train.py                       # downloads TF Flowers
    python training/train.py --data-dir path/to/flowers   # e.g. Kaggle flowers-recognition
"""

import argparse
import json
import pathlib
import shutil
import tarfile
import urllib.request

import keras
import tensorflow as tf
from keras import layers

ROOT = pathlib.Path(__file__).resolve().parent.parent
TF_FLOWERS_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# Folder names differ between TF Flowers ("roses") and Kaggle flowers-recognition ("rose").
CANONICAL = {"daisy": "daisy", "dandelion": "dandelion", "rose": "rose", "roses": "rose",
             "sunflower": "sunflower", "sunflowers": "sunflower", "tulip": "tulip", "tulips": "tulip"}

IMAGE_SIZE = (180, 180)


def download_tf_flowers(dest: pathlib.Path) -> pathlib.Path:
    target = dest / "flower_photos"
    if target.exists():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "flower_photos.tgz"
    print(f"Downloading {TF_FLOWERS_URL}")
    urllib.request.urlretrieve(TF_FLOWERS_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    archive.unlink()
    (target / "LICENSE.txt").unlink(missing_ok=True)
    return target


def learning_rate(epoch):
    lr_start = 0.00001
    lr_max = 0.0004
    lr_min = 0.00001
    lr_rampup_epochs = 8
    lr_sustain_epochs = 0
    lr_exp_decay = 0.8

    if epoch < lr_rampup_epochs:
        return (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
    if epoch < lr_rampup_epochs + lr_sustain_epochs:
        return lr_max
    return (lr_max - lr_min) * lr_exp_decay ** (epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min


def build_model(num_classes: int, fine_tune: bool, weights="imagenet") -> keras.Model:
    augment = keras.Sequential([
        layers.RandomContrast(factor=0.10),
        layers.RandomFlip(mode="horizontal"),
        layers.RandomRotation(factor=0.10),
    ], name="augmentation")

    conv_base = keras.applications.InceptionResNetV2(
        weights=weights, include_top=False, input_shape=IMAGE_SIZE + (3,))
    conv_base.trainable = fine_tune

    return keras.Sequential([
        keras.Input(shape=IMAGE_SIZE + (3,)),
        augment,
        # InceptionResNetV2 was pretrained on [-1, 1] inputs; the notebook's 1/255 works
        # but matching the pretraining range converges faster.
        layers.Rescaling(1.0 / 127.5, offset=-1),
        conv_base,
        layers.Dropout(0.2),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ], name="petals")


def copy_samples(model, class_dirs: list[str], val_files: list[str], out: pathlib.Path, per_class=4):
    """Copy a few validation photos per class for the species pages.

    TF Flowers has some mislabelled photos, so only photos the model confidently gets right are used.
    """
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for label, folder in enumerate(class_dirs):
        picked = 0
        for src in (f for f in val_files if pathlib.Path(f).parent.name == folder):
            img = keras.utils.img_to_array(keras.utils.load_img(src, target_size=IMAGE_SIZE))
            probs = model.predict(img[None], verbose=0)[0]
            if probs.argmax() == label and probs[label] >= 0.95:
                shutil.copy(src, out / f"{CANONICAL[folder]}-{picked}.jpg")
                picked += 1
                if picked == per_class:
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, help="folder with one sub-folder per class")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--frozen-base", action="store_true",
                        help="train only the head (much faster, a few points less accurate)")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "models")
    args = parser.parse_args()

    data_dir = args.data_dir or download_tf_flowers(ROOT / "data")

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="both", seed=1337,
        image_size=IMAGE_SIZE, batch_size=args.batch_size)
    class_dirs = train_ds.class_names
    labels = [CANONICAL[c] for c in class_dirs]
    val_files = val_ds.file_paths
    train_count = len(train_ds.file_paths)
    print("Classes:", labels)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(len(labels), fine_tune=not args.frozen_base)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    args.out.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        train_ds, epochs=args.epochs, validation_data=val_ds,
        callbacks=[
            keras.callbacks.LearningRateScheduler(learning_rate, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                          verbose=1, restore_best_weights=True),
        ])

    loss, accuracy = model.evaluate(val_ds, verbose=1)
    print(f"Validation accuracy: {accuracy * 100:.2f}%")

    # Save an uncompiled copy: Adam's state would triple the file size and isn't needed to serve.
    inference = build_model(len(labels), fine_tune=not args.frozen_base, weights=None)
    inference.set_weights(model.get_weights())
    inference.save(args.out / "petals.keras")
    meta = {
        "labels": labels,
        "image_size": list(IMAGE_SIZE),
        "val_accuracy": accuracy,
        "val_loss": loss,
        "train_images": train_count,
        "val_images": len(val_files),
        "frozen_base": args.frozen_base,
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    copy_samples(model, class_dirs, val_files, ROOT / "app" / "static" / "samples")
    print(f"Saved model to {args.out / 'petals.keras'}")


if __name__ == "__main__":
    main()
