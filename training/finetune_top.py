"""Second training stage: fine-tune the top of InceptionResNetV2 on cached features.

train.py without --frozen-base backpropagates through the whole network, which needs more
memory than an 8 GB Mac has. This script continues from a --frozen-base model instead:

1. Everything up to --after (default mixed_6a) stays frozen, batch-norm included, so its output
   for a given photo never changes. That output (9 x 9 x 1088 at 180 x 180) is computed once for
   --copies augmented versions of every training photo and kept in memory as float16.
2. The layers after it (50M of the network's 54M weights for mixed_6a) and the classification
   head are then trained on those features, using a random copy of each photo every epoch.

The result is saved in the same format as train.py's, so the app and build_pages.py use it as is.

Usage:
    python training/train.py --frozen-base                   # stage 1
    python training/finetune_top.py --init-from models/petals.keras --out models/finetuned
"""

import argparse
import json
import pathlib

import keras
import numpy as np
import tensorflow as tf

from train import (CANONICAL, IMAGE_SIZE, ROOT, base_top, build_model, copy_samples, copy_weights,
                   download_tf_flowers, learning_rate)


class CachedFeatures(keras.utils.PyDataset):
    """Batches of cached features, drawing one random augmented copy of each photo per epoch."""

    def __init__(self, features, labels, batch_size, seed=0, num_classes=None):
        super().__init__()
        self.features, self.labels, self.batch_size = features, labels, batch_size
        self.num_classes = num_classes  # one-hot labels, for label smoothing
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self):
        return -(-len(self.labels) // self.batch_size)

    def on_epoch_end(self):
        n = len(self.labels)
        self.order = self.rng.permutation(n)
        self.copy = self.rng.integers(len(self.features), size=n)

    def __getitem__(self, i):
        idx = np.sort(self.order[i * self.batch_size:(i + 1) * self.batch_size])
        x = np.stack([self.features[c][j] for c, j in zip(self.copy[idx], idx)]).astype(np.float32)
        y = self.labels[idx]
        return x, (np.eye(self.num_classes, dtype=np.float32)[y] if self.num_classes else y)


def in_order(files, class_dirs, batch_size):
    """Photos in a fixed order (features are cached per photo), loaded as image_dataset_from_directory does."""
    labels = np.array([class_dirs.index(pathlib.Path(f).parent.name) for f in files])

    def load(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(img, IMAGE_SIZE)

    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(lambda f, y: (load(f), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), labels, files


def extract(bottom, rescale, augment, ds, copies):
    """Run the frozen bottom of the network over ds, once per copy; copy 0 is not augmented."""
    out = []
    for c in range(copies):
        feats = []
        for images, _ in ds:
            if c:
                images = augment(images, training=True)
            feats.append(keras.ops.convert_to_numpy(bottom(rescale(images), training=False)).astype(np.float16))
        out.append(np.concatenate(feats))
        print(f"Cached features for copy {c + 1}/{copies}: {out[-1].shape}")
    return out


class Checkpoint(keras.callbacks.Callback):
    """Saves the trained weights (float16) and history whenever val_loss improves.

    On a machine short of memory the run may be stopped before it finishes; --finalize then
    builds the model from the last checkpoint.
    """

    def __init__(self, path):
        super().__init__()
        self.path, self.best, self.history = path, np.inf, {}

    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(float(v))
        if logs["val_loss"] < self.best:
            self.best = logs["val_loss"]
            self.path.unlink(missing_ok=True)
            np.savez(self.path, *[w.numpy().astype(np.float16) for w in self.model.trainable_weights])
            # Like EarlyStopping(restore_best_weights=True): the final model is the best epoch's.
            acc = logs.get("val_accuracy", logs.get("val_categorical_accuracy"))
            self.path.with_suffix(".json").write_text(json.dumps(
                {"epoch": epoch + 1, "val_loss": logs["val_loss"], "val_accuracy": acc, "history": self.history}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-from", type=pathlib.Path, default=ROOT / "models" / "petals.keras",
                        help="a model trained with train.py --frozen-base")
    parser.add_argument("--init-meta", type=pathlib.Path,
                        help="its metadata.json, to show both stages in the training history")
    parser.add_argument("--after", default="mixed_6a", help="fine-tune the layers after this one")
    parser.add_argument("--copies", type=int, default=3, help="augmented copies of each training photo")
    parser.add_argument("--data-dir", type=pathlib.Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr-max", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--train-bn", action="store_true",
                        help="let batch-norm layers after --after update their statistics")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="use AdamW with this decay")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "models")
    parser.add_argument("--samples-dir", type=pathlib.Path, default=ROOT / "app" / "static" / "samples")
    parser.add_argument("--finalize", action="store_true",
                        help="don't train: save the model from the checkpoint of an interrupted run in --out")
    args = parser.parse_args()
    keras.utils.set_random_seed(args.seed)

    data_dir = args.data_dir or download_tf_flowers(ROOT / "data")
    # Same split as train.py, so validation photos were never trained on in either stage.
    split = keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="both", seed=1337, image_size=IMAGE_SIZE)
    class_dirs = split[0].class_names
    labels = [CANONICAL[c] for c in class_dirs]
    (train_ds, y_train, train_files), (val_ds, y_val, val_files) = [
        in_order(ds.file_paths, class_dirs, args.batch_size) for ds in split]

    model = build_model(len(labels), fine_tune=True, weights=None, freeze_bn=not args.train_bn,
                        fine_tune_after=args.after)
    copy_weights(keras.models.load_model(args.init_from), model)
    augment = model.get_layer("augmentation")
    rescale = next(l for l in model.layers if isinstance(l, keras.layers.Rescaling))
    base = next(l for l in model.layers if isinstance(l, keras.Model) and l.name != "augmentation")
    bottom = keras.Model(base.input, base.get_layer(args.after).output)
    top = base_top(base, args.after)

    # The layers after the base, shared with `model`, so training updates it in place.
    after_base = model.layers[model.layers.index(base) + 1:]
    trainer = keras.Sequential([keras.Input(top.input.shape[1:]), top, *after_base])
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.out / "checkpoint.npz")

    if args.finalize:
        saved = np.load(checkpoint.path)
        for w, i in zip(trainer.trainable_weights, range(len(saved.files)), strict=True):
            w.assign(saved[f"arr_{i}"].astype(np.float32))
        state = json.loads(checkpoint.path.with_suffix(".json").read_text())
        print(f"Finalizing from epoch {state['epoch']}: val_accuracy {state['val_accuracy']:.4f}")
        write_outputs(args, model, labels, class_dirs, y_train, y_val, val_files,
                      state["val_accuracy"], state["val_loss"], state["history"])
        return

    train_feats = extract(bottom, rescale, augment, train_ds, args.copies)
    val_feats = extract(bottom, rescale, augment, val_ds, 1)
    del bottom
    smooth = args.label_smoothing > 0
    trainer.compile(
        optimizer=keras.optimizers.AdamW(weight_decay=args.weight_decay) if args.weight_decay else "adam",
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing) if smooth
        else "sparse_categorical_crossentropy",
        metrics=["categorical_accuracy" if smooth else "accuracy"])
    num_classes = len(labels) if smooth else None
    print(f"Training {sum(np.prod(w.shape) for w in trainer.trainable_weights) / 1e6:.1f}M weights")

    history = trainer.fit(
        CachedFeatures(train_feats, y_train, args.batch_size, args.seed, num_classes),
        validation_data=CachedFeatures(val_feats, y_val, args.batch_size, num_classes=num_classes),
        epochs=args.epochs,
        callbacks=[
            keras.callbacks.LearningRateScheduler(
                lambda epoch: learning_rate(epoch, args.lr_max, args.warmup), verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                          verbose=1, restore_best_weights=True),
            checkpoint,
        ], verbose=2)

    # The frozen bottom gives the same features at inference, so this is the full model's accuracy.
    loss, accuracy = trainer.evaluate(CachedFeatures(val_feats, y_val, args.batch_size, num_classes=num_classes),
                                      verbose=0)
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    del train_feats, val_feats
    write_outputs(args, model, labels, class_dirs, y_train, y_val, val_files, accuracy, loss, history.history)
    checkpoint.path.unlink(missing_ok=True)


def write_outputs(args, model, labels, class_dirs, y_train, y_val, val_files, accuracy, loss, history):
    # `model` was never compiled or trained itself, so it saves without optimizer state.
    model.save(args.out / "petals.keras")

    # Name the metrics as train.py does, for the app's training chart.
    stage2 = {k.replace("categorical_accuracy", "accuracy"): [float(v) for v in vs]
              for k, vs in history.items()}
    hist, stage1_epochs = stage2, 0
    if args.init_meta:
        stage1 = json.loads(args.init_meta.read_text())["history"]
        stage1_epochs = len(stage1["accuracy"])
        hist = {k: stage1.get(k, []) + stage2[k] for k in stage2}
    meta = {
        "labels": labels,
        "image_size": list(IMAGE_SIZE),
        "val_accuracy": accuracy,
        "val_loss": loss,
        "train_images": len(y_train),
        "val_images": len(y_val),
        "frozen_base": False,
        "hyperparameters": {"method": "cached features", "fine_tune_after": args.after,
                            "copies": args.copies, "lr_max": args.lr_max, "warmup": args.warmup,
                            "batch_size": args.batch_size, "freeze_bn": not args.train_bn,
                            "patience": args.patience, "seed": args.seed,
                            "label_smoothing": args.label_smoothing, "weight_decay": args.weight_decay,
                            "init_from": args.init_from.name, "stage1_epochs": stage1_epochs},
        "history": hist,
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    copy_samples(model, class_dirs, val_files, args.samples_dir)
    print(f"Saved model to {args.out / 'petals.keras'}")


if __name__ == "__main__":
    main()
