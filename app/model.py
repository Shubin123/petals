"""Loads the trained Keras model and turns uploaded photos into species scores."""

import io
import json
import pathlib

import numpy as np
from PIL import Image, ImageOps

MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"


class Classifier:
    def __init__(self, model_dir: pathlib.Path = MODELS):
        self.model_path = model_dir / "petals.keras"
        self.meta_path = model_dir / "metadata.json"
        self.model = None
        self.meta = None

    @property
    def ready(self) -> bool:
        return self.model_path.exists() and self.meta_path.exists()

    def load(self):
        if self.model is None and self.ready:
            import keras  # imported lazily so the server starts quickly without a model

            self.meta = json.loads(self.meta_path.read_text())
            self.model = keras.models.load_model(self.model_path)
        return self.model is not None

    def _prepare(self, data: bytes) -> np.ndarray:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        img = img.resize(tuple(self.meta["image_size"]), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32)

    def predict(self, images: list[bytes]) -> list[dict]:
        """Score one or more photos of the same plant; scores are averaged across photos."""
        batch = np.stack([self._prepare(b) for b in images])
        probs = self.model.predict(batch, verbose=0).mean(axis=0)
        ranked = sorted(zip(self.meta["labels"], probs.tolist()), key=lambda p: p[1], reverse=True)
        return [{"id": label, "score": score} for label, score in ranked]
