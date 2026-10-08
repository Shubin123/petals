import json

import numpy as np
import pytest
from PIL import Image

from app.model import Classifier
from conftest import LABELS, jpeg


class FakeKeras:
    """Stands in for the Keras model: returns fixed probabilities per photo and records its input."""

    def __init__(self, rows):
        self.rows = np.array(rows, dtype=np.float32)
        self.batch = None

    def predict(self, batch, verbose=0):
        self.batch = batch
        return self.rows[: len(batch)]


@pytest.fixture
def classifier(tmp_path):
    c = Classifier(tmp_path)
    c.meta = {"labels": LABELS, "image_size": [180, 180]}
    return c


def test_not_ready_without_files(tmp_path):
    c = Classifier(tmp_path)
    assert not c.ready
    assert c.load() is False
    assert c.model is None


def test_ready_needs_model_and_metadata(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"labels": LABELS}))
    assert not Classifier(tmp_path).ready
    (tmp_path / "petals.keras").write_bytes(b"")
    assert Classifier(tmp_path).ready


def test_prepare_resizes_to_model_input(classifier):
    arr = classifier._prepare(jpeg((255, 0, 0), size=(400, 300)))
    assert arr.shape == (180, 180, 3)
    assert arr.dtype == np.float32
    # Pixel values stay in 0-255; the model rescales them itself.
    assert 240 <= arr[..., 0].mean() <= 255
    assert arr[..., 1:].mean() < 15


def test_prepare_converts_greyscale_and_alpha_to_rgb(classifier):
    import io

    for mode in ("L", "RGBA", "P"):
        buf = io.BytesIO()
        Image.new(mode, (50, 50)).save(buf, "PNG")
        assert classifier._prepare(buf.getvalue()).shape == (180, 180, 3)


def test_prepare_applies_exif_rotation(classifier):
    import io

    img = Image.new("RGB", (200, 100), (0, 0, 255))
    img.paste((255, 0, 0), (0, 0, 100, 100))  # left half red, right half blue
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° clockwise when displayed
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    arr = classifier._prepare(buf.getvalue())
    # After rotating clockwise the red half is on top.
    assert arr[:60, :, 0].mean() > 200
    assert arr[-60:, :, 2].mean() > 200


def test_predict_averages_photos_and_ranks(classifier):
    classifier.model = FakeKeras([[0.6, 0.1, 0.1, 0.1, 0.1], [0.0, 0.1, 0.1, 0.1, 0.7]])
    results = classifier.predict([jpeg(), jpeg()])
    assert classifier.model.batch.shape == (2, 180, 180, 3)
    assert [r["id"] for r in results] == ["tulip", "daisy", "dandelion", "rose", "sunflower"]
    assert results[0]["score"] == pytest.approx(0.4)
    assert results[1]["score"] == pytest.approx(0.3)
    assert sum(r["score"] for r in results) == pytest.approx(1.0)


def test_predict_rejects_non_image(classifier):
    classifier.model = FakeKeras([[1, 0, 0, 0, 0]])
    with pytest.raises(OSError):
        classifier.predict([b"not an image"])
