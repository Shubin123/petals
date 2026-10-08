"""Tests for the model conversion helpers in training/build_pages.py (need TensorFlow and tensorflowjs)."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")
# Imported by name rather than with importorskip("tensorflowjs"): build_pages stubs out the
# converter's unused optional imports first.
build_pages = pytest.importorskip("build_pages")

import keras  # noqa: E402
from keras import layers  # noqa: E402

from train import build_model  # noqa: E402


@pytest.fixture(scope="module")
def trained():
    keras.utils.set_random_seed(0)
    model = build_model(5, fine_tune=False, weights=None)
    head = model.layers[-1]
    head.set_weights([w + np.random.default_rng(0).normal(0, 0.05, w.shape) for w in head.get_weights()])
    return model


@pytest.fixture(scope="module")
def images():
    return np.random.default_rng(1).uniform(0, 255, (3, 180, 180, 3)).astype(np.float32)


def test_inference_model_drops_training_only_layers(trained):
    model = build_pages.inference_model(trained)
    kinds = {type(l) for l in model.layers}
    assert layers.Dropout not in kinds
    assert all(l.name != "augmentation" for l in model.layers)
    assert model.input_shape == (None, 180, 180, 3)
    assert model.inputs[0].name == "image"


def test_inference_model_matches_trained(trained, images):
    expected = trained.predict(images, verbose=0)
    got = build_pages.inference_model(trained).predict(images, verbose=0)
    np.testing.assert_allclose(got, expected, atol=1e-5)
    np.testing.assert_allclose(got.sum(axis=1), 1, atol=1e-5)


def test_quantize_like_tfjs(trained, images):
    model = build_pages.inference_model(trained)
    q = build_pages.quantize_like_tfjs(model)
    for orig, quant in zip(model.weights, q.weights):
        w, wq = orig.numpy(), quant.numpy()
        if w.size >= 2:
            assert len(np.unique(wq)) <= 256
            step = (w.max() - w.min()) / 255
            assert np.abs(w - wq).max() <= step / 2 + 1e-6
    # Quantizing leaves the original model untouched.
    np.testing.assert_allclose(model.predict(images, verbose=0),
                               build_pages.inference_model(trained).predict(images, verbose=0), atol=1e-6)
