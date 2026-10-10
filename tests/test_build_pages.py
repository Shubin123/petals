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


def test_copy_weights_between_frozen_and_fine_tuned(trained, images):
    from train import copy_weights

    for freeze_bn in (False, True):
        target = build_model(5, fine_tune=True, weights=None, freeze_bn=freeze_bn)
        copy_weights(trained, target)
        np.testing.assert_allclose(target.predict(images, verbose=0), trained.predict(images, verbose=0), atol=1e-5)
        back = build_model(5, fine_tune=False, weights=None)
        copy_weights(target, back)
        np.testing.assert_allclose(back.predict(images, verbose=0), trained.predict(images, verbose=0), atol=1e-5)


def test_copy_weights_rejects_other_architectures(trained):
    from train import copy_weights

    with pytest.raises(ValueError):
        copy_weights(trained, build_model(3, fine_tune=True, weights=None))


def test_freeze_bn_only_freezes_batch_norm():
    model = build_model(5, fine_tune=True, weights=None, freeze_bn=True)
    base = next(l for l in model.layers if isinstance(l, keras.Model) and l.name != "augmentation")
    bn = [l for l in base.layers if isinstance(l, layers.BatchNormalization)]
    assert bn and not any(l.trainable for l in bn)
    assert all(l.trainable for l in base.layers if isinstance(l, layers.Conv2D))


def test_learning_rate_schedule():
    from train import learning_rate

    assert learning_rate(0) == pytest.approx(1e-5)
    assert learning_rate(8) == pytest.approx(4e-4)
    assert learning_rate(9) < learning_rate(8)
    assert learning_rate(0, lr_max=1e-4, lr_rampup_epochs=2) == pytest.approx(1e-5)
    assert learning_rate(2, lr_max=1e-4, lr_rampup_epochs=2) == pytest.approx(1e-4)


def test_fine_tune_after_trains_only_the_top():
    from train import base_top

    model = build_model(5, fine_tune=True, weights=None, freeze_bn=True, fine_tune_after="block17_20_ac")
    base = next(l for l in model.layers if isinstance(l, keras.Model) and l.name != "augmentation")
    top = {id(l) for l in base_top(base, "block17_20_ac").layers}
    trainable = [l for l in base.layers if l.trainable and l.weights]
    assert trainable and all(id(l) in top for l in trainable)
    # Every convolution after the split trains, including mixed_7a's branches, which come before
    # mixed_7a itself in base.layers.
    convs = [l for l in base_top(base, "block17_20_ac").layers if isinstance(l, layers.Conv2D)]
    assert len(convs) > 50 and all(l.trainable for l in convs)
    assert not base.get_layer("block17_20_conv").trainable
    n_trainable = sum(np.prod(w.shape) for w in model.trainable_weights)
    assert 27e6 < n_trainable < 28e6


def test_cached_features_draws_one_copy_per_photo():
    from finetune_top import CachedFeatures

    n = 10
    feats = [np.full((n, 2), c, np.float16) + np.arange(n, dtype=np.float16)[:, None] * 10 for c in range(3)]
    labels = np.arange(n)
    ds = CachedFeatures(feats, labels, batch_size=4, seed=0)
    assert len(ds) == 3
    seen = []
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.dtype == np.float32
        # Each row is photo y's features from one of the copies.
        assert np.all((x[:, 0] - y * 10 >= 0) & (x[:, 0] - y * 10 <= 2))
        seen.extend(y)
    assert sorted(seen) == list(range(n))
    first = ds.copy.copy()
    ds.on_epoch_end()
    assert not np.array_equal(first, ds.copy) or n < 3
