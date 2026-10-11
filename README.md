# Petals

Petals identifies **35 plant classes** from photos or a live camera: the original daisy, dandelion, rose, sunflower and tulip classes, plus **30 woody plant species from their leaves**. The [web app](https://shubin123.github.io/petals/) runs a compact, quantized TensorFlow.js model entirely in the browser and stores observations locally.

The tree coverage includes maples, elms, oaks, white fir, holly, pawpaw, redbud, honey locust, and Japanese cedar. Browse **Species** for the complete list. Flower classes such as rose and tulip cover groups of species; they do not determine an individual botanical species or cultivar.

- **Identify**: use the always-on live camera, or drop/select up to five photos of the same plant. To paste, copy the image itself (not its address), then click **Paste image** or press **⌘V / Ctrl+V** on the Identify page. The paste button requests clipboard access when required. Scores are averaged; matches below 60% are undetermined.
  Identification shows an animated loading bar while the model runs. Images dragged from browser tabs are downloaded when the source website allows it; blocked sources show instructions to use Copy image → Paste image. File drops are accepted across the Identify page.
- **Species**: leaf/flower guidance and credited validation photos for each supported class.
- **My observations**: save photos with a place and notes, using IndexedDB on Pages or SQLite with the optional server.
- **How it works**: model coverage, validation/test results, data credits, training curve, and an interactive augmentation demo.

Use a close-up flower photo for the five flower classes. For trees, photograph a single leaf, complete compound leaf, or short needle-bearing twig against a plain, light background. The leaf data does not establish accuracy on whole trees, bark, dense foliage, or every plant encountered outdoors. Unsupported plants can receive high softmax scores. Scores compare the supported classes and are not a reliable unknown-plant detector.

## Data and training

The expanded model uses frozen **MobileNetV2 ImageNet features** with a newly trained 35-class softmax head. Features are cached for an unaltered and an augmented copy of every training photo, so repeated head training is inexpensive. The existing 180×180 preprocessing is retained in Python and the browser. The old InceptionResNetV2 five-flower model is preserved locally under `models/previous/` when the expanded model is installed.

| Source | Use and credit |
|---|---|
| [TensorFlow Flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) | Five flower classes. Photos are CC BY 2.0; individual Flickr photographers are credited in the original `LICENSE.txt` and the distributed sample credits. |
| [LeafSnap30 v1.0](https://doi.org/10.5281/zenodo.5061353) | 30 woody plant species, with cropped lab leaves and mobile field photos. Ranguelova, Meijer, Oostrum, Liu and Bos, Netherlands eScience Center (2021), CC BY 4.0. Archive MD5: `cf24bee185d2c2e42b8a18effc5ef554`. |
| [Original Leafsnap](https://leafsnap.com/dataset/) | Kumar, Belhumeur, Biswas, Jacobs, Kress, Lopez and Soares, *Leafsnap: A Computer Vision System for Automatic Plant Species Identification*, ECCV 2012. LeafSnap30 derives from this dataset; the app does not cover all 185 original species. |

Preparation checks the archive checksum, reads only the indexed cropped lab/original field photos, corrects two scientific-name typos, and excludes 240 photos whose identical bytes have conflicting class labels. The resulting 10,825 photos are split into 7,593 training, 1,615 validation and 1,617 test photos. The split is deterministic and stratified by class and source. Lab leaf views sharing a filename stem stay together; exact duplicate contents never cross partitions. Field tree/individual IDs are unavailable, so this is not a guarantee of independence between photographed trees. Validation selects the best epoch; test photos are never used for training or early stopping.

Exact validation/test metrics, per-source accuracy, and training history are recorded in [`site/model/metadata.json`](site/model/metadata.json). The build checks float16 weight compression against the full validation split. Distributed samples are high-confidence validation examples, **not an accuracy benchmark**. Their original paths, licenses, authors, and resizing modifications are recorded in [`site/samples/credits.json`](site/samples/credits.json).

## Reproduce the expanded model

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv -r requirements.txt
uv pip install -p .venv tensorflowjs==4.22.0

# Downloads LeafSnap30 (~264 MB) and TF Flowers if absent.
.venv/bin/python training/prepare_plants.py
.venv/bin/python training/train_plants.py
.venv/bin/python training/install_plants.py
.venv/bin/python training/build_pages.py --check --manifest data/plants/manifest.json

# Optional local API; Pages requires no server or API key.
.venv/bin/uvicorn app.main:app --port 8000
```

The flower archive's `data/flower_photos/LICENSE.txt` is retained before distributing samples. Preparation restores it from the original archive if an older training run deleted it. Feature caches are checked against the manifest; remove `models/plants/features.npz` before using changed data. Training options include `--epochs`, `--batch-size`, `--out`, and `--manifest`. Apple silicon uses `tensorflow-metal`; TensorFlow 2.18 is pinned for plugin compatibility.

The original five-flower training scripts remain available in `training/train.py` and `training/finetune_top.py`. Their InceptionResNetV2 architecture and augmentation were adapted from [Flowers Notebook – CNN](https://www.kaggle.com/code/nikhilmishra21/flowers-notebook-cnn) and [Exercise: Data Augmentation](https://www.kaggle.com/code/ryanholbrook/exercise-data-augmentation).

## Browser build and tests

The committed `site/` includes the converted model. After changing `app/static/`, rebuild with `training/build_pages.py`. Pushing to `main` runs tests and deploys GitHub Pages through `.github/workflows/pages.yml`.

```bash
uv pip install -p .venv -r requirements-dev.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python -m pytest
```

Tests cover API validation, preprocessing, data split isolation, source/build synchronization, attributed sample coverage, real Chromium inference for every class, browser/Keras score agreement, live camera lifecycle, and saving/deleting observations. TensorFlow conversion checks run when TensorFlow and tensorflowjs are installed; Pages inference tests also run in CI without Python TensorFlow.
