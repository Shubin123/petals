# Petals

A small [Pl@ntNet](https://plantnet.org/en/)-style web app: photograph a flower, and Petals tells you what it is, how sure it is, and lets you keep a log of your observations.

It recognises five flowers: **daisy, dandelion, rose, sunflower and tulip**.

- **Identify**: drop, paste or take up to five photos of the same plant. Scores are averaged across photos. The result is shown as a determination slip with ranked alternatives. Anything under 60% confidence is marked undetermined.
- **Species**: a page for each flower with identification tips and sample photos from the dataset.
- **My observations**: save identifications with a place and notes (stored in SQLite).
- **How it works**: the model, its training curve, and a live demo of the data augmentation applied to your own photo.

## Where the model comes from

| Source | What Petals uses from it |
|---|---|
| [Flowers Notebook – CNN](https://www.kaggle.com/code/nikhilmishra21/flowers-notebook-cnn) | InceptionResNetV2 transfer learning at 180×180, five-class softmax head with dropout, warmup/exponential-decay learning-rate schedule, early stopping |
| [Exercise: Data Augmentation](https://www.kaggle.com/code/ryanholbrook/exercise-data-augmentation) | `RandomContrast(0.10)`, `RandomFlip('horizontal')`, `RandomRotation(0.10)` as layers inside the model, so they are active only during training |

Changes from the notebooks:

- Keras 3 layer names (`layers.RandomFlip` instead of `layers.experimental.preprocessing.RandomFlip`).
- Inputs are rescaled to [-1, 1] to match InceptionResNetV2's pretraining, instead of [0, 1].
- Training uses the public TensorFlow Flowers dataset, which has the same five classes as the Kaggle `flowers-recognition` dataset. Either one works.

## Run it

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv -r requirements.txt

# Train (downloads ~220 MB of photos into data/, writes models/petals.keras)
.venv/bin/python training/train.py --frozen-base   # head only: ~25 min on an M1, 86% accuracy
.venv/bin/python training/train.py                 # full fine-tune as in the notebook: hours on a laptop, ~93% reported

# Serve
.venv/bin/uvicorn app.main:app --port 8000
```

Open http://localhost:8000.

Training options: `--data-dir` (use your own folder of class sub-folders, e.g. Kaggle's `flowers-recognition/flowers`), `--epochs`, `--batch-size`, `--patience`.

On Apple silicon, `tensorflow-metal` trains on the GPU. TensorFlow 2.18 is pinned because later versions don't load the Metal plugin.

## GitHub Pages

https://shubin123.github.io/petals/ runs the same front end with no server: the model is converted to TensorFlow.js and runs in the browser, and observations are kept in the browser's IndexedDB. The build lives in `site/` and is committed, because the trained model in `models/` isn't.

After retraining or changing anything in `app/static/`, rebuild it:

```bash
uv pip install -p .venv tensorflowjs==4.22.0
.venv/bin/python training/build_pages.py           # add --check to measure the quantized model's accuracy
```

Pushing to `main` runs the tests and deploys `site/` (`.github/workflows/pages.yml`).

## Tests

```bash
uv pip install -p .venv -r requirements-dev.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python -m pytest
```

- `tests/test_api.py`, `tests/test_model.py`: the FastAPI server and photo preprocessing, with a stand-in model.
- `tests/test_site.py`: `site/` is complete and matches `app/static/`.
- `tests/test_pages.py`: serves `site/` under `/petals/` like GitHub Pages and drives it in Chromium: identifies every sample photo with the in-browser model, saves and deletes an observation. With a trained model in `models/`, also checks the browser's scores match Keras.
- `tests/test_build_pages.py`: the conversion helpers (needs TensorFlow; skipped otherwise).

## Layout

```
training/train.py   data pipeline, augmentation, model, LR schedule, export
training/build_pages.py  static build for GitHub Pages (TF.js model) into site/
app/main.py         FastAPI: /api/identify, /api/species, /api/observations, /api/status
app/model.py        loads models/petals.keras and scores photos
app/species.json    species descriptions
app/static/         the front end (plain HTML/CSS/JS, no build step)
```
