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

## Layout

```
training/train.py   data pipeline, augmentation, model, LR schedule, export
app/main.py         FastAPI: /api/identify, /api/species, /api/observations, /api/status
app/model.py        loads models/petals.keras and scores photos
app/species.json    species descriptions
app/static/         the front end (plain HTML/CSS/JS, no build step)
```
