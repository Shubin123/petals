"""Petals web app: identify flowers from photos and keep a log of observations."""

import json
import pathlib
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from .model import Classifier

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data"
UPLOADS = DATA / "uploads"
DB_PATH = DATA / "petals.db"

SPECIES = json.loads((HERE / "species.json").read_text())
MAX_PHOTOS = 5
MAX_BYTES = 15 * 1024 * 1024
# Below this the top match is shown as uncertain: the model only knows five species.
CONFIDENT = 0.6

classifier = Classifier()


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@asynccontextmanager
async def lifespan(_app):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS observations (
            id TEXT PRIMARY KEY, species TEXT NOT NULL, score REAL NOT NULL,
            photos TEXT NOT NULL, note TEXT, place TEXT, created TEXT NOT NULL)""")
    await run_in_threadpool(classifier.load)
    yield


app = FastAPI(title="Petals", lifespan=lifespan)


async def read_photos(photos: list[UploadFile]) -> list[bytes]:
    if not photos:
        raise HTTPException(400, "Add at least one photo.")
    if len(photos) > MAX_PHOTOS:
        raise HTTPException(400, f"Use up to {MAX_PHOTOS} photos of the same plant.")
    data = []
    for photo in photos:
        if not (photo.content_type or "").startswith("image/"):
            raise HTTPException(400, f"{photo.filename} is not an image.")
        blob = await photo.read()
        if len(blob) > MAX_BYTES:
            raise HTTPException(400, f"{photo.filename} is larger than 15 MB.")
        data.append(blob)
    return data


@app.get("/api/status")
def status():
    meta = classifier.meta or {}
    return {
        "ready": classifier.model is not None,
        "labels": meta.get("labels"),
        "val_accuracy": meta.get("val_accuracy"),
        "train_images": meta.get("train_images"),
        "val_images": meta.get("val_images"),
        "history": meta.get("history"),
        "frozen_base": meta.get("frozen_base"),
        "hyperparameters": meta.get("hyperparameters"),
        "confident_threshold": CONFIDENT,
    }


@app.get("/api/species")
def species():
    samples = HERE / "static" / "samples"
    return {
        sid: {**info, "id": sid,
              "samples": sorted(f"/samples/{p.name}" for p in samples.glob(f"{sid}-*.jpg"))}
        for sid, info in SPECIES.items()
    }


@app.post("/api/identify")
async def identify(photos: list[UploadFile] = File(...)):
    if classifier.model is None:
        raise HTTPException(503, "No trained model found. Run training/train.py, then restart the server.")
    data = await read_photos(photos)
    try:
        results = await run_in_threadpool(classifier.predict, data)
    except OSError:
        raise HTTPException(400, "One of the photos could not be read. Try a JPEG or PNG.")
    return {"results": results, "confident": results[0]["score"] >= CONFIDENT}


@app.get("/api/observations")
def list_observations():
    with db() as conn:
        rows = conn.execute("SELECT * FROM observations ORDER BY created DESC").fetchall()
    return [{**dict(r), "photos": json.loads(r["photos"])} for r in rows]


@app.post("/api/observations")
async def save_observation(
    photos: list[UploadFile] = File(...),
    species: str = Form(...),
    score: float = Form(...),
    note: str = Form(""),
    place: str = Form(""),
):
    if species not in SPECIES:
        raise HTTPException(400, "Unknown species.")
    data = await read_photos(photos)
    obs_id = uuid.uuid4().hex
    names = []
    for i, blob in enumerate(data):
        ext = pathlib.Path(photos[i].filename or "").suffix.lower()
        name = f"{obs_id}-{i}{ext if ext in {'.jpg', '.jpeg', '.png', '.webp'} else '.jpg'}"
        (UPLOADS / name).write_bytes(blob)
        names.append(f"/uploads/{name}")
    row = (obs_id, species, score, json.dumps(names), note.strip()[:500], place.strip()[:120],
           datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with db() as conn:
        conn.execute("INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?)", row)
    return {"id": obs_id}


@app.delete("/api/observations/{obs_id}")
def delete_observation(obs_id: str):
    with db() as conn:
        row = conn.execute("SELECT photos FROM observations WHERE id = ?", (obs_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Observation not found.")
        conn.execute("DELETE FROM observations WHERE id = ?", (obs_id,))
    for url in json.loads(row["photos"]):
        (UPLOADS / pathlib.Path(url).name).unlink(missing_ok=True)
    return {"deleted": obs_id}


UPLOADS.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")
app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")
