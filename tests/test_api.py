import pytest
from fastapi.testclient import TestClient

from app import main
from conftest import LABELS, jpeg


class FakeClassifier:
    def __init__(self, scores):
        self.model = object()
        self.meta = {"labels": LABELS, "val_accuracy": 0.9, "train_images": 10, "val_images": 2,
                     "history": {"accuracy": [0.5], "val_accuracy": [0.6]}, "frozen_base": True}
        self.scores = scores
        self.calls = []

    def load(self):
        return True

    def predict(self, images):
        self.calls.append(images)
        if any(not b.startswith(b"\xff\xd8") for b in images):
            raise OSError("cannot identify image file")
        ranked = sorted(zip(LABELS, self.scores), key=lambda p: p[1], reverse=True)
        return [{"id": i, "score": s} for i, s in ranked]


@pytest.fixture
def fake():
    return FakeClassifier([0.05, 0.05, 0.8, 0.05, 0.05])


@pytest.fixture
def client(tmp_path, monkeypatch, fake):
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "petals.db")
    monkeypatch.setattr(main, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(main, "classifier", fake)
    with TestClient(main.app) as c:
        yield c


def photos(*blobs, name="flower.jpg", type="image/jpeg"):
    return [("photos", (name, b, type)) for b in blobs]


def test_status(client):
    s = client.get("/api/status").json()
    assert s["ready"] is True
    assert s["labels"] == LABELS
    assert s["val_accuracy"] == 0.9
    assert s["confident_threshold"] == main.CONFIDENT


def test_status_without_model(client, fake):
    fake.model = None
    fake.meta = None
    s = client.get("/api/status").json()
    assert s["ready"] is False
    assert s["labels"] is None


def test_species_lists_all_labels(client):
    species = client.get("/api/species").json()
    assert sorted(species) == LABELS
    for sid, info in species.items():
        assert info["id"] == sid
        assert {"common", "scientific", "family", "description", "look_for", "flowering", "wiki"} <= info.keys()
        assert all(s.startswith(f"/samples/{sid}-") for s in info["samples"])


def test_index_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Petals" in r.text


def test_identify(client, fake):
    r = client.post("/api/identify", files=photos(jpeg(), jpeg()))
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0] == {"id": "rose", "score": 0.8}
    assert body["confident"] is True
    assert len(fake.calls[0]) == 2


def test_identify_low_confidence(client, fake):
    fake.scores = [0.3, 0.25, 0.2, 0.15, 0.1]
    body = client.post("/api/identify", files=photos(jpeg())).json()
    assert body["results"][0]["id"] == "daisy"
    assert body["confident"] is False


def test_identify_at_threshold_is_confident(client, fake):
    fake.scores = [main.CONFIDENT, 0.1, 0.1, 0.1, 0.1]
    assert client.post("/api/identify", files=photos(jpeg())).json()["confident"] is True


def test_identify_without_model(client, fake):
    fake.model = None
    r = client.post("/api/identify", files=photos(jpeg()))
    assert r.status_code == 503
    assert "train.py" in r.json()["detail"]


def test_identify_requires_photos(client):
    assert client.post("/api/identify").status_code == 422


def test_identify_too_many_photos(client):
    r = client.post("/api/identify", files=photos(*[jpeg()] * (main.MAX_PHOTOS + 1)))
    assert r.status_code == 400
    assert "up to 5" in r.json()["detail"]


def test_identify_rejects_non_images(client):
    r = client.post("/api/identify", files=photos(b"hello", name="notes.txt", type="text/plain"))
    assert r.status_code == 400
    assert "notes.txt is not an image." == r.json()["detail"]


def test_identify_unreadable_image(client):
    r = client.post("/api/identify", files=photos(b"garbage", name="broken.jpg"))
    assert r.status_code == 400
    assert "could not be read" in r.json()["detail"]


def test_identify_too_large(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_BYTES", 100)
    r = client.post("/api/identify", files=photos(jpeg(size=(300, 300))))
    assert r.status_code == 400
    assert "larger than 15 MB" in r.json()["detail"]


def test_observation_lifecycle(client, tmp_path):
    assert client.get("/api/observations").json() == []

    r = client.post("/api/observations", files=photos(jpeg(), jpeg()),
                    data={"species": "rose", "score": "0.8", "note": "  red, by the gate  ", "place": "Park"})
    assert r.status_code == 200
    obs_id = r.json()["id"]

    [obs] = client.get("/api/observations").json()
    assert obs["id"] == obs_id
    assert obs["species"] == "rose"
    assert obs["score"] == 0.8
    assert obs["note"] == "red, by the gate"
    assert obs["place"] == "Park"
    assert len(obs["photos"]) == 2
    for url in obs["photos"]:
        assert url.startswith("/uploads/") and url.endswith(".jpg")
        assert (tmp_path / "uploads" / url.removeprefix("/uploads/")).read_bytes() == jpeg()

    assert client.delete(f"/api/observations/{obs_id}").json() == {"deleted": obs_id}
    assert client.get("/api/observations").json() == []
    assert list((tmp_path / "uploads").iterdir()) == []


def test_observations_newest_first(client, monkeypatch):
    stamps = iter(["2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00"])

    class Clock:
        @staticmethod
        def now(tz):
            from datetime import datetime
            return datetime.fromisoformat(next(stamps))

    monkeypatch.setattr(main, "datetime", Clock)
    for sp in ("daisy", "tulip"):
        client.post("/api/observations", files=photos(jpeg()), data={"species": sp, "score": "0.9"})
    assert [o["species"] for o in client.get("/api/observations").json()] == ["tulip", "daisy"]


def test_observation_trims_long_fields(client):
    client.post("/api/observations", files=photos(jpeg()),
                data={"species": "daisy", "score": "0.7", "note": "n" * 600, "place": "p" * 200})
    [obs] = client.get("/api/observations").json()
    assert len(obs["note"]) == 500
    assert len(obs["place"]) == 120


def test_observation_unknown_extension_saved_as_jpg(client):
    client.post("/api/observations", files=photos(jpeg(), name="photo.exe"),
                data={"species": "daisy", "score": "0.7"})
    [obs] = client.get("/api/observations").json()
    assert obs["photos"][0].endswith("-0.jpg")


def test_observation_unknown_species(client):
    r = client.post("/api/observations", files=photos(jpeg()), data={"species": "cactus", "score": "0.9"})
    assert r.status_code == 400
    assert client.get("/api/observations").json() == []


def test_delete_missing_observation(client):
    assert client.delete("/api/observations/nope").status_code == 404
