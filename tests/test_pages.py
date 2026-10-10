"""Runs the GitHub Pages build in Chromium, served from a sub-path the way Pages serves it."""

import functools
import http.server
import io
import json
import threading
from urllib.parse import urlsplit

import pytest
from PIL import Image

from conftest import LABELS, ROOT

pytest.importorskip("playwright")

SITE = ROOT / "site"
SPECIES = json.loads((SITE / "species.json").read_text())
SAMPLES = [s for info in SPECIES.values() for s in info["samples"]]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def site_url(tmp_path_factory):
    root = tmp_path_factory.mktemp("pages")
    (root / "petals").symlink_to(SITE, target_is_directory=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                             functools.partial(QuietHandler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/petals/"
    server.shutdown()


@pytest.fixture(scope="module")
def site(browser, site_url):
    """One page for the module, so the 55 MB model is downloaded once."""
    context = browser.new_context(base_url=site_url)
    page = context.new_page()
    page.requests = []
    page.errors = []
    page.on("request", lambda r: page.requests.append(r.url))
    page.on("pageerror", lambda e: page.errors.append(str(e)))
    page.goto(site_url)
    yield page
    context.close()


@pytest.fixture
def page(site, site_url):
    site.goto(site_url + "#/")
    site.wait_for_selector("#drop")
    yield site
    assert not site.errors, site.errors
    assert not [u for u in site.requests if "/api/" in u], "static build called the server API"


def test_species_pages(page, site_url):
    page.goto(site_url + "#/species")
    page.wait_for_selector(".flora li")
    cards = page.locator(".flora li")
    assert cards.count() == len(LABELS)
    page.wait_for_function("[...document.querySelectorAll('.flora img')].every(i => i.complete && i.naturalWidth > 0)")

    page.click("text=Sunflower")
    page.wait_for_selector("h1:has-text('Sunflower')")
    assert page.locator(".species-detail .sci").inner_text() == "Helianthus annuus"
    assert page.locator(".gallery img").count() == len(SPECIES["sunflower"]["samples"])


def test_how_it_works_reads_metadata(page, site_url):
    meta = json.loads((SITE / "model" / "metadata.json").read_text())
    page.goto(site_url + "#/how")
    page.wait_for_selector(".stats")
    assert f"{round(meta['val_accuracy'] * 100)}%" in page.locator(".stats").inner_text()
    assert page.locator("#chart svg").count() == 1
    page.wait_for_selector("#aug-grid canvas")
    assert page.locator("#aug-grid canvas").count() == 8


def test_model_banner_hidden(page):
    page.wait_for_function("document.getElementById('model-banner').hidden")


def identify_in_page(page, urls):
    """Score sample photos with the in-browser model, using the same code path as the Identify button."""
    return page.evaluate("""async (urls) => {
        const { backend } = await import("./app.js");
        const out = [];
        for (const u of urls) {
            const blob = await (await fetch(u)).blob();
            out.push(await backend.identify([new File([blob], u.split("/").pop(), { type: blob.type })]));
        }
        return out;
    }""", urls)


def test_identifies_every_sample(page):
    results = identify_in_page(page, SAMPLES)
    wrong = []
    for url, r in zip(SAMPLES, results):
        scores = r["results"]
        assert sorted(s["id"] for s in scores) == LABELS
        assert sum(s["score"] for s in scores) == pytest.approx(1, abs=1e-4)
        assert scores == sorted(scores, key=lambda s: -s["score"])
        if scores[0]["id"] != url.split("/")[1].split("-")[0]:
            wrong.append((url, scores[0]))
    # Samples were picked because the trained model scores them >= 95%, so all should be right.
    assert not wrong


def test_matches_keras_model(page):
    """The quantized TF.js model should give nearly the same scores as the Keras model it came from."""
    if not (ROOT / "models" / "petals.keras").exists():
        pytest.skip("no trained model in models/")
    pytest.importorskip("keras")
    from app.model import Classifier

    classifier = Classifier()
    classifier.load()
    urls = SAMPLES[::4]
    browser_results = identify_in_page(page, urls)
    for url, r in zip(urls, browser_results):
        keras_scores = {s["id"]: s["score"] for s in classifier.predict([(SITE / url).read_bytes()])}
        for s in r["results"]:
            assert s["score"] == pytest.approx(keras_scores[s["id"]], abs=0.05), url


def test_identify_save_and_delete_observation(page, site_url):
    photo = SITE / SPECIES["daisy"]["samples"][0]
    page.set_input_files("#file", [photo, SITE / SPECIES["daisy"]["samples"][1]])
    assert page.locator("#thumbs li").count() == 2
    page.click("#go")
    page.wait_for_selector(".slip-name", timeout=120_000)
    assert page.locator(".slip-name").inner_text() == "Bellis perennis"
    assert page.locator(".slip-det").inner_text() == "Det."
    assert page.locator("#bars li").count() == len(LABELS)
    assert page.locator("#mini li").count() == 2

    page.fill("input[name=place]", "Test meadow")
    page.fill("textarea[name=note]", "  white rays  ")
    page.click("#save-form button[type=submit]")
    page.wait_for_selector("#save-status:has-text('Observation saved')")

    page.click("text=View my observations")
    page.wait_for_selector(".log li")
    entry = page.locator(".log li")
    assert entry.count() == 1
    assert "Daisy" in entry.inner_text()
    assert "Test meadow" in entry.inner_text()
    assert entry.locator(".note").inner_text() == "white rays"
    assert "this browser only" in page.locator(".page-head").inner_text()
    page.wait_for_function("document.querySelector('.log img').naturalWidth > 0")

    # Saved observations survive a reload.
    page.reload()
    page.wait_for_selector(".log li")

    page.once("dialog", lambda d: d.accept())
    entry.locator("button:has-text('Delete')").click()
    page.wait_for_selector(".empty-state")


def test_rejects_non_image(page):
    page.set_input_files("#file", files=[{"name": "notes.txt", "mimeType": "text/plain", "buffer": b"hi"}])
    assert page.locator("#error").inner_text() == "Only image files can be added."
    assert page.locator("#identify-actions").is_hidden()


def test_unreadable_image_shows_error(page):
    page.set_input_files("#file", files=[{"name": "broken.jpg", "mimeType": "image/jpeg", "buffer": b"junk"}])
    page.click("#go")
    page.wait_for_selector("#error:has-text('could not be read')", timeout=120_000)
    assert page.locator("#go").is_enabled()


@pytest.fixture(scope="module")
def camera_browser(playwright, tmp_path_factory):
    """Feed a real sample flower through Chromium's camera and the normal frame pipeline."""
    import numpy as np

    rgb = np.asarray(Image.open(SITE / SPECIES["daisy"]["samples"][0]).convert("RGB")
                     .resize((640, 360)), dtype=np.float32)
    r, g, b = rgb.transpose(2, 0, 1)
    y = 16 + 0.257 * r + 0.504 * g + 0.098 * b
    u = 128 - 0.148 * r - 0.291 * g + 0.439 * b
    v = 128 + 0.439 * r - 0.368 * g - 0.071 * b
    subsample = lambda plane: plane.reshape(180, 2, 320, 2).mean(axis=(1, 3))
    frame = tmp_path_factory.mktemp("camera") / "daisy.y4m"
    frame.write_bytes(b"YUV4MPEG2 W640 H360 F30:1 Ip A1:1 C420jpeg\nFRAME\n" + b"".join(
        plane.clip(0, 255).astype(np.uint8).tobytes() for plane in (y, subsample(u), subsample(v))))
    browser = playwright.chromium.launch(args=[
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-video-capture={frame}",
    ])
    yield browser
    browser.close()


@pytest.fixture
def camera_page(camera_browser, site_url):
    context = camera_browser.new_context(permissions=["camera"], base_url=site_url)
    page = context.new_page()
    errors, api_requests = [], []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: api_requests.append(r.url) if "/api/" in r.url else None)
    page.add_init_script("""(() => {
        window.cameraStreams = [];
        const getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
        navigator.mediaDevices.getUserMedia = async (constraints) => {
            const stream = await getUserMedia(constraints);
            window.cameraStreams.push(stream);
            return stream;
        };
    })();""")
    page.goto(site_url)
    page.wait_for_selector("#drop")
    yield page
    context.close()
    assert not errors, errors
    assert not api_requests, "static camera flow called the server API"


def defer_camera_predictions(page):
    page.evaluate("""async () => {
        const { backend } = await import('./app.js');
        window.pendingPredictions = [];
        window.activePredictions = 0;
        window.maxActivePredictions = 0;
        backend.identify = (photos) => {
            window.activePredictions++;
            window.maxActivePredictions = Math.max(window.maxActivePredictions, window.activePredictions);
            return new Promise((resolve, reject) => {
                window.pendingPredictions.push({ resolve, reject, photos });
            }).finally(() => { window.activePredictions--; });
        };
    }""")


def resolve_camera_prediction(page, index=0, confident=True):
    page.evaluate("""({ index, confident }) => {
        window.pendingPredictions[index].resolve({
            confident,
            results: [
                { id: 'daisy', score: confident ? 0.8 : 0.4 },
                ...['dandelion', 'rose', 'sunflower', 'tulip'].map(id => ({ id, score: confident ? 0.05 : 0.15 })),
            ],
        });
    }""", {"index": index, "confident": confident})


def test_live_camera_identifies_repeatedly_and_saves_frame(camera_page):
    page = camera_page
    file_choosers = []
    page.on("filechooser", lambda chooser: file_choosers.append(chooser))
    page.evaluate("""async () => {
        const { backend } = await import('./app.js');
        const identify = backend.identify.bind(backend);
        window.completedPredictions = 0;
        backend.identify = async (photos) => {
            const result = await identify(photos);
            window.completedPredictions++;
            return result;
        };
    }""")
    page.locator("#live-camera").check()
    page.wait_for_selector("#live-result:not([hidden])", timeout=120_000)
    assert page.locator("#live-name").inner_text() == "Daisy"
    assert page.locator("#live-determination").inner_text() == "Live match"
    page.wait_for_function("window.completedPredictions >= 2", timeout=120_000)
    assert page.evaluate("document.querySelector('#live-video').currentTime > 0")
    assert not file_choosers

    page.click("#live-use")
    page.wait_for_selector(".slip-name")
    assert page.locator(".slip-name").inner_text() == "Bellis perennis"
    page.wait_for_function("window.cameraStreams.every(s => s.getTracks().every(t => t.readyState === 'ended'))")
    page.wait_for_function("document.querySelector('#hero').naturalWidth > 0")
    page.fill("input[name=place]", "Live camera meadow")
    page.click("#save-form button[type=submit]")
    page.wait_for_selector("#save-status:has-text('Observation saved')")
    saved = page.evaluate("""async () => {
        const { backend } = await import('./app.js');
        const store = await backend.store('readonly');
        const rows = await new Promise(resolve => { const req = store.getAll(); req.onsuccess = () => resolve(req.result); });
        return rows.map(row => ({ name: row.photos[0].name, type: row.photos[0].type, size: row.photos[0].size }));
    }""")
    assert len(saved) == 1
    assert saved[0]["name"] == "live-camera.jpg"
    assert saved[0]["type"] == "image/jpeg"
    assert saved[0]["size"] > 0
    page.click("text=View my observations")
    page.wait_for_selector(".log li")
    assert "Daisy" in page.locator(".log li").inner_text()
    assert "Live camera meadow" in page.locator(".log li").inner_text()


def test_live_camera_toggle_keeps_photos_and_serializes_predictions(camera_page):
    page = camera_page
    page.set_input_files("#file", SITE / SPECIES["sunflower"]["samples"][0])
    defer_camera_predictions(page)
    page.clock.install()
    page.locator("#live-camera").check()
    page.wait_for_function("window.pendingPredictions.length === 1")
    assert page.locator("#drop").is_hidden()
    page.locator("#live-camera").uncheck()
    page.wait_for_function("window.cameraStreams[0].getTracks().every(t => t.readyState === 'ended')")
    assert page.locator("#live-panel").is_hidden()
    assert page.locator("#thumbs li").count() == 1
    assert page.locator("#identify-actions").is_visible()
    page.clock.fast_forward(4000)
    assert page.evaluate("window.pendingPredictions.length") == 1

    # Switching back on waits for the old request, without showing its stale result.
    page.locator("#live-camera").check()
    page.wait_for_function("window.cameraStreams.length === 2 && document.querySelector('#live-video').readyState >= 2")
    resolve_camera_prediction(page)
    page.wait_for_function("window.pendingPredictions.length === 2")
    assert page.locator("#live-result").is_hidden()
    resolve_camera_prediction(page, index=1, confident=False)
    page.wait_for_selector("#live-result:not([hidden])")
    assert page.locator("#live-determination").inner_text() == "Undetermined. Closest match:"
    assert page.locator("#live-score").inner_text() == "40%"
    assert page.evaluate("window.maxActivePredictions") == 1
    page.locator("#live-camera").uncheck()


@pytest.mark.parametrize("leave", ["navigate", "pagehide"])
def test_live_camera_releases_tracks_and_ignores_pending_results(camera_page, site_url, leave):
    page = camera_page
    defer_camera_predictions(page)
    page.clock.install()
    page.locator("#live-camera").check()
    page.wait_for_function("window.pendingPredictions.length === 1")
    if leave == "navigate":
        page.click("nav a[data-route=species]")
        page.wait_for_selector(".flora")
    else:
        page.evaluate("window.dispatchEvent(new PageTransitionEvent('pagehide'))")
    page.wait_for_function("window.cameraStreams[0].getTracks().every(t => t.readyState === 'ended')")
    resolve_camera_prediction(page)
    page.clock.fast_forward(4000)
    assert page.evaluate("window.pendingPredictions.length") == 1
    assert not page.url.endswith("#/result")
    if leave == "navigate":
        page.goto(site_url)
        page.wait_for_selector("#drop")
        assert not page.locator("#live-camera").is_checked()
    else:
        assert page.locator("#live-panel").is_hidden()


def test_camera_stream_requested_before_switch_off_is_released(camera_page):
    page = camera_page
    defer_camera_predictions(page)
    page.evaluate("""() => {
        const getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
        navigator.mediaDevices.getUserMedia = async (constraints) => {
            const stream = await getUserMedia(constraints);
            return new Promise(resolve => { window.releaseCamera = () => resolve(stream); });
        };
    }""")
    page.locator("#live-camera").check()
    page.wait_for_function("typeof window.releaseCamera === 'function'")
    page.locator("#live-camera").uncheck()
    page.evaluate("window.releaseCamera()")
    page.wait_for_function("window.cameraStreams[0].getTracks().every(t => t.readyState === 'ended')")
    assert page.locator("#live-panel").is_hidden()
    assert page.evaluate("window.pendingPredictions.length") == 0


@pytest.mark.parametrize("name, message", [
    ("NotAllowedError", "Camera access was blocked"),
    ("NotFoundError", "No camera was found"),
    ("unsupported", "camera support"),
])
def test_camera_access_errors_restore_photo_selection(camera_page, name, message):
    page = camera_page
    page.evaluate("""name => {
        if (name === 'unsupported') {
            Object.defineProperty(navigator, 'mediaDevices', { value: undefined });
            return;
        }
        navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('Camera unavailable', name); };
    }""", name)
    page.click("label[for=live-camera]")
    page.wait_for_selector(f"#error:has-text('{message}')")
    assert not page.locator("#live-camera").is_checked()
    assert page.locator("#live-panel").is_hidden()
    assert page.locator("#drop").is_visible()
    page.set_input_files("#file", SITE / SPECIES["daisy"]["samples"][0])
    assert page.locator("#thumbs li").count() == 1


def test_live_camera_retries_failed_identification(camera_page):
    page = camera_page
    defer_camera_predictions(page)
    page.clock.install()
    page.locator("#live-camera").check()
    page.wait_for_function("window.pendingPredictions.length === 1")
    page.evaluate("window.pendingPredictions[0].reject(new Error('Temporary identification failure'))")
    page.wait_for_selector("#error:has-text('Temporary identification failure')")
    assert page.locator("#live-camera").is_checked()
    assert page.locator("#live-use").is_disabled()
    page.clock.fast_forward(1600)
    page.wait_for_function("window.pendingPredictions.length === 2")
    resolve_camera_prediction(page, index=1)
    page.wait_for_selector("#live-result:not([hidden])")
    assert page.locator("#error").inner_text() == ""
    assert page.locator("#live-name").inner_text() == "Daisy"
    page.locator("#live-camera").uncheck()


def test_live_camera_server_identification_and_observation(camera_browser, site_url, tmp_path, monkeypatch):
    """Send the browser's camera JPEGs through the real FastAPI upload and save handlers."""
    from fastapi.testclient import TestClient

    from app import main
    from test_api import FakeClassifier

    classifier = FakeClassifier([0.05, 0.05, 0.8, 0.05, 0.05])
    monkeypatch.setattr(main, "classifier", classifier)
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "petals.db")
    monkeypatch.setattr(main, "UPLOADS", tmp_path / "uploads")
    context = camera_browser.new_context(permissions=["camera"], base_url=site_url)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        with TestClient(main.app) as client:
            def api_route(route):
                req = route.request
                response = client.request(req.method, urlsplit(req.url).path,
                                          content=req.post_data_buffer,
                                          headers={"content-type": req.headers.get("content-type", "")})
                route.fulfill(status=response.status_code, body=response.content,
                              content_type=response.headers.get("content-type", "application/json"))

            page.route("**/api/**", api_route)
            page.route(site_url, lambda route: route.fulfill(
                body=(ROOT / "app/static/index.html").read_text(), content_type="text/html"))
            page.goto(site_url)
            page.wait_for_selector("#drop")
            with page.expect_response("**/api/identify"):
                page.locator("#live-camera").check()
            page.wait_for_selector("#live-result:not([hidden])")
            assert page.locator("#live-name").inner_text() == "Rose"
            with page.expect_response("**/api/identify"):
                pass
            assert len(classifier.calls) >= 2
            for call in classifier.calls:
                assert len(call) == 1
                image = Image.open(io.BytesIO(call[0]))
                assert image.format == "JPEG"
                assert max(image.size) <= 640
            page.evaluate("window.cameraStream = document.querySelector('#live-video').srcObject")
            page.click("#live-use")
            page.wait_for_selector(".slip-name")
            assert page.locator(".slip-name").inner_text() == SPECIES["rose"]["scientific"]
            assert page.evaluate("window.cameraStream.getTracks().every(t => t.readyState === 'ended')")
            page.fill("input[name=place]", "Server camera meadow")
            page.click("#save-form button[type=submit]")
            page.wait_for_selector("#save-status:has-text('Observation saved')")
            rows = client.get("/api/observations").json()
            assert len(rows) == 1
            assert rows[0]["species"] == "rose"
            assert rows[0]["place"] == "Server camera meadow"
            assert len(rows[0]["photos"]) == 1
            saved = (main.UPLOADS / rows[0]["photos"][0].rsplit("/", 1)[1]).read_bytes()
            assert Image.open(io.BytesIO(saved)).format == "JPEG"
    finally:
        context.close()
    assert not errors, errors
