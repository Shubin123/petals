"""Runs the GitHub Pages build in Chromium, served from a sub-path the way Pages serves it."""

import functools
import http.server
import json
import threading

import pytest

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
