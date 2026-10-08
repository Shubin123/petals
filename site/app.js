const view = document.getElementById("view");
const MAX_PHOTOS = 5;

const state = {
  photos: [],      // File objects chosen on the Identify page
  result: null,    // { results, confident, photos, urls }
  species: null,
  status: null,
};

const api = async (path, opts) => {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
};

/* ---------- Backends ---------- */

// Talks to the FastAPI server in app/main.py.
const server = {
  species: () => api("/api/species"),
  status: () => api("/api/status"),
  identify: (photos) => {
    const form = new FormData();
    photos.forEach((f) => form.append("photos", f));
    return api("/api/identify", { method: "POST", body: form });
  },
  observations: () => api("/api/observations"),
  saveObservation: (photos, fields) => {
    const fd = new FormData();
    photos.forEach((f) => fd.append("photos", f));
    for (const [k, v] of Object.entries(fields)) fd.append(k, v);
    return api("/api/observations", { method: "POST", body: fd });
  },
  deleteObservation: (id) => api(`/api/observations/${id}`, { method: "DELETE" }),
};

// The GitHub Pages build (training/build_pages.py): the model runs in the browser with
// TensorFlow.js and observations live in IndexedDB.
const CONFIDENT = 0.6;
const TFJS = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";

const browser = {
  meta: null,
  model: null,

  async species() {
    return api("species.json");
  },

  async status() {
    this.meta ??= await api("model/metadata.json");
    return { ...this.meta, ready: true, confident_threshold: CONFIDENT };
  },

  async loadModel() {
    if (!window.tf) {
      await new Promise((resolve, reject) => {
        const s = Object.assign(document.createElement("script"), { src: TFJS, onload: resolve });
        s.onerror = () => reject(new Error("Couldn't load TensorFlow.js. Check your connection."));
        document.head.append(s);
      });
    }
    await this.status();
    this.model ??= await tf.loadGraphModel("model/model.json");
    return this.model;
  },

  // Same steps as Classifier._prepare in app/model.py: apply EXIF rotation, convert to RGB,
  // squash to the model's input size with bilinear resampling, keep 0–255 values.
  async pixels(file) {
    const [w, h] = this.meta.image_size;
    let bitmap;
    try {
      bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    } catch {
      throw new Error(`${file.name} could not be read. Try a JPEG or PNG.`);
    }
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingQuality = "medium";
    ctx.drawImage(bitmap, 0, 0, w, h);
    bitmap.close();
    return ctx.getImageData(0, 0, w, h);
  },

  async identify(photos) {
    const model = await this.loadModel();
    const images = await Promise.all(photos.map((f) => this.pixels(f)));
    const probs = tf.tidy(() => {
      const batch = tf.stack(images.map((img) => tf.browser.fromPixels(img, 3).toFloat()));
      return model.predict(batch).mean(0).arraySync();
    });
    return rank(this.meta.labels, probs);
  },

  db: null,
  async store(mode) {
    this.db ??= await new Promise((resolve, reject) => {
      const req = indexedDB.open("petals", 1);
      req.onupgradeneeded = () => req.result.createObjectStore("observations", { keyPath: "id" });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    return this.db.transaction("observations", mode).objectStore("observations");
  },

  async observations() {
    const store = await this.store("readonly");
    const rows = await done(store.getAll());
    return rows
      .sort((a, b) => b.created.localeCompare(a.created))
      .map((o) => ({ ...o, photos: o.photos.map((blob) => URL.createObjectURL(blob)) }));
  },

  async saveObservation(photos, { species, score, note = "", place = "" }) {
    const id = crypto.randomUUID().replaceAll("-", "");
    const row = {
      id, species, score: Number(score), photos: [...photos],
      note: note.trim().slice(0, 500), place: place.trim().slice(0, 120),
      created: new Date().toISOString(),
    };
    await done((await this.store("readwrite")).add(row));
    return { id };
  },

  async deleteObservation(id) {
    await done((await this.store("readwrite")).delete(id));
    return { deleted: id };
  },
};

const done = (req) => new Promise((resolve, reject) => {
  req.onsuccess = () => resolve(req.result);
  req.onerror = () => reject(req.error);
});

function rank(labels, probs) {
  const results = labels.map((id, i) => ({ id, score: probs[i] })).sort((a, b) => b.score - a.score);
  return { results, confident: results[0].score >= CONFIDENT };
}

const backend = document.documentElement.dataset.mode === "static" ? browser : server;

const el = (html) => {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
};

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => `${Math.round(x * 100)}%`;
const today = () => new Date().toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });

async function loadSpecies() {
  state.species ??= await backend.species();
  return state.species;
}

/* ---------- Identify ---------- */

function renderIdentify() {
  view.replaceChildren(document.getElementById("tpl-identify").content.cloneNode(true));
  const drop = view.querySelector("#drop");
  const file = view.querySelector("#file");
  const camera = view.querySelector("#camera");
  const thumbs = view.querySelector("#thumbs");
  const actions = view.querySelector("#identify-actions");
  const error = view.querySelector("#error");
  const go = view.querySelector("#go");

  const refresh = () => {
    thumbs.replaceChildren(...state.photos.map((f, i) => {
      const li = el(`<li><img alt="Photo ${i + 1}"><button class="remove" type="button" aria-label="Remove photo ${i + 1}">×</button></li>`);
      li.querySelector("img").src = URL.createObjectURL(f);
      li.querySelector("button").onclick = () => { state.photos.splice(i, 1); refresh(); };
      return li;
    }));
    drop.classList.toggle("has-photos", state.photos.length > 0);
    actions.hidden = state.photos.length === 0;
    view.querySelector("#add-more").hidden = state.photos.length >= MAX_PHOTOS;
  };

  const add = (files) => {
    error.textContent = "";
    const images = [...files].filter((f) => f.type.startsWith("image/"));
    if (images.length < files.length) error.textContent = "Only image files can be added.";
    const room = MAX_PHOTOS - state.photos.length;
    if (images.length > room) error.textContent = `Use up to ${MAX_PHOTOS} photos of the same plant.`;
    state.photos.push(...images.slice(0, room));
    refresh();
  };

  view.querySelectorAll("[data-pick]").forEach((b) => (b.onclick = () => (b.dataset.pick === "camera" ? camera : file).click()));
  view.querySelector("#add-more").onclick = () => file.click();
  view.querySelector("#clear").onclick = () => { state.photos = []; error.textContent = ""; refresh(); };
  file.onchange = camera.onchange = (e) => { add(e.target.files); e.target.value = ""; };

  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); add(e.dataTransfer.files); });
  document.onpaste = (e) => { if (location.hash === "#/" || location.hash === "") add(e.clipboardData.files); };

  go.onclick = async () => {
    error.textContent = "";
    go.disabled = true;
    go.textContent = "Identifying…";
    try {
      const data = await backend.identify(state.photos);
      state.result = { ...data, photos: [...state.photos], urls: state.photos.map((f) => URL.createObjectURL(f)) };
      location.hash = "#/result";
    } catch (err) {
      error.textContent = err.message;
      go.disabled = false;
      go.textContent = "Identify";
    }
  };

  refresh();
}

/* ---------- Result ---------- */

async function renderResult() {
  if (!state.result) { location.hash = "#/"; return; }
  const species = await loadSpecies();
  const { results, confident, urls, photos } = state.result;
  const top = results[0];
  const sp = species[top.id];

  view.replaceChildren(document.getElementById("tpl-result").content.cloneNode(true));
  const hero = view.querySelector("#hero");
  hero.src = urls[0];
  hero.alt = `Your photo, identified as ${sp.common}`;

  const mini = view.querySelector("#mini");
  if (urls.length > 1) {
    mini.replaceChildren(...urls.map((u, i) => {
      const li = el(`<li><button type="button" aria-pressed="${i === 0}" aria-label="Show photo ${i + 1}"><img alt=""></button></li>`);
      li.querySelector("img").src = u;
      li.querySelector("button").onclick = (e) => {
        hero.src = u;
        mini.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b === e.currentTarget));
      };
      return li;
    }));
  }

  view.querySelector("#slip").innerHTML = `
    <div class="slip-head"><span>Petals determination</span><span>${today()}</span></div>
    <div>
      <p class="slip-det">${confident ? "Det." : "Undetermined. Closest match:"}</p>
      <p class="slip-name">${esc(sp.scientific)}</p>
      <p class="slip-common">${esc(sp.common)}</p>
    </div>
    <div class="meter">
      <div class="meter-label"><span>Confidence</span><strong>${pct(top.score)}</strong></div>
      <div class="meter-track"><div class="meter-fill" style="width:${top.score * 100}%"></div></div>
    </div>
    <dl>
      <dt>Family</dt><dd>${esc(sp.family)}</dd>
      <dt>Flowering</dt><dd>${esc(sp.flowering)}</dd>
      <dt>Photos</dt><dd>${photos.length}</dd>
    </dl>
    ${confident ? "" : `<p class="caveat">The model isn't sure. Try a closer, well-lit photo of the flower head, or add more photos. If the plant isn't a daisy, dandelion, rose, sunflower or tulip, Petals can't name it.</p>`}
    <div class="slip-foot"><a href="#/species/${top.id}">About ${esc(sp.common.toLowerCase())}s</a><a href="${sp.wiki}" target="_blank" rel="noopener">Wikipedia</a></div>`;

  view.querySelector("#bars").replaceChildren(...results.map((r) => {
    const s = species[r.id];
    return el(`<li>
      <a href="#/species/${r.id}">${esc(s.common)}<span class="sci">${esc(s.scientific)}</span></a>
      <div class="track" role="img" aria-label="${esc(s.common)} ${pct(r.score)}"><div class="fill" style="width:${r.score * 100}%"></div></div>
      <span class="pct">${pct(r.score)}</span>
    </li>`);
  }));

  const formEl = view.querySelector("#save-form");
  const status = view.querySelector("#save-status");
  formEl.onsubmit = async (e) => {
    e.preventDefault();
    const btn = formEl.querySelector("button[type=submit]");
    btn.disabled = true;
    status.textContent = "";
    const fields = { ...Object.fromEntries(new FormData(formEl)), species: top.id, score: top.score };
    try {
      await backend.saveObservation(photos, fields);
      status.innerHTML = `Observation saved. <a href="#/observations">View my observations</a>`;
      btn.textContent = "Saved";
      state.photos = [];
    } catch (err) {
      status.textContent = err.message;
      btn.disabled = false;
    }
  };
}

/* ---------- Species ---------- */

const placeholder = `<div class="placeholder"><svg viewBox="0 0 32 32" aria-hidden="true"><use href="#petal-mark"/></svg></div>`;

async function renderSpeciesList() {
  const species = await loadSpecies();
  view.replaceChildren(el(`<section>
    <div class="page-head">
      <h1>Species</h1>
      <p>The five flowers Petals has been trained to recognise.</p>
    </div>
    <ul class="flora">${Object.values(species).map((s) => `
      <li><a href="#/species/${s.id}">
        <div class="plate">${s.samples[0] ? `<img src="${s.samples[0]}" alt="${esc(s.common)}" loading="lazy">` : placeholder}</div>
        <h3>${esc(s.common)}</h3>
        <span class="sci">${esc(s.scientific)} · ${esc(s.family)}</span>
      </a></li>`).join("")}
    </ul>
  </section>`));
}

async function renderSpecies(id) {
  const species = await loadSpecies();
  const s = species[id];
  if (!s) { renderNotFound(); return; }
  view.replaceChildren(el(`<section>
    <a class="back" href="#/species">All species</a>
    <div class="species-detail">
      <div class="text">
        <h1>${esc(s.common)}</h1>
        <p class="sci">${esc(s.scientific)}</p>
        <p class="lede">${esc(s.description)}</p>
        <div>
          <h3>What to look for</h3>
          <ul class="look">${s.look_for.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
        </div>
        <dl>
          <dt>Family</dt><dd>${esc(s.family)}</dd>
          <dt>Flowering</dt><dd>${esc(s.flowering)}</dd>
          <dt>Includes</dt><dd>${esc(s.also)}</dd>
        </dl>
        <p><a href="${s.wiki}" target="_blank" rel="noopener">Read more on Wikipedia</a></p>
      </div>
      <div class="sheet gallery">${s.samples.length
        ? s.samples.map((u) => `<img src="${u}" alt="${esc(s.common)} from the training set" loading="lazy">`).join("")
        : `<p class="empty">Sample photos appear here after the model is trained.</p>`}
      </div>
    </div>
  </section>`));
}

/* ---------- Observations ---------- */

async function renderObservations() {
  const [species, obs] = await Promise.all([loadSpecies(), backend.observations()]);
  const section = el(`<section>
    <div class="page-head">
      <h1>My observations</h1>
      <p>Flowers you've identified and saved.${backend === browser ? " They're kept in this browser only." : ""}</p>
    </div>
    <ul class="log"></ul>
  </section>`);
  const log = section.querySelector(".log");
  if (!obs.length) {
    log.replaceWith(el(`<div class="empty-state">
      <p>No observations yet. Identify a flower and save it to start your log.</p>
      <a class="btn primary" href="#/">Identify a flower</a>
    </div>`));
  }
  for (const o of obs) {
    const s = species[o.species];
    const when = new Date(o.created).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
    const li = el(`<li>
      <img src="${o.photos[0]}" alt="${esc(s.common)}">
      <div class="what">
        <h3><a href="#/species/${o.species}">${esc(s.common)}</a></h3>
        <span class="sci">${esc(s.scientific)}</span>
        <span class="meta">${when}${o.place ? ` · ${esc(o.place)}` : ""} · ${pct(o.score)} confidence${o.photos.length > 1 ? ` · ${o.photos.length} photos` : ""}</span>
        ${o.note ? `<p class="note">${esc(o.note)}</p>` : ""}
      </div>
      <button class="btn quiet" type="button">Delete</button>
    </li>`);
    li.querySelector("button").onclick = async () => {
      if (!confirm(`Delete this ${s.common.toLowerCase()} observation?`)) return;
      await backend.deleteObservation(o.id);
      renderObservations();
    };
    log.append(li);
  }
  view.replaceChildren(section);
}

/* ---------- How it works ---------- */

async function renderHow() {
  const [species, status] = await Promise.all([loadSpecies(), backend.status()]);
  const sample = Object.values(species).flatMap((s) => s.samples)[0];
  const section = el(`<div class="how">
    <section>
      <div class="page-head">
        <h1>How it works</h1>
        <p>Petals is a convolutional neural network that looks at your photo and scores how much it resembles each of five flowers.</p>
      </div>
      ${status.ready ? `<div class="stats">
        <div><strong>${pct(status.val_accuracy)}</strong><span>accuracy on held-out photos</span></div>
        <div><strong>${status.train_images?.toLocaleString() ?? "–"}</strong><span>training photos</span></div>
        <div><strong>${status.val_images?.toLocaleString() ?? "–"}</strong><span>validation photos</span></div>
      </div>` : `<p class="loading">No model has been trained yet.</p>`}
    </section>

    <section>
      <h2>The model</h2>
      <ol class="steps">
        <li>Your photo is resized to 180 × 180 pixels.</li>
        <li>An <strong>InceptionResNetV2</strong> network, pretrained on ImageNet's 1.2 million photos, turns it into a set of visual features: petal shapes, colours, textures.</li>
        <li>A small classification layer turns those features into a score for each of the five flowers.</li>
        <li>With several photos of the same plant, the scores are averaged.</li>
      </ol>
      <div class="prose">
        <p>${status.frozen_base
          ? "The pretrained network was kept as it is, and only the classification layer was trained on the TensorFlow Flowers dataset."
          : "The whole network was fine-tuned on the TensorFlow Flowers dataset."} The learning rate warms up for eight epochs, then decays exponentially. Training stops early when validation loss stops improving. This follows the approach in <a href="https://www.kaggle.com/code/nikhilmishra21/flowers-notebook-cnn" target="_blank" rel="noopener">Flowers Notebook – CNN</a>.</p>
      </div>
    </section>

    <section>
      <h2>Data augmentation</h2>
      <div class="prose">
        <p>A few thousand photos isn't much for a network this size. To stop it memorising them, every training photo is randomly altered each time the model sees it, so it learns what makes a tulip a tulip rather than what one particular photo looks like. The three transformations come from Kaggle Learn's <a href="https://www.kaggle.com/code/ryanholbrook/exercise-data-augmentation" target="_blank" rel="noopener">Data Augmentation exercise</a>:</p>
        <p><strong>Contrast</strong> varies by up to ±10%. <strong>Horizontal flip</strong> happens half the time, because a flower facing left is still the same flower. <strong>Rotation</strong> goes up to ±36°, since stems lean in every direction. Vertical flips are left out: flowers in photos almost always face up.</p>
        <p>Below are eight versions of one photo, made with the same transformations used in training. ${state.photos[0] ? "It's the photo you chose on the Identify page." : "Choose your own photo to try it."}</p>
      </div>
      <div class="aug-controls">
        <label><input type="checkbox" data-aug="contrast" checked> Contrast</label>
        <label><input type="checkbox" data-aug="flip" checked> Flip</label>
        <label><input type="checkbox" data-aug="rotate" checked> Rotation</label>
        <button class="btn" type="button" id="reroll">Shuffle again</button>
        <button class="btn quiet" type="button" id="aug-pick">Use my own photo</button>
        <input type="file" id="aug-file" accept="image/*" hidden>
      </div>
      <div class="sheet aug-grid" id="aug-grid"></div>
    </section>

    ${status.ready && status.history ? `<section>
      <h2>Training progress</h2>
      <div class="prose"><p>Accuracy after each pass through the training photos. The validation line is measured on photos the model never trains on, so it shows how well Petals handles new pictures.</p></div>
      <div class="sheet chart-wrap" id="chart"></div>
    </section>` : ""}
  </div>`);
  view.replaceChildren(section);

  setupAugmentation(section, state.photos[0] ?? sample);
  if (status.ready && status.history) drawChart(section.querySelector("#chart"), status.history);
}

/* Faithful re-implementations of the Keras layers used in training */
const SIZE = 180;

function randomContrast(img, factor = 0.1) {
  const f = 1 + (Math.random() * 2 - 1) * factor;
  const d = img.data;
  const mean = [0, 0, 0];
  for (let i = 0; i < d.length; i += 4) for (let c = 0; c < 3; c++) mean[c] += d[i + c];
  const n = d.length / 4;
  for (let c = 0; c < 3; c++) mean[c] /= n;
  for (let i = 0; i < d.length; i += 4) for (let c = 0; c < 3; c++) d[i + c] = (d[i + c] - mean[c]) * f + mean[c];
  return img;
}

function randomFlip(img) {
  if (Math.random() < 0.5) return img;
  const out = new ImageData(SIZE, SIZE);
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    const s = (y * SIZE + (SIZE - 1 - x)) * 4, t = (y * SIZE + x) * 4;
    for (let c = 0; c < 4; c++) out.data[t + c] = img.data[s + c];
  }
  return out;
}

// Keras RandomRotation: factor is a fraction of 2π, empty corners filled by reflecting the image.
function randomRotation(img, factor = 0.1) {
  const theta = (Math.random() * 2 - 1) * factor * 2 * Math.PI;
  const cos = Math.cos(theta), sin = Math.sin(theta), c0 = (SIZE - 1) / 2;
  const reflect = (v) => {
    const period = 2 * SIZE;
    v = ((v % period) + period) % period;
    return v < SIZE ? v : period - 1 - v;
  };
  const out = new ImageData(SIZE, SIZE);
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    const dx = x - c0, dy = y - c0;
    const sx = reflect(Math.round(cos * dx + sin * dy + c0));
    const sy = reflect(Math.round(-sin * dx + cos * dy + c0));
    const s = (sy * SIZE + sx) * 4, t = (y * SIZE + x) * 4;
    for (let c = 0; c < 4; c++) out.data[t + c] = img.data[s + c];
  }
  return out;
}

function setupAugmentation(root, source) {
  const grid = root.querySelector("#aug-grid");
  const fileInput = root.querySelector("#aug-file");
  let base = null;

  const load = (src) => {
    if (!src) { grid.innerHTML = `<p class="loading" style="grid-column:1/-1">Choose a photo to see augmentation in action.</p>`; return; }
    const img = new Image();
    img.onload = () => {
      const c = new OffscreenCanvas(SIZE, SIZE);
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0, SIZE, SIZE); // squashed to 180×180, as in training
      base = ctx.getImageData(0, 0, SIZE, SIZE);
      draw();
    };
    img.src = src instanceof File ? URL.createObjectURL(src) : src;
  };

  const draw = () => {
    if (!base) return;
    const on = Object.fromEntries([...root.querySelectorAll("[data-aug]")].map((i) => [i.dataset.aug, i.checked]));
    grid.replaceChildren(...Array.from({ length: 8 }, (_, i) => {
      let img = new ImageData(new Uint8ClampedArray(base.data), SIZE, SIZE);
      if (i > 0) {
        if (on.contrast) img = randomContrast(img);
        if (on.flip) img = randomFlip(img);
        if (on.rotate) img = randomRotation(img);
      }
      const cv = document.createElement("canvas");
      cv.width = cv.height = SIZE;
      cv.getContext("2d").putImageData(img, 0, 0);
      cv.setAttribute("role", "img");
      cv.setAttribute("aria-label", i === 0 ? "Original photo" : `Augmented version ${i}`);
      if (i === 0) cv.className = "original";
      return cv;
    }));
  };

  root.querySelectorAll("[data-aug]").forEach((i) => (i.onchange = draw));
  root.querySelector("#reroll").onclick = draw;
  root.querySelector("#aug-pick").onclick = () => fileInput.click();
  fileInput.onchange = () => fileInput.files[0] && load(fileInput.files[0]);
  load(source);
}

function drawChart(wrap, history) {
  const train = history.accuracy, val = history.val_accuracy;
  const n = train.length;
  const W = 640, H = 280, m = { t: 16, r: 92, b: 36, l: 44 };
  const lo = Math.max(0, Math.floor(Math.min(...train, ...val) * 10) / 10);
  const x = (i) => m.l + (n === 1 ? 0 : (i / (n - 1)) * (W - m.l - m.r));
  const y = (v) => m.t + (1 - (v - lo) / (1 - lo)) * (H - m.t - m.b);
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const ticks = [];
  for (let v = lo; v <= 1.0001; v += 0.1) ticks.push(+v.toFixed(1));
  const xticks = [...new Set([0, Math.floor((n - 1) / 2), n - 1])];

  wrap.innerHTML = `
    <div class="chart-legend" aria-hidden="true">
      <span><i style="background:var(--series-train)"></i>Training photos</span>
      <span><i class="dashed" style="border-color:var(--series-val)"></i>Validation photos</span>
    </div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy per epoch. Final validation accuracy ${pct(val[n - 1])}.">
      ${ticks.map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--rule)" stroke-width="1"/>
        <text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end" font-size="12" fill="var(--muted)">${Math.round(v * 100)}%</text>`).join("")}
      ${xticks.map((i) => `<text x="${x(i)}" y="${H - 12}" text-anchor="middle" font-size="12" fill="var(--muted)">Epoch ${i + 1}</text>`).join("")}
      <path d="${path(train)}" fill="none" stroke="var(--series-train)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
      <path d="${path(val)}" fill="none" stroke="var(--series-val)" stroke-width="2" stroke-dasharray="6 4" stroke-linejoin="round" stroke-linecap="round"/>
      <text x="${x(n - 1) + 8}" y="${y(train[n - 1]) + (train[n - 1] >= val[n - 1] ? -4 : 12)}" font-size="12" fill="var(--ink-2)">Training ${pct(train[n - 1])}</text>
      <text x="${x(n - 1) + 8}" y="${y(val[n - 1]) + (train[n - 1] >= val[n - 1] ? 12 : -4)}" font-size="12" fill="var(--ink-2)">Validation ${pct(val[n - 1])}</text>
      <g id="hover" visibility="hidden">
        <line id="hx" y1="${m.t}" y2="${H - m.b}" stroke="var(--muted)" stroke-width="1"/>
        <circle id="ht" r="4.5" fill="var(--series-train)" stroke="var(--sheet)" stroke-width="2"/>
        <circle id="hv" r="4.5" fill="var(--series-val)" stroke="var(--sheet)" stroke-width="2"/>
      </g>
      <rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent" id="hit"/>
    </svg>
    <div class="chart-tip" hidden></div>
    <details class="chart-table"><summary>Show as table</summary>
      <table><thead><tr><th>Epoch</th><th>Training</th><th>Validation</th></tr></thead>
      <tbody>${train.map((v, i) => `<tr><td>${i + 1}</td><td>${(v * 100).toFixed(1)}%</td><td>${(val[i] * 100).toFixed(1)}%</td></tr>`).join("")}</tbody></table>
    </details>`;

  const svg = wrap.querySelector("svg"), tip = wrap.querySelector(".chart-tip"), hover = svg.querySelector("#hover");
  svg.querySelector("#hit").addEventListener("pointermove", (e) => {
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const i = Math.max(0, Math.min(n - 1, Math.round(((pt.x - m.l) / (W - m.l - m.r)) * (n - 1))));
    hover.setAttribute("visibility", "visible");
    svg.querySelector("#hx").setAttribute("x1", x(i));
    svg.querySelector("#hx").setAttribute("x2", x(i));
    svg.querySelector("#ht").setAttribute("cx", x(i)); svg.querySelector("#ht").setAttribute("cy", y(train[i]));
    svg.querySelector("#hv").setAttribute("cx", x(i)); svg.querySelector("#hv").setAttribute("cy", y(val[i]));
    const p = new DOMPoint(x(i), y(Math.max(train[i], val[i]))).matrixTransform(svg.getScreenCTM());
    const box = wrap.getBoundingClientRect();
    tip.hidden = false;
    tip.style.left = `${p.x - box.left}px`;
    tip.style.top = `${p.y - box.top}px`;
    tip.innerHTML = `Epoch ${i + 1}<br>Training ${(train[i] * 100).toFixed(1)}%<br>Validation ${(val[i] * 100).toFixed(1)}%`;
  });
  svg.querySelector("#hit").addEventListener("pointerleave", () => { hover.setAttribute("visibility", "hidden"); tip.hidden = true; });
}

/* ---------- Router ---------- */

function renderNotFound() {
  view.replaceChildren(el(`<div class="empty-state"><h1>Page not found</h1><a class="btn primary" href="#/">Identify a flower</a></div>`));
}

async function route() {
  const [, page = "", arg] = location.hash.split("/");
  const nav = { "": "identify", result: "identify", species: "species", observations: "observations", how: "how" }[page];
  document.querySelectorAll("nav a").forEach((a) =>
    a.dataset.route === nav ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
  try {
    if (page === "") renderIdentify();
    else if (page === "result") await renderResult();
    else if (page === "species") await (arg ? renderSpecies(arg) : renderSpeciesList());
    else if (page === "observations") await renderObservations();
    else if (page === "how") await renderHow();
    else renderNotFound();
  } catch (err) {
    view.replaceChildren(el(`<div class="empty-state"><p>${esc(err.message)}</p><a class="btn" href="#/">Back to Identify</a></div>`));
  }
  view.focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

window.addEventListener("hashchange", route);
route();

// Exported for the browser tests in tests/test_pages.py.
export { backend, rank };

backend.status().then((s) => {
  state.status = s;
  document.getElementById("model-banner").hidden = s.ready;
}).catch(() => {});
