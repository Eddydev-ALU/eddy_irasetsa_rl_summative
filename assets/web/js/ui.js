// Console panels: classifier bars, quality meter, guidance glyph, capture tray, reward timeline.
const $ = (id) => document.getElementById(id);
const PLANES = ["HC", "AC", "FL"];
const PLANE_NAMES = { HC: "Head circumference", AC: "Abdominal circumference", FL: "Femur length" };
const css = getComputedStyle(document.documentElement);
const C = {
  text: css.getPropertyValue("--text").trim(), text2: css.getPropertyValue("--text-2").trim(),
  text3: css.getPropertyValue("--text-3").trim(), line: css.getPropertyValue("--line").trim(),
  amber: css.getPropertyValue("--amber").trim(), green: css.getPropertyValue("--green").trim(),
  red: css.getPropertyValue("--red").trim(),
};

// Line-icon guidance glyphs (as a phone app would show them). 0..8 = env action ids.
const ARROW = (rot) => `<g transform="rotate(${rot} 50 50)"><line x1="50" y1="78" x2="50" y2="24"/><polyline points="32,42 50,24 68,42"/></g>`;
const ROT = (cw) => `<g transform="${cw ? "" : "translate(100 0) scale(-1 1)"}"><path d="M 74 38 A 28 28 0 1 0 78 58"/><polyline points="62,36 75,37 76,24"/></g>`;
const TILT = (fwd) => `<g transform="${fwd ? "" : "translate(0 100) scale(1 -1)"}">
  <line x1="50" y1="84" x2="50" y2="54" class="ghost"/><line x1="50" y1="84" x2="64" y2="56"/>
  <path d="M 30 40 A 30 30 0 0 1 70 40" /><polyline points="60,30 71,40 59,46"/></g>`;
const GLYPHS = [
  ARROW(0), ARROW(180), ARROW(-90), ARROW(90), ROT(true), ROT(false), TILT(true), TILT(false),
  `<rect x="24" y="24" width="52" height="52" rx="2"/><circle cx="50" cy="50" r="12"/><line x1="24" y1="36" x2="36" y2="24" class="ghost"/>`,
];

export class UI {
  constructor() {
    this.bars = {};
    const wrap = $("bars");
    for (const k of [...PLANES, "none"]) {
      const row = document.createElement("div");
      row.className = "bar mono";
      row.innerHTML = `<span class="name">${k === "none" ? "NONE" : k}</span><span class="track"><span class="fill"></span></span><span class="val">0.00</span>`;
      wrap.appendChild(row);
      this.bars[k] = { row, fill: row.querySelector(".fill"), val: row.querySelector(".val") };
    }
    this.slots = {};
    const tray = $("tray");
    for (const k of PLANES) {
      const s = document.createElement("div");
      s.className = "slot";
      s.innerHTML = `<span class="slot-label">${k}</span><span class="slot-name">${PLANE_NAMES[k]}</span><img alt=""><div class="slot-meta mono"><span class="q"></span><span class="st"></span></div>`;
      tray.appendChild(s);
      this.slots[k] = s;
    }
    this.tl = $("timeline");
    this.rewards = [];
    this.events = [];
    this.episodes = [];
    this.maxSteps = 200;
    this.setGlyph(null);
    window.addEventListener("resize", () => this.drawTimeline());
  }

  setHeader({ episode, nEpisodes, step, maxSteps, split, agent }) {
    if (episode != null) $("hdr-episode").textContent = `EP ${episode}/${nEpisodes ?? "—"}`;
    if (step != null) $("hdr-step").textContent = `STEP ${String(step).padStart(3, "0")}/${maxSteps ?? 200}`;
    if (split) {
      const el = $("hdr-split");
      el.textContent = split === "test" ? "Split TEST · unseen" : "Split TRAIN";
      el.style.color = split === "test" ? C.amber : "";
    }
    if (agent) $("hdr-agent").textContent = `Agent ${agent}`;
  }

  setStatus(kind, text) {
    const el = $("hdr-status");
    el.className = `status ${kind}`;
    el.querySelector("b").textContent = text;
  }

  setFps(fps) { $("hdr-fps").textContent = `${fps} fps`; }

  setGlyph(action, name, guidance) {
    $("glyph").innerHTML = `<svg viewBox="0 0 100 100">${action == null ? `<circle cx="50" cy="50" r="6" class="ghost"/>` : GLYPHS[action]}</svg>`;
    $("guidance-text").textContent = guidance || "Waiting for agent";
    $("action-id").textContent = action == null ? "—" : `action ${action} · ${name}`;
  }

  setState(st) {
    let top = "none", best = -1;
    for (const k of [...PLANES, "none"]) if (st.probs[k] > best) { best = st.probs[k]; top = k; }
    for (const k of [...PLANES, "none"]) {
      const b = this.bars[k];
      b.fill.style.width = `${(st.probs[k] * 100).toFixed(1)}%`;
      b.val.textContent = st.probs[k].toFixed(2);
      b.row.classList.toggle("top", k === top && k !== "none");
    }
    const q = st.quality;
    const qf = $("qfill");
    qf.style.width = `${(q * 100).toFixed(1)}%`;
    qf.classList.toggle("ok", q >= 0.65);
    $("qval").textContent = q.toFixed(2);
    const p = st.probe;
    $("ro-u").textContent = p.u.toFixed(3);
    $("ro-v").textContent = p.v.toFixed(3);
    $("ro-th").textContent = `${((p.theta * 180) / Math.PI).toFixed(1)}°`;
    $("ro-ph").textContent = `${((p.phi * 180) / Math.PI).toFixed(1)}°`;
    $("tray-count").textContent = `${PLANES.filter((k) => st.captured[k]).length} / 3`;
    this.maxSteps = st.max_steps || 200;
  }

  setGroundTruth(on, st) {
    $("gt-panel").classList.toggle("on", on);
    if (!st) return;
    const f = st.fetus;
    const deg = (x) => `${((x * 180) / Math.PI).toFixed(0)}°`;
    const al = PLANES.map((k) => `${k} ${st.alignment[k].toFixed(2)}`).join("  ");
    $("gt-body").innerHTML = on
      ? `Lie ${f.presentation}<br>Axis ${deg(f.axis)}<br>Align ${al}`
      : "hidden from agent";
  }

  capture(plane, q, step, img) {
    const s = this.slots[plane];
    s.querySelector("img").src = img;
    s.querySelector(".q").textContent = `Q ${q.toFixed(2)}`;
    s.querySelector(".st").textContent = `step ${String(step).padStart(3, "0")}`;
    requestAnimationFrame(() => s.classList.add("filled"));
  }

  toast(text, kind = "") {
    const t = $("bmode-toast");
    t.textContent = text;
    t.className = `bmode-toast mono show ${kind}`;
    clearTimeout(this._toastT);
    this._toastT = setTimeout(() => t.classList.remove("show"), 1600);
  }

  flash() {
    const f = $("bmode-flash");
    f.classList.remove("go");
    void f.offsetWidth;
    f.classList.add("go");
  }

  resetEpisode() {
    for (const k of PLANES) {
      const s = this.slots[k];
      s.classList.remove("filled");
      s.querySelector("img").removeAttribute("src");
    }
    this.rewards = [];
    this.events = [];
    this.setReturn(0, null);
    this.drawTimeline();
  }

  setReturn(ret, r) {
    const el = $("ret");
    el.textContent = `${ret >= 0 ? "+" : "−"}${Math.abs(ret).toFixed(2)}`;
    el.classList.toggle("neg", ret < 0);
    $("last-r").textContent = r == null ? "r —" : `r ${r >= 0 ? "+" : "−"}${Math.abs(r).toFixed(3)}`;
  }

  pushReward(step, r, event) {
    this.rewards.push(r);
    if (event) this.events.push({ step, kind: event[0], plane: event[1] });
    this.drawTimeline();
  }

  setEpisodes(list, current) {
    const el = $("episodes");
    el.innerHTML = `<div class="label" style="margin-bottom:6px">Episodes${list.length > 3 ? ` <span class="dim">· ${list.filter((e) => e.done && e.success).length}/${list.filter((e) => e.done).length} solved</span>` : ""}</div><div class="ep-list" id="ep-list">` + list.map((e) => {
      const cls = e.done ? (e.success ? "ok" : "bad") : "";
      return `<div class="ep mono ${e.episode === current && !e.done ? "current" : ""}"><span>EP ${e.episode}</span><span class="dot ${cls}"></span>` +
        `<span>${e.done ? `${e.planes}/3` : "…"}</span><span>${e.done ? `${e.steps} st` : ""}</span><span style="text-align:right">${e.done ? (e.ret >= 0 ? "+" : "−") + Math.abs(e.ret).toFixed(1) : ""}</span></div>`;
    }).join("") + "</div>";
    const box = $("ep-list");
    box.scrollTop = box.scrollHeight;  // keep the newest episode in view; scroll up for earlier ones
  }

  drawTimeline() {
    const c = this.tl;
    const dpr = Math.min(window.devicePixelRatio, 2);
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== w * dpr) { c.width = w * dpr; c.height = h * dpr; }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const N = this.maxSteps, pad = 4;
    const x = (i) => pad + ((w - 2 * pad) * i) / N;
    const mid = h * 0.5;
    const bar = (r) => Math.sign(r) * Math.sqrt(Math.min(Math.abs(r), 40) / 40) * h * 0.42;  // sqrt scale
    // zero line
    g.strokeStyle = C.line; g.lineWidth = 1;
    g.beginPath(); g.moveTo(pad, mid + 0.5); g.lineTo(w - pad, mid + 0.5); g.stroke();
    // step ticks every 50
    g.fillStyle = C.text3; g.font = "10px 'Plex Mono', monospace";
    for (let s = 0; s <= N; s += 50) {
      g.textAlign = s === N ? "right" : "left";
      g.fillText(String(s), s === N ? x(s) - 2 : x(s) + 2, h - 2);
    }
    g.textAlign = "left";
    // reward bars (clipped, sign-coloured)
    const bw = Math.max(1, (w - 2 * pad) / N - 1);
    this.rewards.forEach((r, i) => {
      const v = bar(r);
      g.fillStyle = r >= 0 ? C.amber : C.red;
      g.globalAlpha = Math.abs(r) > 1 ? 0.95 : 0.5;
      g.fillRect(x(i), v >= 0 ? mid - v : mid, bw, Math.max(1, Math.abs(v)));
    });
    g.globalAlpha = 1;
    // cumulative return line (own scale)
    let cum = 0;
    const cs = this.rewards.map((r) => (cum += r));
    const maxAbs = Math.max(10, ...cs.map(Math.abs));
    g.strokeStyle = C.text; g.lineWidth = 1.25; g.beginPath();
    cs.forEach((v, i) => { const y = mid - (v / maxAbs) * h * 0.45; i ? g.lineTo(x(i + 1), y) : g.moveTo(x(i + 1), y); });
    g.stroke();
    // capture events
    for (const e of this.events) {
      const ok = e.kind === "capture";
      if (!ok && e.kind !== "bad_capture") continue;
      g.strokeStyle = ok ? C.green : C.red; g.lineWidth = 1;
      g.beginPath(); g.moveTo(x(e.step) + 0.5, 6); g.lineTo(x(e.step) + 0.5, h - 14); g.stroke();
      if (ok) { g.fillStyle = C.green; g.fillText(e.plane, x(e.step) + 4, 14); }
    }
  }
}
