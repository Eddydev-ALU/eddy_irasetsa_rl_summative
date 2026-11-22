// FetalScanNav console: wires the data stream (live SSE or recorded demo) to the 3D scene, B-mode and panels.
import { Scene3D } from "./scene.js";
import { BMode } from "./bmode.js";
import { UI } from "./ui.js";

const scene = new Scene3D(document.getElementById("scene"));
const bmode = new BMode(document.getElementById("bmode"), document.getElementById("bmode-scale"));
const ui = new UI();

let last = null;          // last state message
let gt = false;
let live = false;
let paused = false;
const episodes = [];
let meta = {};

function onMessage(m) {
  if (m.type === "control") {
    paused = m.paused;
    ui.setStatus(paused ? "paused" : live ? "live" : "demo", paused ? "Paused" : live ? "Live" : "Demo");
    return;
  }
  if (m.meta) meta = { ...meta, ...m.meta };
  if (m.type === "episode_start") {
    last = m;
    scene.resetEpisode(m.state);
    bmode.setState(m.state);
    ui.resetEpisode();
    ui.setState(m.state);
    ui.setGlyph(null);
    ui.setHeader({ episode: m.episode, nEpisodes: m.n_episodes, step: 0, maxSteps: m.state.max_steps, split: m.state.split, agent: meta.agent });
    if (meta.fps) ui.setFps(meta.fps);
    if (!episodes.find((e) => e.episode === m.episode)) episodes.push({ episode: m.episode, done: false });
    ui.setEpisodes(episodes.slice(-4), m.episode);
    ui.setGroundTruth(gt, m.state);
    return;
  }
  if (m.type === "step") {
    last = m;
    const st = m.state;
    scene.setState(st);
    scene.addTrailPoint(st);
    bmode.setState(st);
    ui.setState(st);
    ui.setGlyph(m.action, m.action_name, m.guidance);
    ui.setHeader({ step: st.steps, maxSteps: st.max_steps });
    ui.setReturn(m.ret, m.reward);
    ui.pushReward(st.steps, m.reward, m.event);
    ui.setGroundTruth(gt, st);
    const ev = m.event;
    if (ev && ev[0] === "capture") {
      bmode.freeze(500);
      ui.flash();
      // grab the frozen frame after it has rendered
      setTimeout(() => ui.capture(ev[1], ev[2], st.steps, bmode.snapshot()), 60);
      ui.toast(`Frozen · ${ev[1]} saved · Q ${ev[2].toFixed(2)}`, "ok");
    } else if (ev && ev[0] === "bad_capture") {
      ui.toast(`Rejected · ${ev[3]} · q ${ev[2].toFixed(2)}`, "bad");
    } else if (ev && ev[0] === "exit") {
      ui.toast("Probe left scan region", "bad");
    }
    return;
  }
  if (m.type === "episode_end") {
    const e = episodes.find((x) => x.episode === m.episode);
    if (e) Object.assign(e, { done: true, success: m.success, planes: m.planes, steps: m.steps, ret: m.ret });
    ui.setEpisodes(episodes.slice(-4), m.episode);
    ui.toast(m.success ? `Scan complete · 3/3 planes · ${m.steps} steps` : `Episode ended · ${m.planes}/3 planes`, m.success ? "ok" : "bad");
  }
}

// ---------------------------------------------------------------- data sources
async function control(action) {
  if (live) {
    try { await fetch("/control", { method: "POST", body: JSON.stringify({ action }) }); } catch (e) { /* offline */ }
  } else {
    demo.control(action);
  }
}

const demo = {
  msgs: [], i: 0, timer: null, fps: 8,
  async start() {
    const res = await fetch("demo_episode.json");
    const data = await res.json();
    this.msgs = data.messages;
    meta = { ...data.meta };
    this.fps = data.meta.fps || 8;
    ui.setStatus("demo", "Demo replay");
    this.loop();
  },
  loop() {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      if (!paused) this.next();
      this.loop();
    }, 1000 / this.fps);
  },
  next() {
    if (this.i >= this.msgs.length) {
      this.i = 0;
      episodes.length = 0;
    }
    const m = this.msgs[this.i++];
    onMessage(m);
    // linger briefly on episode ends
    if (m.type === "episode_end") { clearTimeout(this.timer); this.timer = setTimeout(() => this.loop(), 1800); }
  },
  control(action) {
    if (action === "toggle") paused = !paused;
    if (action === "step") { paused = true; this.next(); }
    if (action === "restart") {
      // rewind to the start of the current episode
      for (let j = this.i - 1; j >= 0; j--) if (this.msgs[j].type === "episode_start") { this.i = j; break; }
    }
    ui.setStatus(paused ? "paused" : "demo", paused ? "Paused" : "Demo replay");
  },
};

async function connect() {
  const wantDemo = new URLSearchParams(location.search).has("demo");
  let status = null;
  if (!wantDemo) {
    try { status = await (await fetch("/api/status")).json(); } catch (e) { status = null; }
  }
  if (status && status.live && status.meta && status.meta.mode !== "demo") {
    live = true;
    meta = status.meta;
    ui.setStatus("live", "Live");
    if (meta.fps) ui.setFps(meta.fps);
    const es = new EventSource("/events");
    es.onmessage = (e) => onMessage(JSON.parse(e.data));
    es.onerror = () => ui.setStatus("", "Disconnected");
  } else {
    await demo.start();
  }
}

// -------------------------------------------------------------------- keyboard
window.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); control("toggle"); }
  else if (e.code === "ArrowRight") control("step");
  else if (e.code === "KeyR") control("restart");
  else if (e.code === "KeyG") {
    gt = !gt;
    scene.setGroundTruth(gt);
    ui.setGroundTruth(gt, last && last.state);
  } else if (["Digit1", "Digit2", "Digit3"].includes(e.code)) {
    const name = scene.setView(Number(e.code.slice(-1)));
    if (name) document.getElementById("view-name").textContent = name;
  }
});

function frame() {
  scene.render();
  bmode.render();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
connect();
window.__fsn = { scene, bmode, ui, onMessage };
