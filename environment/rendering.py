"""Python side of the three.js visualization.

A small standard-library HTTP server that
  * serves the static web app in assets/web/,
  * streams every environment step to the page with Server-Sent Events (/events),
  * accepts playback controls from the page (POST /control: pause, step, restart).

Without a live agent the page replays assets/web/demo_episode.json:
    uv run python -m environment.rendering --demo
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent / "assets" / "web"


def _jsonable(x):
    import numpy as np

    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class _Handler(SimpleHTTPRequestHandler):
    bridge: "VizBridge" = None

    def log_message(self, *args):  # keep the terminal clean for the agent log
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.startswith("/events"):
            return self._sse()
        if self.path.startswith("/api/status"):
            return self._json({"live": True, "meta": self.bridge.meta})
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/control"):
            n = int(self.headers.get("Content-Length", 0))
            cmd = json.loads(self.rfile.read(n) or b"{}")
            self.bridge.control(cmd)
            return self._json({"ok": True, "paused": self.bridge.paused})
        self.send_error(404)

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = self.bridge.subscribe()
        try:
            for msg in self.bridge.history_snapshot():
                self.wfile.write(f"data: {msg}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=10)
                except queue.Empty:
                    msg = None
                self.wfile.write((f"data: {msg}\n\n" if msg else ": keep-alive\n\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.bridge.unsubscribe(q)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128   # browsers open many parallel connections for ES modules


class VizBridge:
    """Streams env states to connected browsers and relays their playback controls."""

    def __init__(self, port: int = 8765, meta: dict | None = None):
        self.port = port
        self.meta = meta or {}
        self.paused = False
        self.restart_requested = False
        self._step_once = False
        self._subs: list[queue.Queue] = []
        self._history: list[str] = []   # current episode, replayed to late joiners
        self._lock = threading.Lock()
        handler = partial(type("H", (_Handler,), {"bridge": self}), directory=str(WEB_DIR))
        # fall back to the next free port if this one is taken (e.g. a viewer is already open)
        for p in range(port, port + 20):
            try:
                self.server = _Server(("127.0.0.1", p), handler)
                break
            except OSError:
                continue
        else:
            raise OSError(f"no free port in {port}-{port + 19}")
        self.port = p
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/"

    def start(self, open_browser: bool = True):
        self.thread.start()
        if open_browser:
            webbrowser.open(self.url)
        return self

    def wait_for_client(self, timeout: float = 15.0):
        t0 = time.time()
        while not self._subs and time.time() - t0 < timeout:
            time.sleep(0.1)
        return bool(self._subs)

    # pub/sub
    def subscribe(self):
        q = queue.Queue()
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def history_snapshot(self):
        with self._lock:
            return list(self._history)

    def publish(self, kind: str, payload: dict):
        msg = json.dumps({"type": kind, **_jsonable(payload)})
        with self._lock:
            if kind == "episode_start":
                self._history = []
            self._history.append(msg)
            for q in self._subs:
                q.put(msg)

    # playback control
    def control(self, cmd: dict):
        action = cmd.get("action")
        if action == "pause":
            self.paused = True
        elif action == "resume":
            self.paused = False
        elif action == "toggle":
            self.paused = not self.paused
        elif action == "step":
            self.paused = True
            self._step_once = True
        elif action == "restart":
            self.restart_requested = True
        self.publish("control", {"paused": self.paused})

    def wait_turn(self, delay: float):
        """Block between agent steps, honouring pause / single-step from the page."""
        time.sleep(delay)
        while self.paused and not self._step_once and not self.restart_requested:
            time.sleep(0.05)
        self._step_once = False

    def stop(self):
        self.server.shutdown()


def main():
    p = argparse.ArgumentParser(description="Serve the FetalScanNav visualization")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--demo", action="store_true", help="replay assets/web/demo_episode.json")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    bridge = VizBridge(args.port, meta={"mode": "demo"}).start(open_browser=False)
    url = bridge.url + ("?demo=1" if args.demo else "")
    print(f"Serving {WEB_DIR} at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bridge.stop()


if __name__ == "__main__":
    main()
