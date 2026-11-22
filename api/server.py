"""Tiny inference API for the trained probe-guidance policy (integration evidence).

    uv run python -m api.server [--algo ppo] [--port 8000]

    GET  /health  -> {"status": "ok", "algo": ..., "obs_dim": ...}
    POST /act     body {"observation": [obs_dim floats]}
                  -> {"action_id", "action_name", "guidance_text", "action_probs"}

A phone app builds the observation from the probe IMU + the capstone CNN outputs and
shows `guidance_text` as an on-screen arrow. Standard library only.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from environment.custom_env import ACTION_NAMES, GUIDANCE_TEXT
from main import pick_model
from training.common import load_policy


class Policy:
    def __init__(self, algo: str | None = None, path: Path | None = None):
        if path is None:
            algo, path = pick_model(algo)
        if path is None:
            raise SystemExit("No trained model found under models/*/best/")
        _, self.net, ckpt = load_policy(path)
        self.algo = ckpt["algo"]
        self.obs_dim = ckpt["config"].get("obs_dim", 16)
        self.path = path

    @torch.no_grad()
    def act(self, obs):
        if len(obs) != self.obs_dim:
            raise ValueError(f"observation must have {self.obs_dim} values, got {len(obs)}")
        out = self.net(torch.tensor(obs, dtype=torch.float32))
        # DQN outputs Q-values; softmax gives a readable preference distribution for both cases
        probs = torch.softmax(out, -1)
        a = int(out.argmax())
        return {"action_id": a, "action_name": ACTION_NAMES[a], "guidance_text": GUIDANCE_TEXT[a],
                "action_probs": [round(float(p), 4) for p in probs]}

    def export_onnx(self, out: Path):
        torch.onnx.export(self.net, torch.zeros(1, self.obs_dim), out, input_names=["observation"],
                          output_names=["logits"], dynamic_axes={"observation": {0: "batch"}})


def make_handler(policy: Policy):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"status": "ok", "algo": policy.algo, "obs_dim": policy.obs_dim,
                                        "model": str(policy.path)})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/act":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                obs = json.loads(self.rfile.read(n))["observation"]
                return self._send(200, policy.act(obs))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--export-torchscript", type=Path, default=None, help="save a TorchScript policy and exit")
    args = ap.parse_args()
    policy = Policy(args.algo)
    if args.export_torchscript:
        torch.jit.save(torch.jit.script(policy.net), str(args.export_torchscript))
        print("saved", args.export_torchscript)
        return
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(policy))
    print(f"Policy API ({policy.algo}) on http://127.0.0.1:{args.port}  · GET /health · POST /act")
    srv.serve_forever()


if __name__ == "__main__":
    main()
