"""Run the best trained agent on FetalScanNav-v0 with the 3D console and a verbose terminal log.

    uv run main.py                         # overall best model, 3 episodes, unseen test split
    uv run main.py --algo ppo --split train --episodes 5 --fps 12
    uv run main.py --no-browser
    uv run main.py --record assets/web/demo_episode.json   # save a replay for the offline page
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from environment.custom_env import ACTION_NAMES, GUIDANCE_TEXT, PLANES, FetalScanNavEnv

ROOT = Path(__file__).resolve().parent
ALGOS = ("dqn", "reinforce", "ppo", "a2c")
console = Console(highlight=False)


def best_dir(algo: str) -> Path:
    return ROOT / "models" / ("dqn" if algo == "dqn" else f"pg/{algo}") / "best"


def pick_model(algo: str | None):
    """Overall best = highest mean return on the training-distribution evaluation
    (the unseen test split is never used for selection)."""
    cands = []
    for a in ([algo] if algo else ALGOS):
        s = best_dir(a) / "summary.json"
        if s.exists():
            cands.append((json.loads(s.read_text())["eval_train"]["mean_return"], a))
    if not cands:
        return None, None
    _, a = max(cands)
    return a, best_dir(a) / "model.pt"


def load_agent(algo, path):
    from training.common import load_policy

    act, _, ckpt = load_policy(path)
    return act, ckpt


def oracle_agent(env):
    from analysis.baselines import oracle_policy

    return lambda obs: oracle_policy(env, obs, None)[0]


def fmt_step(step, a, st, obs_probs, r, ret, event):
    p = st["probe"]
    probs = " ".join(f"{k}{obs_probs[i]:.2f}" for i, k in enumerate(("HC", "AC", "FL", "∅")))
    line = (f"[dim]{step:>3d}[/] │ {ACTION_NAMES[a]:<14s} │ u {p['u']:.3f} v {p['v']:.3f} "
            f"θ {np.degrees(p['theta']):6.1f}° φ {np.degrees(p['phi']):5.1f}° │ q {st['quality']:.2f} │ "
            f"{probs} │ r {r:+6.2f} │ R {ret:+7.2f}")
    if event and event[0] == "capture":
        return f"[bold #e8a33d]{line}  ◀ CAPTURED {event[1]} (q={event[2]:.2f})[/]"
    if event and event[0] == "bad_capture":
        return f"[#c8553d]{line}  ✗ rejected capture: {event[3]}[/]"
    if event and event[0] == "exit":
        return f"[#c8553d]{line}  ✗ probe left scan region[/]"
    if st["motion"]:
        return f"{line}  [dim]· fetal movement[/]"
    return line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=ALGOS + ("oracle",), default=None, help="default: overall best model")
    ap.add_argument("--model", type=Path, default=None, help="explicit model.pt path")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-gui", action="store_true", help="terminal only")
    ap.add_argument("--record", type=Path, default=None, help="write a replay JSON instead of serving")
    args = ap.parse_args()

    env = FetalScanNavEnv(split=args.split)
    if args.algo == "oracle":
        algo, path, cfg, run = "oracle", None, {}, "ground-truth reference"
        act = oracle_agent(env)
    else:
        if args.model:
            algo, path = (args.algo or "model"), args.model
        else:
            algo, path = pick_model(args.algo)
        if path is None or not Path(path).exists():
            console.print("[#c8553d]No trained model found. Train first, e.g.[/] uv run python -m training.pg_training --algo ppo --run all")
            sys.exit(1)
        act, ckpt = load_agent(algo, path)
        algo = ckpt.get("algo", algo)
        cfg = ckpt["config"]
        src = Path(path).parent / "source_run.txt"
        run = src.read_text().strip() if src.exists() else Path(path).parent.name

    agent_label = f"{algo.upper()} · {run}"
    hdr = Table.grid(padding=(0, 2))
    hdr.add_row("[dim]model[/]", str(path.relative_to(ROOT) if path and path.is_relative_to(ROOT) else path))
    hdr.add_row("[dim]algorithm[/]", agent_label)
    hdr.add_row("[dim]split[/]", f"{args.split}" + ("  (unseen: transverse lie, large offsets, more movement)" if args.split == "test" else ""))
    hdr.add_row("[dim]episodes[/]", f"{args.episodes}  seed {args.seed}  {args.fps:g} steps/s")
    if cfg:
        hdr.add_row("[dim]config[/]", ", ".join(f"{k}={v}" for k, v in cfg.items() if k not in ("total_frames", "obs_dim")))
    console.print(Panel(hdr, title="[bold]FetalScanNav · probe guidance agent[/]", border_style="#8c877f"))

    recording = args.record is not None
    bridge = None
    meta = {"agent": agent_label, "fps": args.fps, "split": args.split, "mode": "live"}
    if not recording and not args.no_gui:
        from environment.rendering import VizBridge

        bridge = VizBridge(args.port, meta=meta).start(open_browser=not args.no_browser)
        console.print(f"[dim]3D console:[/] {bridge.url}")
        if not args.no_browser:
            bridge.wait_for_client(10)
            time.sleep(1.5)

    messages = []

    def publish(kind, payload):
        if bridge:
            bridge.publish(kind, payload)
        if recording:
            from environment.rendering import _jsonable

            messages.append({"type": kind, **_jsonable(payload)})

    delay = 0 if (recording or not bridge) and args.no_gui or recording else 1.0 / args.fps
    results = []
    ep = 0
    while ep < args.episodes:
        obs, info = env.reset(seed=args.seed + ep, options={"split": args.split})
        console.rule(f"[bold]Episode {ep + 1}/{args.episodes}[/]  [dim]hidden lie: {info['presentation']}[/]", style="#8c877f")
        console.print("[dim]step │ action         │ probe pose                         │ q    │ plane probabilities       │ reward  │ return[/]")
        publish("episode_start", {"episode": ep + 1, "n_episodes": args.episodes, "state": env.render_state(), "meta": meta})
        ret, done, restarted = 0.0, False, False
        while not done:
            a = act(obs)
            obs, r, te, tr, info = env.step(a)
            ret += r
            done = te or tr
            st = env.render_state()
            console.print(fmt_step(info["steps"], a, st, obs[7:11], r, ret, info["event"]))
            publish("step", {"episode": ep + 1, "action": a, "action_name": ACTION_NAMES[a], "guidance": GUIDANCE_TEXT[a],
                             "reward": r, "ret": ret, "event": info["event"], "state": st})
            if bridge:
                bridge.wait_turn(delay)
                if bridge.restart_requested:
                    bridge.restart_requested = False
                    restarted = True
                    console.print("[dim]↺ episode restarted from the console[/]")
                    break
            elif delay:
                time.sleep(delay)
        if restarted:
            continue
        cq = info["capture_qualities"]
        res = {"episode": ep + 1, "success": bool(info["success"]), "planes": int(info["planes_captured"]),
               "steps": info["steps"], "ret": ret, "qualities": {k: float(cq[i]) for i, k in enumerate(PLANES)}}
        results.append(res)
        publish("episode_end", res)
        colour = "#6faf8f" if res["success"] else "#c8553d"
        qs = "  ".join(f"{k} {v:.2f}" if v > 0 else f"{k} —" for k, v in res["qualities"].items())
        console.print(Panel(f"[{colour}]{'SUCCESS' if res['success'] else 'INCOMPLETE'}[/]   planes {res['planes']}/3   "
                            f"steps {res['steps']}   return {ret:+.2f}\n[dim]capture quality[/]  {qs}",
                            border_style=colour, title=f"Episode {ep + 1} summary"))
        ep += 1
        if bridge and ep < args.episodes:
            time.sleep(2.0)

    sr = np.mean([r["success"] for r in results])
    console.print(f"\n[bold]Overall[/]  success {sr:.0%}  mean return {np.mean([r['ret'] for r in results]):+.2f}  "
                  f"mean steps {np.mean([r['steps'] for r in results]):.1f}")

    if recording:
        args.record.write_text(json.dumps({"meta": {**meta, "mode": "demo"}, "messages": messages}))
        console.print(f"[dim]wrote replay to {args.record}[/]")
    if bridge:
        console.print("[dim]Console stays open; press Ctrl+C to quit.[/]")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            bridge.stop()


if __name__ == "__main__":
    main()
