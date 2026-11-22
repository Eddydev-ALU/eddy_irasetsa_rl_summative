"""Run every hyperparameter sweep (DQN, REINFORCE, PPO, A2C) in parallel worker processes.

    uv run python -m training.run_sweeps                 # all algorithms, all runs
    uv run python -m training.run_sweeps --workers 6 --algos ppo a2c
    uv run python -m training.run_sweeps --frames 50000  # quick smoke test

Each run writes logs/<algo>/run_xx_*.csv and models/.../run_xx/{model.pt,config.json,summary.json};
its console output goes to logs/<algo>/run_xx_stdout.txt. The best run per algorithm is copied
to models/.../best/ at the end.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from training.common import LOGS, select_best

ALGOS = ("dqn", "reinforce", "ppo", "a2c")


def _job(algo: str, run_id: int, frames: int):
    from training import dqn_training, pg_training

    out = LOGS / algo
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"run_{run_id:02d}_stdout.txt", "w") as f, contextlib.redirect_stdout(f):
        t0 = time.time()
        if algo == "dqn":
            s = dqn_training.train(run_id, total_frames=frames)
        else:
            s = pg_training.TRAINERS[algo](run_id, total_frames=frames)
    return algo, run_id, s, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=list(ALGOS), choices=ALGOS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--frames", type=int, default=None, help="override the frame budget (default: per-trainer TOTAL_FRAMES)")
    args = ap.parse_args()

    from training import dqn_training, pg_training

    jobs = []
    for algo in args.algos:
        runs = dqn_training.SWEEP if algo == "dqn" else pg_training.SWEEPS[algo][1]
        frames = args.frames or (dqn_training.TOTAL_FRAMES if algo == "dqn" else pg_training.TOTAL_FRAMES)
        jobs += [(algo, r, frames) for r in runs]
    # interleave algorithms so partial results are balanced if interrupted
    jobs.sort(key=lambda j: (j[1], ALGOS.index(j[0])))
    print(f"{len(jobs)} runs on {args.workers} workers", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_job, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            algo, run_id, s, dt = fut.result()
            tr, te = s["eval_train"], s["eval_test"]
            print(f"[{i:>2}/{len(jobs)}] {algo:<9} run_{run_id:02d}  train {tr['mean_return']:7.2f} "
                  f"({tr['success_rate']:4.0%})  test {te['mean_return']:7.2f} ({te['success_rate']:4.0%})  "
                  f"{dt / 60:4.1f} min   elapsed {(time.time() - t0) / 60:5.1f} min", flush=True)

    best = {a: select_best(a) for a in args.algos}
    print("best runs:", json.dumps(best))


if __name__ == "__main__":
    main()
