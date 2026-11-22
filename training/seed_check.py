"""Seed-robustness check: retrain each algorithm's best config with fresh seeds.

    uv run python -m training.seed_check                     # 3 seeds x 4 algorithms
    uv run python -m training.seed_check --seeds 1001 1002 1003 1004 --algos ppo

Results go to models/seeds/<algo>/run_<seed>/summary.json and logs/seeds/<algo>/ so the
sweep tables and best/ models are untouched. A summary is written to logs/seeds/seed_check.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("RL_WARNINGS", "False")

import numpy as np  # noqa: E402

from training import common  # noqa: E402

ALGOS = ("dqn", "reinforce", "ppo", "a2c")
META = ("seed", "total_frames", "obs_dim")


def _job(algo, seed, cfg, frames):
    common.LOGS = common.ROOT / "logs" / "seeds"
    common.MODELS = common.ROOT / "models" / "seeds"
    common.CONFIGS = common.ROOT / "logs" / "seeds" / "configs"
    from training import dqn_training, pg_training

    overrides = {k: v for k, v in cfg.items() if k not in META}
    (common.LOGS / algo).mkdir(parents=True, exist_ok=True)
    with open(common.LOGS / algo / f"run_{seed}_stdout.txt", "w") as f, contextlib.redirect_stdout(f):
        if algo == "dqn":
            s = dqn_training.train(seed, overrides=overrides, total_frames=frames, seed=seed)
        else:
            s = pg_training.TRAINERS[algo](seed, overrides=overrides, total_frames=frames, seed=seed)
    return algo, seed, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=list(ALGOS), choices=ALGOS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1001, 1002, 1003])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    jobs = []
    for algo in args.algos:
        best = common.model_dir(algo) / "best"
        cfg = json.loads((best / "config.json").read_text())
        jobs += [(algo, s, cfg, cfg["total_frames"]) for s in args.seeds]
    print(f"{len(jobs)} seed runs on {args.workers} workers", flush=True)

    res = {a: [] for a in args.algos}
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(_job, *j) for j in jobs]):
            algo, seed, s = fut.result()
            res[algo].append({"seed": seed, "train": s["eval_train"], "test": s["eval_test"]})
            print(f"{algo:<9} seed {seed}  train {s['eval_train']['mean_return']:7.2f} ({s['eval_train']['success_rate']:4.0%})  "
                  f"test {s['eval_test']['mean_return']:7.2f} ({s['eval_test']['success_rate']:4.0%})  "
                  f"elapsed {(time.time() - t0) / 60:5.1f} min", flush=True)

    out = {}
    for algo, rows in res.items():
        src = (common.model_dir(algo) / "best" / "source_run.txt").read_text().strip()
        f = lambda split, key: [r[split][key] for r in rows]  # noqa: E731
        out[algo] = {
            "config_from": src, "seeds": [r["seed"] for r in rows],
            "train_return_mean": float(np.mean(f("train", "mean_return"))), "train_return_std": float(np.std(f("train", "mean_return"))),
            "train_success_mean": float(np.mean(f("train", "success_rate"))),
            "test_return_mean": float(np.mean(f("test", "mean_return"))), "test_return_std": float(np.std(f("test", "mean_return"))),
            "test_success_mean": float(np.mean(f("test", "success_rate"))),
            "runs_that_learned": int(sum(r["train"]["success_rate"] > 0.5 for r in rows)),
            "per_seed": rows,
        }
        print(f"{algo:<9} best config ({src}) over {len(rows)} seeds: train success {out[algo]['train_success_mean']:.0%}, "
              f"test success {out[algo]['test_success_mean']:.0%}, learned in {out[algo]['runs_that_learned']}/{len(rows)}")
    p = common.ROOT / "logs" / "seeds" / "seed_check.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
