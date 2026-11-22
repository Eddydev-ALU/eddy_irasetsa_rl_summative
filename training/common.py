"""Shared training utilities: env factory, seeding, logging, evaluation, saving."""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchrl.envs import Compose, GymWrapper, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.gym_like import default_info_dict_reader

from environment.custom_env import FetalScanNavEnv

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
CONFIGS = ROOT / "training" / "configs"
EVAL_SEEDS = 20           # fixed seeds used for periodic evaluation
FINAL_EVAL_EPISODES = 100  # per split for the final evaluation


def model_dir(algo: str) -> Path:
    return MODELS / ("dqn" if algo == "dqn" else f"pg/{algo}")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(split: str = "train", seed: int = 0) -> TransformedEnv:
    base = GymWrapper(FetalScanNavEnv(split=split), categorical_action_encoding=True)
    base.set_info_dict_reader(default_info_dict_reader(["success", "planes_captured"]))
    env = TransformedEnv(base, Compose(StepCounter(), RewardSum()))
    env.set_seed(seed)
    return env


def mlp(in_dim: int, out_dim: int, width: int, depth: int = 2) -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]
        d = width
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def greedy_act_fn(net: nn.Module):
    """Deterministic action: argmax of Q-values (DQN) or of logits (policy gradient)."""

    @torch.no_grad()
    def act(obs: np.ndarray) -> int:
        return int(net(torch.as_tensor(obs, dtype=torch.float32)).argmax(-1))

    return act


def evaluate(act, split="train", episodes=EVAL_SEEDS, seed_base=100_000):
    env = FetalScanNavEnv(split=split)
    rows = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed_base + ep)
        ret, done = 0.0, False
        while not done:
            obs, r, te, tr, info = env.step(act(obs))
            ret += r
            done = te or tr
        cq = info["capture_qualities"]
        rows.append(
            {
                "return": ret,
                "success": float(info["success"]),
                "steps": info["steps"],
                "planes": info["planes_captured"],
                "capture_quality": float(cq[cq > 0].mean()) if (cq > 0).any() else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    return {
        "mean_return": float(df["return"].mean()),
        "std_return": float(df["return"].std(ddof=0)),
        "success_rate": float(df["success"].mean()),
        "mean_planes": float(df["planes"].mean()),
        "mean_steps": float(df["steps"].mean()),
        "mean_capture_quality": float(df["capture_quality"].mean()),
    }


def episodes_to_converge(returns: np.ndarray, window: int = 20, tol: float = 0.05) -> int:
    """First episode after which the 20-episode moving average stays within
    `tol` (5 %) of the curve's range below its final plateau (mean of last 10 %)."""
    if len(returns) < window * 2:
        return len(returns)
    ma = pd.Series(returns).rolling(window).mean().to_numpy()[window - 1:]
    plateau = ma[-max(1, len(ma) // 10):].mean()
    band = tol * (ma.max() - ma.min() + 1e-8)
    ok = ma >= plateau - band
    bad = np.where(~ok)[0]
    first = 0 if len(bad) == 0 else bad[-1] + 1
    return int(min(first + window - 1, len(returns)))


class RunLogger:
    def __init__(self, algo: str, run_id: int):
        self.algo, self.run_id = algo, run_id
        self.dir = LOGS / algo
        self.dir.mkdir(parents=True, exist_ok=True)
        self.episodes, self.updates, self.evals = [], [], []
        self.t0 = time.time()

    def log_episodes(self, batch, frames: int):
        done = batch["next", "done"].squeeze(-1)
        if not done.any():
            return
        rets = batch["next", "episode_reward"].squeeze(-1)[done]
        lens = batch["next", "step_count"].squeeze(-1)[done]
        succ = batch["next", "success"].reshape(done.shape)[done]
        planes = batch["next", "planes_captured"].reshape(done.shape)[done]
        for r, l, s, p in zip(rets.tolist(), lens.tolist(), succ.tolist(), planes.tolist()):
            self.episodes.append(
                {"episode": len(self.episodes) + 1, "frames": frames, "return": r, "length": l,
                 "success": float(s), "planes": p}
            )

    def log_update(self, frames: int, **vals):
        self.updates.append({"update": len(self.updates) + 1, "frames": frames, **vals})

    def log_eval(self, frames: int, res: dict):
        self.evals.append({"frames": frames, "episode": len(self.episodes), **res})

    def recent(self, n=50):
        if not self.episodes:
            return float("nan"), float("nan")
        e = self.episodes[-n:]
        return np.mean([x["return"] for x in e]), np.mean([x["success"] for x in e])

    def flush(self):
        tag = f"run_{self.run_id:02d}"
        pd.DataFrame(self.episodes).to_csv(self.dir / f"{tag}_episodes.csv", index=False)
        pd.DataFrame(self.updates).to_csv(self.dir / f"{tag}_updates.csv", index=False)
        pd.DataFrame(self.evals).to_csv(self.dir / f"{tag}_eval.csv", index=False)


def finish_run(algo: str, run_id: int, cfg: dict, net: nn.Module, logger: RunLogger, extra_state=None):
    """Final evaluation on train + unseen test split, save model/config/summary."""
    logger.flush()
    act = greedy_act_fn(net)
    train_res = evaluate(act, "train", FINAL_EVAL_EPISODES, seed_base=500_000)
    test_res = evaluate(act, "test", FINAL_EVAL_EPISODES, seed_base=600_000)
    rets = np.array([e["return"] for e in logger.episodes]) if logger.episodes else np.zeros(1)
    summary = {
        "algo": algo,
        "run": run_id,
        "eval_train": train_res,
        "eval_test": test_res,
        "episodes_trained": len(logger.episodes),
        "episodes_to_converge": episodes_to_converge(rets),
        "wall_time_s": round(time.time() - logger.t0, 1),
    }
    out = model_dir(algo) / f"run_{run_id:02d}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"algo": algo, "config": cfg, "policy_state": net.state_dict(), **(extra_state or {})}, out / "model.pt")
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    CONFIGS.mkdir(parents=True, exist_ok=True)
    (CONFIGS / f"{algo}_run_{run_id:02d}.json").write_text(json.dumps(cfg, indent=2))
    return summary


def select_best(algo: str):
    """Copy the run with the highest final train-eval mean return to best/."""
    d = model_dir(algo)
    runs = sorted(d.glob("run_*/summary.json"))
    if not runs:
        return None
    best = max(runs, key=lambda p: json.loads(p.read_text())["eval_train"]["mean_return"])
    dst = d / "best"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(best.parent, dst)
    (dst / "source_run.txt").write_text(best.parent.name + "\n")
    return best.parent.name


def load_policy(path: Path):
    """Load a saved model.pt and return (greedy act fn, network, checkpoint)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    net = mlp(16, 9, cfg["width"])
    net.load_state_dict(ckpt["policy_state"])
    net.eval()
    return greedy_act_fn(net), net, ckpt


def print_progress(algo, run_id, frames, total, logger, extra=""):
    r, s = logger.recent()
    print(f"[{algo} run_{run_id:02d}] frames {frames:>7d}/{total} | ep {len(logger.episodes):>5d} "
          f"| ret(50) {r:7.2f} | succ(50) {s:4.2f} {extra}", flush=True)
