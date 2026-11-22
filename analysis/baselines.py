"""Reference baselines for FetalScanNav-v0.

random      : uniform random actions.
hillclimb   : uses only the observation. Repeats a move while quality rises,
              otherwise tries the next move; captures when q is high.
oracle      : cheats by reading the hidden target pose (upper reference only).

    uv run python -m analysis.baselines
"""

import json

import numpy as np

from environment.custom_env import FetalScanNavEnv, _plane_angle_dist, _wrap_pi


def random_policy(env, obs, state):
    return int(env.np_random.integers(9)), state


def hillclimb_policy(env, obs, state):
    q, dq = obs[5], obs[6]
    probs, caps = obs[7:11], obs[11:14]
    k = int(np.argmax(probs))
    if q >= 0.75 and k < 3 and caps[k] == 0:
        return 8, state
    if state is None:
        state = {"a": 0, "n": 0}
    # keep going while improving; otherwise switch action
    if dq <= 0 or state["n"] > 12:
        state["a"] = (state["a"] + 1) % 8
        state["n"] = 0
    # stay inside the region
    u, v = obs[0], obs[1]
    if np.hypot(u - 0.5, v - 0.5) > 0.42:
        a = 3 if u < 0.5 else 2
        if abs(v - 0.5) > abs(u - 0.5):
            a = 0 if v < 0.5 else 1
        return a, state
    state["n"] += 1
    return state["a"], state


def oracle_policy(env, obs, state):
    open_ = np.where(~env.captured)[0]
    t = env.targets
    k = open_[np.argmin(np.hypot(t[open_, 0] - env.u, t[open_, 1] - env.v))]
    du, dv = t[k, 0] - env.u, t[k, 1] - env.v
    s = env.cfg.slide_step
    if max(abs(du), abs(dv)) > s * 0.6:
        if abs(dv) >= abs(du):
            return (0 if dv > 0 else 1), state
        return (3 if du > 0 else 2), state
    dr = _wrap_pi(2 * (t[k, 2] - env.theta)) / 2
    if abs(dr) > env.cfg.rot_step * 0.6:
        return (4 if dr > 0 else 5), state
    dt = t[k, 3] - env.phi
    if abs(dt) > env.cfg.tilt_step * 0.6:
        return (6 if dt > 0 else 7), state
    return 8, state


def evaluate(policy, split="train", episodes=100, seed=10_000):
    env = FetalScanNavEnv(split=split)
    rets, succ, steps, quals = [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        state, ret, done = None, 0.0, False
        while not done:
            a, state = policy(env, obs, state)
            obs, r, te, tr, info = env.step(a)
            ret += r
            done = te or tr
        rets.append(ret)
        succ.append(info["success"])
        steps.append(info["steps"])
        cq = info["capture_qualities"]
        quals.append(cq[cq > 0].mean() if (cq > 0).any() else 0.0)
    return {
        "mean_return": float(np.mean(rets)),
        "std_return": float(np.std(rets)),
        "success_rate": float(np.mean(succ)),
        "mean_steps": float(np.mean(steps)),
        "mean_capture_quality": float(np.mean(quals)),
    }


if __name__ == "__main__":
    out = {}
    for name, pol in [("random", random_policy), ("hillclimb", hillclimb_policy), ("oracle", oracle_policy)]:
        for split in ("train", "test"):
            res = evaluate(pol, split)
            out[f"{name}/{split}"] = res
            print(f"{name:10s} {split:5s} " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()))
    with open("assets/plots/baselines.json", "w") as f:
        json.dump(out, f, indent=2)
