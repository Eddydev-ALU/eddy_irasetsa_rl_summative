"""Regenerate every report figure and results table from logs/ and models/.

    uv run python -m analysis.make_plots
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOGS, MODELS, OUT = ROOT / "logs", ROOT / "models", ROOT / "assets" / "plots"
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
# Fixed categorical order (validated: CVD dE >= 9.1 adjacent, normal-vision >= 22.9).
COLOR = {"dqn": "#2a78d6", "reinforce": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2, "figure.facecolor": "white",
    "axes.prop_cycle": plt.cycler(color=[COLOR[a] for a in ALGOS]),
})


def mdir(algo):
    return MODELS / ("dqn" if algo == "dqn" else f"pg/{algo}")


def summaries(algo):
    rows = []
    for p in sorted(mdir(algo).glob("run_*/summary.json")):
        s = json.loads(p.read_text())
        c = json.loads((p.parent / "config.json").read_text())
        rows.append({"run": s["run"], **s, "config": c})
    return rows


def best_run(algo):
    src = mdir(algo) / "best" / "source_run.txt"
    return int(src.read_text().strip().split("_")[1]) if src.exists() else None


def load(algo, run, kind):
    p = LOGS / algo / f"run_{run:02d}_{kind}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def smooth(x, w=50):
    return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------- figures
def fig_cumulative_rewards():
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    for ax, algo in zip(axes.flat, ALGOS):
        r = best_run(algo)
        ep, ev = load(algo, r, "episodes"), load(algo, r, "eval")
        if ep.empty:
            continue
        ax.plot(ep["episode"], ep["return"], color=COLOR[algo], alpha=0.18, linewidth=0.8)
        ax.plot(ep["episode"], smooth(ep["return"]), color=COLOR[algo], label="training return (50-ep mean)")
        if not ev.empty:
            ax.plot(ev["episode"], ev["mean_return"], "o", color=INK, markersize=3.5, label="greedy eval (20 seeds)")
        ax.set_title(f"{LABEL[algo]} · run_{r:02d}")
        ax.set_xlabel("episode")
        ax.set_ylabel("episode return")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Cumulative reward per episode, best run of each method", fontweight="semibold")
    save(fig, "cumulative_rewards.png")


def fig_overlay():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for algo in ALGOS:
        r = best_run(algo)
        ev = load(algo, r, "eval")
        if ev.empty:
            continue
        y = smooth(ev["mean_return"], 3)
        ax.plot(ev["frames"] / 1e3, y, color=COLOR[algo], label=f"{LABEL[algo]} (run_{r:02d})")
        ax.annotate(LABEL[algo], (ev["frames"].iloc[-1] / 1e3, y[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_xlabel("environment frames (thousands)")
    ax.set_ylabel("greedy eval return (3-point mean)")
    ax.set_title("Evaluation return vs. frames, best run per method (same 20 seeds)")
    ax.legend(loc="lower right")
    save(fig, "all_methods_overlay.png")


def fig_dqn_loss():
    r = best_run("dqn")
    up = load("dqn", r, "updates")
    if up.empty:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    a1.plot(up["frames"] / 1e3, up["loss"], color=COLOR["dqn"], alpha=0.2, linewidth=0.6)
    a1.plot(up["frames"] / 1e3, smooth(up["loss"], 200), color=COLOR["dqn"], label="TD loss (200-update mean)")
    a1.set_yscale("log")
    a1.set_ylabel("TD loss (smooth L1, log)")
    a1.set_title(f"DQN objective · run_{r:02d}")
    a1.legend(loc="upper right")
    a2.plot(up["frames"] / 1e3, up["epsilon"], color=INK2, linewidth=1.5)
    a2.set_ylabel("ε (exploration)")
    a2.set_xlabel("environment frames (thousands)")
    save(fig, "dqn_loss.png")


def fig_pg_entropy():
    fig, ax = plt.subplots(figsize=(8, 4))
    for algo in ["reinforce", "ppo", "a2c"]:
        r = best_run(algo)
        up = load(algo, r, "updates")
        if up.empty:
            continue
        ax.plot(up["frames"] / 1e3, smooth(up["entropy"], 10), color=COLOR[algo], label=f"{LABEL[algo]} (run_{r:02d})")
    ax.axhline(np.log(9), color=INK2, linewidth=1, linestyle=":")
    ax.text(2, np.log(9) - 0.08, "uniform policy, ln 9", fontsize=8, color=INK2, va="top")
    ax.set_xlabel("environment frames (thousands)")
    ax.set_ylabel("policy entropy (nats)")
    ax.set_title("Policy entropy during training, policy-gradient methods")
    ax.legend(loc="lower left")
    save(fig, "pg_entropy.png")


def fig_convergence():
    from training.common import episodes_to_converge

    fig = plt.figure(figsize=(10, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1])
    axb = fig.add_subplot(gs[:, 2])
    vals = []
    for i, algo in enumerate(ALGOS):
        r = best_run(algo)
        ep = load(algo, r, "episodes")
        if ep.empty:
            vals.append(0)
            continue
        c = episodes_to_converge(ep["return"].to_numpy())
        vals.append(c)
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ma = smooth(ep["return"], 20)
        ax.plot(ep["episode"], ma, color=COLOR[algo])
        ax.axvline(c, color=INK, linewidth=1, linestyle="--")
        ax.set_title(f"{LABEL[algo]}: converged at ep {c}", fontsize=9)
        ax.set_xlabel("episode", fontsize=8)
        ax.set_ylabel("20-ep mean return", fontsize=8)
    axb.bar([LABEL[a] for a in ALGOS], vals, color=[COLOR[a] for a in ALGOS], width=0.6)
    for x, v in enumerate(vals):
        axb.text(x, v, f"{v}", ha="center", va="bottom", fontsize=8, color=INK)
    axb.set_ylabel("episodes to converge")
    axb.set_title("Episodes to converge (best run)")
    axb.grid(axis="x", visible=False)
    save(fig, "convergence.png")


def fig_generalization():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    x = np.arange(len(ALGOS))
    w = 0.36
    tr, te, trs, tes, trr, ter = [], [], [], [], [], []
    for algo in ALGOS:
        p = mdir(algo) / "best" / "summary.json"
        s = json.loads(p.read_text()) if p.exists() else None
        tr.append(s["eval_train"]["success_rate"] * 100 if s else 0)
        te.append(s["eval_test"]["success_rate"] * 100 if s else 0)
        trr.append(s["eval_train"]["mean_return"] if s else 0)
        ter.append(s["eval_test"]["mean_return"] if s else 0)
        trs.append(s["eval_train"]["std_return"] if s else 0)
        tes.append(s["eval_test"]["std_return"] if s else 0)
    cols = [COLOR[a] for a in ALGOS]
    a1.bar(x - w / 2, tr, w, color=cols, label="train distribution")
    a1.bar(x + w / 2, te, w, color=cols, alpha=0.45, hatch="///", edgecolor="white", label="unseen test")
    a1.set_ylabel("success rate (%)  · 100 episodes")
    a1.set_title("Success rate: train vs. unseen")
    a2.bar(x - w / 2, trr, w, yerr=trs, color=cols, capsize=3, error_kw={"elinewidth": 1, "ecolor": INK2})
    a2.bar(x + w / 2, ter, w, yerr=tes, color=cols, alpha=0.45, hatch="///", edgecolor="white", capsize=3,
           error_kw={"elinewidth": 1, "ecolor": INK2})
    a2.set_ylabel("mean return ± std")
    a2.set_title("Mean return: train vs. unseen")
    for ax in (a1, a2):
        ax.set_xticks(x, [LABEL[a] for a in ALGOS])
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color=INK2, linewidth=0.8)
    a1.legend(loc="upper right", fontsize=8)
    save(fig, "generalization.png")


def fig_sweeps():
    for algo in ALGOS:
        rows = summaries(algo)
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.2))
        runs = [f"{r['run']:02d}" for r in rows]
        m = [r["eval_train"]["mean_return"] for r in rows]
        s = [r["eval_train"]["std_return"] for r in rows]
        b = best_run(algo)
        cols = [COLOR[algo] if r["run"] == b else "#b9b8b2" for r in rows]
        ax.bar(runs, m, yerr=s, color=cols, capsize=2, error_kw={"elinewidth": 0.8, "ecolor": INK2}, width=0.65)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xlabel("run")
        ax.set_ylabel("final eval return (100 ep)")
        ax.set_title(f"{LABEL[algo]} hyperparameter sweep (best run highlighted)")
        ax.grid(axis="x", visible=False)
        save(fig, f"sweep_{algo}.png")


# -------------------------------------------------------------------- tables
COLUMNS = {
    "dqn": ["lr", "gamma", "buffer", "batch", "eps_end", "eps_decay_frac", "target", "tau", "hard_every", "double", "width", "utd"],
    "reinforce": ["lr", "gamma", "episodes_per_update", "baseline", "entropy_coef", "normalize", "width"],
    "ppo": ["lr", "gamma", "lmbda", "clip", "frames_per_batch", "epochs", "minibatch", "entropy_coef", "critic_coef", "grad_clip"],
    "a2c": ["actor_lr", "critic_lr", "gamma", "n_steps", "lmbda", "entropy_coef", "critic_coef", "shared"],
}


def tables():
    lines = ["# Results tables (generated by analysis/make_plots.py)\n"]
    for algo in ALGOS:
        rows = summaries(algo)
        if not rows:
            continue
        cols = COLUMNS[algo]
        lines.append(f"\n## {LABEL[algo]}\n")
        lines.append("| run | " + " | ".join(cols) + " | eval mean ± std | success % | test success % | converge ep |")
        lines.append("|" + "---|" * (len(cols) + 5))
        for r in rows:
            c = r["config"]
            et, es = r["eval_train"], r["eval_test"]
            lines.append(f"| {r['run']:02d} | " + " | ".join(str(c.get(k, "")) for k in cols)
                         + f" | {et['mean_return']:.2f} ± {et['std_return']:.2f} | {et['success_rate'] * 100:.0f} "
                         f"| {es['success_rate'] * 100:.0f} | {r['episodes_to_converge']} |")
    lines.append("\n## Best run per algorithm (100 episodes per split)\n")
    lines.append("| algorithm | run | train return | train success % | train planes | test return | test success % | test planes | mean steps (train) | capture quality (train) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for algo in ALGOS:
        p = mdir(algo) / "best" / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        t, u = s["eval_train"], s["eval_test"]
        lines.append(f"| {LABEL[algo]} | run_{s['run']:02d} | {t['mean_return']:.2f} ± {t['std_return']:.2f} | {t['success_rate'] * 100:.0f} "
                     f"| {t['mean_planes']:.2f} | {u['mean_return']:.2f} ± {u['std_return']:.2f} | {u['success_rate'] * 100:.0f} "
                     f"| {u['mean_planes']:.2f} | {t['mean_steps']:.1f} | {t['mean_capture_quality']:.2f} |")
    (OUT / "results_tables.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "results_tables.md")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_cumulative_rewards()
    fig_overlay()
    fig_dqn_loss()
    fig_pg_entropy()
    fig_convergence()
    fig_generalization()
    fig_sweeps()
    tables()


if __name__ == "__main__":
    main()
