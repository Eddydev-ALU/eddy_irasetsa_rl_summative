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
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
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
    fig.suptitle("Cumulative reward per episode, best run of each method", fontweight="bold")
    save(fig, "cumulative_rewards.png")


def fig_overlay():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ends = []
    for algo in ALGOS:
        r = best_run(algo)
        ev = load(algo, r, "eval")
        if ev.empty:
            continue
        y = smooth(ev["mean_return"], 3)
        ax.plot(ev["frames"] / 1e3, y, color=COLOR[algo], label=f"{LABEL[algo]} (run_{r:02d})")
        ends.append([y[-1], ev["frames"].iloc[-1] / 1e3, LABEL[algo]])
    # direct labels at line ends, nudged apart so they never overlap
    ends.sort()
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.045 * span)
    for y, x, name in ends:
        ax.annotate(name, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
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
        learned = json.loads((mdir(algo) / "best" / "summary.json").read_text())["eval_train"]["success_rate"] > 0.5
        ax.set_title(f"{LABEL[algo]}: converged at ep {c}" if learned else f"{LABEL[algo]}: plateau at ep {c} (task not learned)", fontsize=9)
        ax.set_xlabel("episode", fontsize=8)
        ax.set_ylabel("20-ep mean return", fontsize=8)
    learned = [json.loads((mdir(a) / "best" / "summary.json").read_text())["eval_train"]["success_rate"] > 0.5 for a in ALGOS]
    axb.bar([LABEL[a] for a in ALGOS], vals, color=[COLOR[a] for a in ALGOS], width=0.6,
            hatch=["" if l else "///" for l in learned], edgecolor="white")
    for x, v in enumerate(vals):
        axb.text(x, v, f"{v}" if learned[x] else f"{v}\n(not learned)", ha="center", va="bottom", fontsize=8, color=INK)
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
    a1.set_ylim(0, 122)
    a1.legend(loc="upper center", fontsize=8, ncol=2)
    for x, (t, u) in enumerate(zip(tr, te)):
        if t == 0 and u == 0:
            a1.text(x, 2, "0 %\n(no plane\ncaptured)", ha="center", va="bottom", fontsize=7.5, color=INK2)
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


def fig_seeds():
    p = ROOT / "logs" / "seeds" / "seed_check.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, algo in enumerate(ALGOS):
        if algo not in d:
            continue
        rows = d[algo]["per_seed"]
        tr = [r["train"]["mean_return"] for r in rows]
        te = [r["test"]["mean_return"] for r in rows]
        jit = np.linspace(-0.05, 0.05, len(rows))             # separate seeds with near-equal scores
        ax.scatter(i - 0.14 + jit, tr, s=44, color=COLOR[algo], edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter(i + 0.14 + jit, te, s=44, marker="D", color=COLOR[algo], alpha=0.55, edgecolor="white",
                   linewidth=1.5, zorder=3)
        ax.text(i, max(tr + te) + 4, f"{d[algo]['runs_that_learned']}/{len(rows)} learned", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(range(len(ALGOS)), [f"{LABEL[a]}\n({d[a]['config_from']})" if a in d else LABEL[a] for a in ALGOS])
    ax.set_ylabel("eval return (100 ep)")
    ax.set_title("Seed robustness: best config of each method retrained with 3 new seeds", pad=12)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(top=ax.get_ylim()[1] + 8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, label="train split (circle)"),
                       Line2D([], [], marker="D", ls="", color=INK2, alpha=0.55, label="unseen test split (diamond)")],
              loc="center right", fontsize=8)
    save(fig, "seed_robustness.png")


# -------------------------------------------------------------------- tables
COLUMNS = {
    "dqn": ["lr", "gamma", "buffer", "batch", "eps_end", "eps_decay_frac", "target", "tau", "hard_every", "double", "width", "utd"],
    "reinforce": ["lr", "gamma", "episodes_per_update", "baseline", "entropy_coef", "normalize", "width"],
    "ppo": ["lr", "gamma", "lmbda", "clip", "frames_per_batch", "epochs", "minibatch", "entropy_coef", "critic_coef", "grad_clip"],
    "a2c": ["actor_lr", "critic_lr", "gamma", "n_steps", "lmbda", "entropy_coef", "critic_coef", "shared", "normalize_adv"],
}


def conv(algo, run):
    from training.common import episodes_to_converge

    ep = load(algo, run, "episodes")
    return episodes_to_converge(ep["return"].to_numpy()) if not ep.empty else ""


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
                         f"| {es['success_rate'] * 100:.0f} | {conv(algo, r['run'])} |")
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
    p = ROOT / "logs" / "seeds" / "seed_check.json"
    if p.exists():
        d = json.loads(p.read_text())
        lines.append("\n## Seed robustness (best config, 3 fresh seeds)\n")
        lines.append("| algorithm | config | seeds learned | train return | train success % | test return | test success % |")
        lines.append("|---|---|---|---|---|---|---|")
        for algo in ALGOS:
            if algo in d:
                v = d[algo]
                lines.append(f"| {LABEL[algo]} | {v['config_from']} | {v['runs_that_learned']}/{len(v['seeds'])} "
                             f"| {v['train_return_mean']:.2f} ± {v['train_return_std']:.2f} | {v['train_success_mean'] * 100:.0f} "
                             f"| {v['test_return_mean']:.2f} ± {v['test_return_std']:.2f} | {v['test_success_mean'] * 100:.0f} |")
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
    fig_seeds()
    tables()


if __name__ == "__main__":
    main()
