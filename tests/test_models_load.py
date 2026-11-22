from pathlib import Path

import pytest

from environment.custom_env import FetalScanNavEnv
from training.common import load_policy

ROOT = Path(__file__).resolve().parent.parent
BEST = [ROOT / "models/dqn/best", ROOT / "models/pg/reinforce/best", ROOT / "models/pg/ppo/best", ROOT / "models/pg/a2c/best"]


@pytest.mark.parametrize("d", BEST, ids=lambda p: p.parent.name)
def test_best_model_loads_and_acts(d):
    if not (d / "model.pt").exists():
        pytest.skip("model not trained yet")
    act, net, ckpt = load_policy(d / "model.pt")
    env = FetalScanNavEnv(split="test")
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        a = act(obs)
        assert env.action_space.contains(a)
        obs, *_ = env.step(a)


def test_api_policy_act():
    from api.server import Policy
    from main import pick_model

    if pick_model(None)[1] is None:
        pytest.skip("model not trained yet")
    p = Policy()
    out = p.act([0.5] * p.obs_dim)
    assert 0 <= out["action_id"] < 9 and abs(sum(out["action_probs"]) - 1) < 1e-3
