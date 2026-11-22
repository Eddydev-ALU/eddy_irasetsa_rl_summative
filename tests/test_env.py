import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.custom_env import CAPTURE, FetalScanNavEnv


def test_check_env():
    check_env(FetalScanNavEnv())


def test_estimate_gated_by_visibility():
    env = FetalScanNavEnv()
    obs, _ = env.reset(seed=7)
    env.u, env.v = 0.03, 0.5          # far lateral: no anatomy in view
    a, vis = env._alignment()
    env.q, env.probs = env._image_signals(a, vis)
    assert np.all(env._obs()[16:] == 0)


def test_obs_in_space_and_seeding():
    a, b = FetalScanNavEnv(), FetalScanNavEnv()
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    assert np.array_equal(oa, ob)
    rng = np.random.default_rng(0)
    for _ in range(300):
        act = int(rng.integers(9))
        oa, ra, ta, tra, _ = a.step(act)
        ob, rb, tb, trb, _ = b.step(act)
        assert a.observation_space.contains(oa)
        assert np.array_equal(oa, ob) and ra == rb
        if ta or tra:
            break


def _place_on(env, k):
    env.u, env.v, env.theta, env.phi = env.targets[k]
    a, vis = env._alignment()
    env.q, env.probs = env._image_signals(a, vis)
    env._phi_s = env._potential(a, vis)


def test_correct_then_duplicate_capture():
    env = FetalScanNavEnv()
    env.reset(seed=1)
    _place_on(env, 0)
    _, r, term, _, info = env.step(CAPTURE)
    assert info["event"][0] == "capture" and info["planes_captured"] == 1 and not term
    assert r > 5
    _place_on(env, 0)
    _, r, _, _, info = env.step(CAPTURE)
    assert info["event"][0] == "bad_capture" and info["planes_captured"] == 1
    assert r < -0.5


def test_low_quality_capture_penalised():
    env = FetalScanNavEnv()
    env.reset(seed=2)
    env.u, env.v = 0.5, 0.95  # far from every target
    a, vis = env._alignment()
    env.q, env.probs = env._image_signals(a, vis)
    _, r, _, _, info = env.step(CAPTURE)
    assert info["event"][0] == "bad_capture" and r < -0.5


def test_success_terminates_with_bonus():
    env = FetalScanNavEnv()
    env.reset(seed=4)
    for k in range(3):
        _place_on(env, k)
        _, r, term, _, info = env.step(CAPTURE)
    assert term and info["success"] and r > 20


def test_exit_terminates():
    env = FetalScanNavEnv()
    env.reset(seed=5)
    for _ in range(40):
        _, r, term, _, info = env.step(0)
        if term:
            break
    assert term and info["event"] == ("exit",) and r <= -4


def test_truncation_at_max_steps():
    env = FetalScanNavEnv()
    env.reset(seed=6)
    for i in range(env.cfg.max_steps):
        _, _, term, trunc, _ = env.step(4 if i % 2 else 5)
    assert trunc and not term


@pytest.mark.parametrize("split,pres", [("train", {"cephalic", "breech"}), ("test", {"transverse"})])
def test_splits(split, pres):
    env = FetalScanNavEnv()
    seen = {env.reset(seed=s, options={"split": split})[1]["presentation"] for s in range(20)}
    assert seen <= pres
