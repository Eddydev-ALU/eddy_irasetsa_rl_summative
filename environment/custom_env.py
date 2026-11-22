"""FetalScanNav-v0: RL probe guidance towards standard fetal ultrasound planes.

A midwife holds an ultrasound probe on a pregnant abdomen and must capture three
standard planes at good quality: the head plane (HC), the abdominal plane (AC)
and the femur plane (FL). The fetus is hidden: its presentation, rotation and
offset are random each episode. The agent only sees what a real scanning app
would have: probe pose readings and the output of an image CNN.

Action space: Discrete(9)
    0 slide cranial (v += d)      1 slide caudal (v -= d)
    2 slide left    (u -= d)      3 slide right  (u += d)
    4 rotate clockwise (th += dth) 5 rotate anticlockwise (th -= dth)
    6 tilt forward  (phi += dphi) 7 tilt backward (phi -= dphi)
    8 freeze & capture (saves frame as the most likely plane)

Observation space: Box(float32, (16,))
    idx   name                    source (real world)               range     update
    0-1   u, v                    IMU + camera / EM tracking        [0, 1]    every step
    2-3   sin(2th), cos(2th)      IMU gyroscope (plane orientation,  [-1, 1]   every step
                                  180-degree symmetric)
    4     phi / phi_max           IMU accelerometer                 [-1, 1]   every step
    5     q (image quality)       CNN on live B-mode frame          [0, 1]    every step
    6     dq (quality change)     derived in app                    [-1, 1]   every step
    7-10  p_HC, p_AC, p_FL, p_none plane classifier (capstone CNN)  [0, 1]    every step
    11-13 cap_HC, cap_AC, cap_FL  app state                         {0, 1}    on capture
    14    t_left                  app timer                         [0, 1]    every step
    15    motion                  frame differencing on B-mode      {0, 1}    every step

Reward (all coefficients in EnvConfig):
    r = -0.05                                   time cost per step
        + 0.5 * (gamma * Phi(s') - Phi(s))      potential-based shaping (Ng et al. 1999),
                                                Phi = best alignment to an uncaptured plane.
                                                Being potential-based, it cannot change
                                                the optimal policy.
        + 10 * q    correct capture of an uncaptured plane with q >= 0.70
        - 3         wrong / duplicate / low-quality capture
        - 5         probe leaves the scannable region (terminates)
        + 20 + 5 * t_left   all three planes captured (terminates)

Start: probe within 0.1 of the abdomen centre, random rotation/tilt, nothing captured.
Terminated: all planes captured (success) or probe leaves region (failure).
Truncated: max_steps reached.

Splits: reset(options={"split": "test"}) samples the unseen distribution
(transverse lie, larger offsets, more fetal movement).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PLANES = ("HC", "AC", "FL")
ACTION_NAMES = (
    "slide cranial",
    "slide caudal",
    "slide left",
    "slide right",
    "rotate cw",
    "rotate ccw",
    "tilt forward",
    "tilt backward",
    "capture",
)
GUIDANCE_TEXT = (
    "Move probe up (towards head)",
    "Move probe down (towards feet)",
    "Move probe left",
    "Move probe right",
    "Rotate probe clockwise",
    "Rotate probe anticlockwise",
    "Fan probe towards head",
    "Fan probe towards feet",
    "Hold still - freeze and save image",
)
OBS_NAMES = (
    "u", "v", "sin_2theta", "cos_2theta", "tilt", "quality", "d_quality",
    "p_HC", "p_AC", "p_FL", "p_none", "cap_HC", "cap_AC", "cap_FL",
    "t_left", "motion",
)
CAPTURE = 8


@dataclass
class EnvConfig:
    max_steps: int = 200
    # actuation
    slide_step: float = 0.025
    rot_step: float = np.deg2rad(10.0)
    tilt_step: float = np.deg2rad(5.0)
    tilt_max: float = np.deg2rad(30.0)
    slide_noise: float = 0.10          # relative noise on slide magnitude
    rot_noise: float = np.deg2rad(1.5)
    region_radius: float = 0.5         # scannable disc in (u, v)
    start_radius: float = 0.10
    # image simulation
    sigma_pos: float = 0.07
    sigma_rot: float = np.deg2rad(25.0)
    sigma_tilt: float = np.deg2rad(12.0)
    target_tilt_range: float = np.deg2rad(8.0)
    sigma_vis: float = 0.18            # how far away the anatomy is still recognisable
    offset_scale: float = 0.12         # detector offset of 1.0 = anatomy 0.12 away in-plane
    offset_noise: float = 0.08
    obs_offset: bool = True
    obs_last_action: bool = False
    obs_privileged: bool = False       # debugging only: true relative pose of the nearest open target
    obs_estimate: bool = False         # CNN plane-localisation estimate (noisy, gated by visibility)
    est_min_vis: float = 0.15
    est_pos_noise: float = 0.02
    est_pos_noise_dist: float = 0.15
    est_rot_noise: float = np.deg2rad(10.0)
    est_tilt_noise: float = np.deg2rad(4.0)
    speckle_noise: float = 0.015
    edge_noise: float = 0.05           # extra noise at the abdomen edge (shadowing)
    classifier_temp: float = 6.0
    classifier_noise: float = 0.25
    # fetus
    max_offset: float = 0.15
    p_move_train: float = 0.01
    p_move_test: float = 0.03
    move_std: float = 0.03
    # reward
    time_cost: float = 0.05
    shaping_coef: float = 0.5
    shaping_gamma: float = 0.99
    capture_coef: float = 10.0
    capture_threshold: float = 0.70
    bad_capture_penalty: float = 3.0
    exit_penalty: float = 5.0
    success_bonus: float = 20.0
    success_time_bonus: float = 5.0

    def to_dict(self):
        return asdict(self)


def _wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _plane_angle_dist(a, b):
    """An image plane is the same when the probe is turned 180 degrees."""
    return np.abs((a - b + np.pi / 2) % np.pi - np.pi / 2)


class FetalScanNavEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 8}

    def __init__(self, config: EnvConfig | None = None, split: str = "train", render_mode=None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.split = split
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(9)
        low = [0, 0, -1, -1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if self.cfg.obs_offset:
            low += [-1, -1]
        if self.cfg.obs_last_action:
            low += [0] * 9
        if self.cfg.obs_privileged:
            low += [-1] * 5
        if self.cfg.obs_estimate:
            low += [-1, -1, -1, -1, -1, 0]
        low = np.array(low, dtype=np.float32)
        high = np.ones_like(low)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    # ------------------------------------------------------------------ fetus
    def _sample_fetus(self, split):
        rng = self.np_random
        if split == "test":
            presentation = "transverse"
            base = rng.choice([0.0, np.pi])
            r = rng.uniform(0.6, 1.0)
        else:
            presentation = str(rng.choice(["cephalic", "breech"]))
            base = -np.pi / 2 if presentation == "cephalic" else np.pi / 2
            r = rng.uniform(0.0, 0.6)
        ang = rng.uniform(0, 2 * np.pi)
        offset = r * self.cfg.max_offset * np.array([np.cos(ang), np.sin(ang)])
        self.presentation = presentation
        self.fetal_axis = base + rng.normal(0, np.deg2rad(12))   # direction pointing to fetal head
        self.fetal_centre = np.array([0.5, 0.5]) + offset
        self.femur_side = rng.choice([-1.0, 1.0])
        self.target_tilt = rng.uniform(-self.cfg.target_tilt_range, self.cfg.target_tilt_range, size=3)
        self.rot_jitter = rng.normal(0, np.deg2rad(8), size=3)
        self.femur_angle = rng.uniform(-np.deg2rad(30), np.deg2rad(30))
        self.targets = self._compute_targets(np.ones(3, bool))

    def _compute_targets(self, which):
        """Target probe pose (u, v, theta, phi) for each plane, from the fetal pose."""
        d = np.array([np.cos(self.fetal_axis), np.sin(self.fetal_axis)])
        perp = np.array([-d[1], d[0]])
        c = self.fetal_centre
        pos = [c + 0.17 * d, c - 0.02 * d, c - 0.15 * d + 0.05 * self.femur_side * perp]
        rot = [
            self.fetal_axis + np.pi / 2 + self.rot_jitter[0],   # HC: transverse to the body axis
            self.fetal_axis + np.pi / 2 + self.rot_jitter[1],   # AC: transverse to the body axis
            self.fetal_axis + self.femur_angle + self.rot_jitter[2],  # FL: along the femur
        ]
        old = getattr(self, "targets", None)
        out = np.zeros((3, 4))
        for k in range(3):
            if which[k] or old is None:
                out[k] = [pos[k][0], pos[k][1], _wrap_pi(rot[k]), self.target_tilt[k]]
            else:
                out[k] = old[k]
        return out

    # ------------------------------------------------------------- simulation
    def _alignment(self):
        cfg = self.cfg
        t = self.targets
        d_pos = np.hypot(t[:, 0] - self.u, t[:, 1] - self.v)
        d_rot = _plane_angle_dist(self.theta, t[:, 2])
        d_tilt = np.abs(self.phi - t[:, 3])
        a = np.exp(-(d_pos / cfg.sigma_pos) ** 2 - (d_rot / cfg.sigma_rot) ** 2 - (d_tilt / cfg.sigma_tilt) ** 2)
        vis = np.exp(-(d_pos / cfg.sigma_vis) ** 2)
        return a, vis

    def _potential(self, a, vis):
        open_ = ~self.captured
        if not open_.any():
            return 0.0
        return float(np.max((0.5 * vis + 0.5 * a)[open_]))

    def _image_signals(self, a, vis):
        cfg = self.cfg
        rng = self.np_random
        edge = np.clip(np.hypot(self.u - 0.5, self.v - 0.5) / cfg.region_radius, 0, 1) ** 2
        q = float(np.clip(a.max() + rng.normal(0, cfg.speckle_noise + cfg.edge_noise * edge), 0, 1))
        s = 0.5 * vis + 0.5 * a
        logits = np.append(cfg.classifier_temp * s, 0.35 * cfg.classifier_temp)
        logits = logits + rng.normal(0, cfg.classifier_noise, size=4)
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        # detector: in-plane offset of the most likely anatomy, rotated into patient (u, v) axes
        k = int(np.argmax(p))
        off = np.zeros(2)
        if k < 3:
            t = self.targets[k]
            ax = np.array([np.cos(self.theta), np.sin(self.theta)])
            lat = np.clip(((t[0] - self.u) * ax[0] + (t[1] - self.v) * ax[1]) / cfg.offset_scale
                          + rng.normal(0, cfg.offset_noise), -1, 1)
            off = lat * ax
        self.offset = off
        self.estimate = self._plane_estimate(p, vis)
        return q, p

    def _plane_estimate(self, p, vis):
        """Noisy relative pose of the most likely *uncaptured* plane, as a plane-localisation
        CNN would regress it from the live frame. Zero (valid=0) if that anatomy is not recognisable."""
        cfg, rng = self.cfg, self.np_random
        open_ = np.where(~self.captured)[0]
        if len(open_) == 0:
            return np.zeros(6)
        k = open_[np.argmax(p[open_])]
        if vis[k] < cfg.est_min_vis:
            return np.zeros(6)
        t = self.targets[k]
        dist = np.hypot(t[0] - self.u, t[1] - self.v)
        pn = cfg.est_pos_noise + cfg.est_pos_noise_dist * dist
        du = t[0] - self.u + rng.normal(0, pn)
        dv = t[1] - self.v + rng.normal(0, pn)
        dr = 2 * (t[2] - self.theta + rng.normal(0, cfg.est_rot_noise))
        dt = (t[3] - self.phi + rng.normal(0, cfg.est_tilt_noise)) / cfg.tilt_max
        return np.clip([du * 3, dv * 3, np.sin(dr), np.cos(dr), dt, 1.0], -1, 1)

    def _obs(self):
        return np.array(
            [
                self.u, self.v, np.sin(2 * self.theta), np.cos(2 * self.theta), self.phi / self.cfg.tilt_max,
                self.q, np.clip(self.dq, -1, 1), *self.probs, *self.captured.astype(float),
                1.0 - self.steps / self.cfg.max_steps, float(self.motion),
                *(self.offset if self.cfg.obs_offset else ()),
                *(np.eye(9)[self.last_action] if self.cfg.obs_last_action and self.last_action is not None
                  else np.zeros(9) if self.cfg.obs_last_action else ()),
                *(self._privileged() if self.cfg.obs_privileged else ()),
                *(self.estimate if self.cfg.obs_estimate else ()),
            ],
            dtype=np.float32,
        )

    def _privileged(self):
        open_ = np.where(~self.captured)[0]
        if len(open_) == 0:
            return np.zeros(5)
        t = self.targets
        k = open_[np.argmin(np.hypot(t[open_, 0] - self.u, t[open_, 1] - self.v))]
        dr = 2 * (t[k, 2] - self.theta)
        return np.clip([(t[k, 0] - self.u) * 3, (t[k, 1] - self.v) * 3, np.sin(dr), np.cos(dr),
                        (t[k, 3] - self.phi) / self.cfg.tilt_max], -1, 1)

    def _info(self):
        return {
            "success": np.bool_(self.captured.all()),
            "planes_captured": np.int64(self.captured.sum()),
            "capture_qualities": self.capture_q.copy(),
            "steps": self.steps,
            "presentation": self.presentation,
            "event": self.event,
        }

    # -------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        split = (options or {}).get("split", self.split)
        self.cur_split = split
        rng = self.np_random
        self._sample_fetus(split)
        r = self.cfg.start_radius * np.sqrt(rng.uniform())
        ang = rng.uniform(0, 2 * np.pi)
        self.u, self.v = 0.5 + r * np.cos(ang), 0.5 + r * np.sin(ang)
        self.theta = rng.uniform(-np.pi, np.pi)
        self.phi = rng.uniform(-self.cfg.tilt_max, self.cfg.tilt_max) * 0.5
        self.captured = np.zeros(3, bool)
        self.capture_q = np.zeros(3)
        self.steps = 0
        self.motion = False
        self.event = None
        self.last_action = None
        a, vis = self._alignment()
        self.q, self.probs = self._image_signals(a, vis)
        self.dq = 0.0
        self._phi_s = self._potential(a, vis)
        return self._obs(), self._info()

    def step(self, action):
        cfg = self.cfg
        rng = self.np_random
        action = int(action)
        self.last_action = action
        self.steps += 1
        reward = -cfg.time_cost
        terminated = False
        self.event = None

        if action < 4:
            mag = cfg.slide_step * (1 + rng.normal(0, cfg.slide_noise))
            jitter = rng.normal(0, 0.1 * cfg.slide_step)
            du, dv = [(0, mag), (0, -mag), (-mag, 0), (mag, 0)][action]
            if action < 2:
                du += jitter
            else:
                dv += jitter
            self.u += du
            self.v += dv
        elif action < 6:
            sign = 1 if action == 4 else -1
            self.theta = _wrap_pi(self.theta + sign * cfg.rot_step + rng.normal(0, cfg.rot_noise))
        elif action < 8:
            sign = 1 if action == 6 else -1
            self.phi = float(np.clip(self.phi + sign * cfg.tilt_step, -cfg.tilt_max, cfg.tilt_max))
        else:
            k = int(np.argmax(self.probs))
            if k < 3 and not self.captured[k] and self.q >= cfg.capture_threshold:
                reward += cfg.capture_coef * self.q
                self.captured[k] = True
                self.capture_q[k] = self.q
                self.event = ("capture", PLANES[k], self.q)
            else:
                reward -= cfg.bad_capture_penalty
                why = "no plane" if k == 3 else ("duplicate" if self.captured[k] else "low quality")
                self.event = ("bad_capture", PLANES[k] if k < 3 else "none", self.q, why)

        # fetal movement moves the targets of planes not yet captured
        p_move = cfg.p_move_test if self.cur_split == "test" else cfg.p_move_train
        self.motion = bool(rng.uniform() < p_move)
        if self.motion:
            self.fetal_centre = self.fetal_centre + rng.normal(0, cfg.move_std, size=2)
            self.fetal_centre = np.clip(self.fetal_centre, 0.25, 0.75)
            self.targets = self._compute_targets(~self.captured)

        if np.hypot(self.u - 0.5, self.v - 0.5) > cfg.region_radius:
            reward -= cfg.exit_penalty
            terminated = True
            self.u, self.v = float(np.clip(self.u, 0, 1)), float(np.clip(self.v, 0, 1))
            self.event = ("exit",)

        a, vis = self._alignment()
        q_prev = self.q
        self.q, self.probs = self._image_signals(a, vis)
        self.dq = self.q - q_prev

        if self.captured.all():
            reward += cfg.success_bonus + cfg.success_time_bonus * (1.0 - self.steps / cfg.max_steps)
            terminated = True

        phi_next = 0.0 if terminated else self._potential(a, vis)
        reward += cfg.shaping_coef * (cfg.shaping_gamma * phi_next - self._phi_s)
        self._phi_s = phi_next

        truncated = (not terminated) and self.steps >= cfg.max_steps
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ----------------------------------------------------------- rendering
    def render_state(self):
        """Full simulator state (incl. hidden ground truth) for the three.js view."""
        a, vis = self._alignment()
        return {
            "probe": {"u": float(self.u), "v": float(self.v), "theta": float(self.theta), "phi": float(self.phi)},
            "fetus": {
                "presentation": self.presentation,
                "axis": float(self.fetal_axis),
                "centre": [float(x) for x in self.fetal_centre],
                "femur_side": float(self.femur_side),
            },
            "targets": {PLANES[k]: [float(x) for x in self.targets[k]] for k in range(3)},
            "alignment": {PLANES[k]: float(a[k]) for k in range(3)},
            "visibility": {PLANES[k]: float(vis[k]) for k in range(3)},
            "offset": [float(x) for x in self.offset],
            "quality": float(self.q),
            "probs": {n: float(p) for n, p in zip((*PLANES, "none"), self.probs)},
            "captured": {PLANES[k]: bool(self.captured[k]) for k in range(3)},
            "capture_q": {PLANES[k]: float(self.capture_q[k]) for k in range(3)},
            "steps": int(self.steps),
            "max_steps": int(self.cfg.max_steps),
            "motion": bool(self.motion),
            "split": self.cur_split,
        }
