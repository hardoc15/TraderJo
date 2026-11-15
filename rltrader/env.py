"""Gymnasium environment: daily long-only portfolio allocation over N ETFs + cash.

Observation: standardized market features at close t, plus current (drifted) weights.
Action:      N+1 real numbers in [-1, 1] mapped to portfolio weights by a softmax.
Reward:      100 * log net portfolio return (after transaction costs), optionally
             minus a variance penalty (mean-variance utility).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .portfolio import rebalance_step

ACTION_SCALE = 3.0


def action_to_weights(action: np.ndarray) -> np.ndarray:
    a = ACTION_SCALE * np.clip(np.asarray(action, dtype=np.float64), -1, 1)
    e = np.exp(a - a.max())
    return e / e.sum()


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, md, start_idx: int, end_idx: int, episode_len: int | None = 504,
                 cost: float = 0.001, reward_mode: str = "logret", risk_lambda: float = 0.02):
        super().__init__()
        self.md, self.lo, self.hi = md, start_idx, end_idx
        self.episode_len, self.cost = episode_len, cost
        self.reward_mode, self.risk_lambda = reward_mode, risk_lambda
        self.n = md.fwd_returns.shape[1]
        obs_dim = md.features.shape[1] + self.n + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n + 1,), np.float32)

    def _obs(self):
        return np.concatenate([self.md.features[self.t], self.w.astype(np.float32)])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.episode_len is None or self.hi - self.lo <= self.episode_len:
            self.t, self.end = self.lo, self.hi
        else:
            self.t = int(self.np_random.integers(self.lo, self.hi - self.episode_len))
            self.end = self.t + self.episode_len
        self.w = np.zeros(self.n + 1)
        self.w[-1] = 1.0                       # start fully in cash
        self.value = 1.0
        return self._obs(), {}

    def step(self, action):
        target = action_to_weights(action)
        r = self.md.fwd_returns[self.t]
        self.value, self.w, turnover, net = rebalance_step(self.value, self.w, target, r, self.cost)
        reward = 100.0 * np.log1p(net)
        if self.reward_mode == "risk":
            reward -= self.risk_lambda * (100.0 * net) ** 2
        self.t += 1
        terminated = self.t >= self.end
        info = {"value": self.value, "turnover": turnover, "net_return": net, "weights": target}
        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(reward), terminated, False, info
