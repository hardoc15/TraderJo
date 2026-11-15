"""Single source of truth for portfolio accounting (used by the RL env,
the baselines, the backtester and the paper trader)."""
from __future__ import annotations

import numpy as np


def rebalance_step(value: float, w_current: np.ndarray, w_target: np.ndarray,
                   asset_returns: np.ndarray, cost: float):
    """Rebalance from w_current to w_target, then hold for one day.

    Weights have length N+1 (N assets + cash, cash last). Cash earns 0.
    Returns (new_value, drifted_weights, turnover, net_return).
    """
    turnover = float(np.abs(w_target[:-1] - w_current[:-1]).sum())
    after_cost = 1.0 - cost * turnover
    gross = 1.0 + float(np.dot(w_target[:-1], asset_returns))
    growth = np.append(1.0 + asset_returns, 1.0) * w_target
    drifted = growth / growth.sum()
    net = after_cost * gross - 1.0
    return value * (1.0 + net), drifted, turnover, net
