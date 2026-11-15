"""Backtester and baseline strategies.

A policy is a callable  policy(t, w_current, md) -> target weights (length N+1, cash last),
called at the close of decision day t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRANSACTION_COST
from .env import action_to_weights
from .portfolio import rebalance_step


def run_backtest(policy, md, lo: int, hi: int, cost: float = TRANSACTION_COST) -> pd.DataFrame:
    n = md.fwd_returns.shape[1]
    w = np.zeros(n + 1)
    w[-1] = 1.0
    value, rows = 1.0, []
    for t in range(lo, hi):
        target = np.asarray(policy(t, w, md), dtype=float)
        target = target / target.sum()
        value, w, turnover, net = rebalance_step(value, w, target, md.fwd_returns[t], cost)
        # The return is realised at the close of t+1, so label the row with that date.
        rows.append({"date": md.dates[t + 1], "value": value, "net_return": net,
                     "turnover": turnover, **{f"w_{c}": x for c, x in
                                              zip(list(md.prices.columns) + ["CASH"], target)}})
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------- baselines
def _fixed_weights(weights, monthly=True):
    weights = np.asarray(weights, dtype=float)

    def policy(t, w, md):
        new_month = t == 0 or md.dates[t].month != md.dates[t - 1].month
        if w[-1] == 1.0 or (monthly and new_month):
            return weights
        return w                      # let weights drift, no trading
    return policy


def buy_and_hold(md, ticker="SPY"):
    cols = list(md.prices.columns)
    w = np.zeros(len(cols) + 1)
    w[cols.index(ticker)] = 1.0
    return _fixed_weights(w, monthly=False)


def equal_weight(md):
    n = md.prices.shape[1]
    return _fixed_weights(np.append(np.full(n, 1 / n), 0.0))


def sixty_forty(md):
    cols = list(md.prices.columns)
    w = np.zeros(len(cols) + 1)
    w[cols.index("SPY")], w[cols.index("TLT")] = 0.6, 0.4
    return _fixed_weights(w)


def sma200_trend(md):
    """Hold each asset (1/N slice) only while it trades above its 200-day average."""
    cols = list(md.prices.columns)
    gaps = md.raw_features[[f"{c}_sma200_gap" for c in cols]].values
    state = {"signal": None}

    def policy(t, w, md):
        sig = gaps[t] > 0
        if state["signal"] is not None and np.array_equal(sig, state["signal"]) and w[-1] != 1.0:
            return w
        state["signal"] = sig
        target = np.append(sig / len(cols), 1 - sig.sum() / len(cols))
        return target
    return policy


def momentum_top2(md):
    """Monthly: hold the two assets with the best 60-day return, if positive."""
    cols = list(md.prices.columns)
    mom = md.raw_features[[f"{c}_ret60" for c in cols]].values

    def policy(t, w, md):
        new_month = md.dates[t].month != md.dates[t - 1].month
        if not (new_month or w[-1] == 1.0):
            return w
        target = np.zeros(len(cols) + 1)
        for i in np.argsort(-mom[t])[:2]:
            target[i if mom[t, i] > 0 else -1] += 0.5
        return target
    return policy


def random_policy(seed=0):
    """Uniform random actions through the same softmax as the RL agent."""
    rng = np.random.default_rng(seed)

    def policy(t, w, md):
        return action_to_weights(rng.uniform(-1, 1, md.prices.shape[1] + 1))
    return policy


def agent_weights(models, obs) -> np.ndarray:
    """Target weights from one model or the average over an ensemble of models."""
    models = models if isinstance(models, (list, tuple)) else [models]
    return np.mean([action_to_weights(m.predict(obs, deterministic=True)[0]) for m in models], axis=0)


def ppo_policy(models, rebalance_every: int = 1):
    """Query the agent(s) every `rebalance_every` days; hold (drift) in between."""
    state = {"n": 0}

    def policy(t, w, md):
        n = state["n"]
        state["n"] += 1
        if n % rebalance_every and w[-1] != 1.0:
            return w
        obs = np.concatenate([md.features[t], w.astype(np.float32)])
        return agent_weights(models, obs)
    return policy


BASELINES = {
    "Buy & hold SPY": buy_and_hold,
    "60/40 SPY/TLT": sixty_forty,
    "Equal weight (monthly)": equal_weight,
    "SMA-200 trend": sma200_trend,
    "Momentum top-2": momentum_top2,
}
