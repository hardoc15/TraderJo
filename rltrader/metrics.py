from __future__ import annotations

import numpy as np

from .config import TRADING_DAYS


def compute_metrics(daily_returns: np.ndarray, turnover: np.ndarray | None = None) -> dict:
    r = np.asarray(daily_returns, dtype=float)
    equity = np.cumprod(1 + r)
    years = len(r) / TRADING_DAYS
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    downside = r[r < 0].std(ddof=1) * np.sqrt(TRADING_DAYS) if (r < 0).sum() > 1 else np.nan
    peak = np.maximum.accumulate(equity)
    out = {
        "total_return": equity[-1] - 1,
        "cagr": equity[-1] ** (1 / years) - 1,
        "ann_vol": vol,
        "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS),   # risk-free rate = 0
        "sortino": r.mean() * TRADING_DAYS / downside if downside else np.nan,
        "max_drawdown": float((equity / peak - 1).min()),
    }
    out["calmar"] = out["cagr"] / abs(out["max_drawdown"]) if out["max_drawdown"] < 0 else np.nan
    if turnover is not None:
        out["avg_daily_turnover"] = float(np.mean(turnover))
    return out
