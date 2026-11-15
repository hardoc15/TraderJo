"""Paper trading with the deployed agent.

The simulated account lives in data/paper_account.json. Each `step` uses only
prices up to the latest close, asks the agent for target weights, and rebalances
the account at that close (same execution assumption as the backtest), paying
the same 10 bps transaction cost.

Optionally the same target weights can be sent to an Alpaca *paper* account
(https://alpaca.markets) by setting ALPACA_API_KEY / ALPACA_SECRET_KEY.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DATA_DIR, MODELS_DIR, TICKERS, TRANSACTION_COST
from .data import load_prices, make_market_data
from .env import action_to_weights

ACCOUNT_FILE = DATA_DIR / "paper_account.json"
START_CASH = 100_000.0


def load_agent():
    from stable_baselines3 import PPO
    meta = json.load(open(MODELS_DIR / "deployed" / "meta.json"))
    model = PPO.load(MODELS_DIR / "deployed" / "model.zip", device="cpu")
    return model, meta


def new_account(start_date: str | None = None) -> dict:
    return {"created": datetime.now().isoformat(timespec="seconds"), "start_date": start_date,
            "cash": START_CASH, "shares": {t: 0.0 for t in TICKERS}, "history": [], "trades": []}


def load_account() -> dict | None:
    return json.load(open(ACCOUNT_FILE)) if ACCOUNT_FILE.exists() else None


def save_account(acct: dict):
    ACCOUNT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(acct, open(ACCOUNT_FILE, "w"), indent=1)


def _equity(acct, px):
    return acct["cash"] + sum(acct["shares"][t] * px[t] for t in TICKERS)


def _weights(acct, px):
    eq = _equity(acct, px)
    return np.array([acct["shares"][t] * px[t] / eq for t in TICKERS] + [acct["cash"] / eq])


def target_weights(model, md, t: int, w_current: np.ndarray) -> np.ndarray:
    obs = np.concatenate([md.features[t], w_current.astype(np.float32)])
    action, _ = model.predict(obs, deterministic=True)
    return action_to_weights(action)


def rebalance(acct: dict, date, px: pd.Series, target: np.ndarray, cost=TRANSACTION_COST):
    eq = _equity(acct, px)
    cur_val = np.array([acct["shares"][t] * px[t] for t in TICKERS])
    traded = np.abs(target[:-1] * eq - cur_val).sum()
    fee = cost * traded
    eq_after = eq - fee
    for i, t in enumerate(TICKERS):
        new_sh = target[i] * eq_after / px[t]
        delta = new_sh - acct["shares"][t]
        if abs(delta * px[t]) > 1.0:
            acct["trades"].append({"date": str(date.date()), "ticker": t,
                                   "side": "BUY" if delta > 0 else "SELL",
                                   "shares": round(abs(delta), 4), "price": round(float(px[t]), 2),
                                   "notional": round(abs(delta) * float(px[t]), 2)})
        acct["shares"][t] = float(new_sh)
    acct["cash"] = float(target[-1] * eq_after)
    acct["history"].append({"date": str(date.date()), "equity": round(eq_after, 2),
                            "fee": round(fee, 2),
                            "weights": {k: round(float(v), 4) for k, v in zip(TICKERS + ["CASH"], target)}})


def step(refresh: bool = True, replay_from: str | None = None, model=None, meta=None) -> dict:
    """Run the paper trader up to the latest close. If replay_from is given, start a
    fresh account on that date and walk forward one trading day at a time."""
    if model is None:
        model, meta = load_agent()
    prices = load_prices(refresh=refresh)
    md = make_market_data(prices, stats=(meta["feat_mean"], meta["feat_std"]))
    acct = new_account(replay_from) if replay_from else (load_account() or new_account())
    done = {h["date"] for h in acct["history"]}
    first = pd.Timestamp(replay_from) if replay_from else md.dates[-1]
    for t in np.flatnonzero(md.dates >= first):
        date = md.dates[t]
        if str(date.date()) in done:
            continue
        px = md.prices.iloc[t]
        w = _weights(acct, px)
        rebalance(acct, date, px, target_weights(model, md, t, w))
    save_account(acct)
    return acct


def account_summary(acct: dict, prices: pd.DataFrame | None = None) -> dict:
    prices = prices if prices is not None else load_prices()
    px = prices.iloc[-1]
    eq = _equity(acct, px)
    return {"equity": eq, "return": eq / START_CASH - 1, "as_of": str(prices.index[-1].date()),
            "positions": [{"ticker": t, "shares": acct["shares"][t], "price": float(px[t]),
                           "value": acct["shares"][t] * float(px[t])} for t in TICKERS],
            "cash": acct["cash"]}


# ------------------------------------------------------------------ Alpaca (optional)
ALPACA_URL = "https://paper-api.alpaca.markets"


def alpaca_sync(target: dict, dry_run: bool = True, buffer: float = 0.98) -> list[dict]:
    """Move an Alpaca PAPER account toward the target weights using notional market
    orders (fractional shares). Sells are sent before buys. Returns the order list."""
    import requests

    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (paper account keys).")
    h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    acct = requests.get(f"{ALPACA_URL}/v2/account", headers=h, timeout=10)
    acct.raise_for_status()
    equity = float(acct.json()["equity"]) * buffer
    pos = requests.get(f"{ALPACA_URL}/v2/positions", headers=h, timeout=10)
    pos.raise_for_status()
    held = {p["symbol"]: float(p["market_value"]) for p in pos.json()}
    orders = []
    for t in TICKERS:
        diff = target[t] * equity - held.get(t, 0.0)
        if abs(diff) >= 1.0:
            orders.append({"symbol": t, "notional": f"{abs(diff):.2f}",
                           "side": "buy" if diff > 0 else "sell",
                           "type": "market", "time_in_force": "day"})
    orders.sort(key=lambda o: o["side"] != "sell")
    if not dry_run:
        for o in orders:
            r = requests.post(f"{ALPACA_URL}/v2/orders", headers=h, json=o, timeout=10)
            o["status"] = r.status_code
            o["response"] = r.json()
    return orders
