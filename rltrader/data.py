"""Download daily prices and build leakage-free features.

Alignment convention: row t of the feature matrix only uses closing prices up to
and including day t. `fwd_returns[t]` is the return from close t to close t+1,
i.e. the return earned by a portfolio chosen at the close of day t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DATA_DIR, DOWNLOAD_START, TICKERS, TRAIN

PRICES_CSV = DATA_DIR / "prices.csv"


def download_prices(tickers=TICKERS, start=DOWNLOAD_START, path=PRICES_CSV) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)["Close"]
    df = df[list(tickers)].dropna()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def load_prices(refresh: bool = False) -> pd.DataFrame:
    if refresh or not PRICES_CSV.exists():
        return download_prices()
    return pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)[TICKERS]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


FEATURE_NAMES = ["ret1", "ret5", "ret20", "ret60", "vol20", "rsi14", "sma50_gap", "sma200_gap"]


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-asset technical features, columns named '<TICKER>_<feature>'."""
    logp = np.log(prices)
    cols = {}
    for tk in prices.columns:
        p, lp = prices[tk], logp[tk]
        r1 = lp.diff()
        cols[f"{tk}_ret1"] = r1
        cols[f"{tk}_ret5"] = lp.diff(5)
        cols[f"{tk}_ret20"] = lp.diff(20)
        cols[f"{tk}_ret60"] = lp.diff(60)
        cols[f"{tk}_vol20"] = r1.rolling(20).std() * np.sqrt(252)
        cols[f"{tk}_rsi14"] = _rsi(p) / 100 - 0.5
        cols[f"{tk}_sma50_gap"] = p / p.rolling(50).mean() - 1
        cols[f"{tk}_sma200_gap"] = p / p.rolling(200).mean() - 1
    return pd.DataFrame(cols, index=prices.index)


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    prices: pd.DataFrame          # aligned close prices
    features: np.ndarray          # (T, F) standardized features
    raw_features: pd.DataFrame    # unstandardized, for rule-based baselines
    fwd_returns: np.ndarray       # (T, N) simple return from t to t+1 (NaN on last row)
    feat_mean: np.ndarray
    feat_std: np.ndarray

    def index_range(self, start: str, end: str | None) -> tuple[int, int]:
        """Inclusive start / exclusive end row indices for decision days in [start, end].
        The last available day is excluded because it has no forward return."""
        mask = self.dates >= pd.Timestamp(start)
        if end is not None:
            mask &= self.dates <= pd.Timestamp(end)
        idx = np.flatnonzero(mask)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        return lo, min(hi, len(self.dates) - 1)


def make_market_data(prices: pd.DataFrame | None = None, stats=None) -> MarketData:
    """Build features. Standardization statistics come from the TRAIN period only
    (or are passed in, e.g. the ones saved with a trained model)."""
    if prices is None:
        prices = load_prices()
    feats = build_features(prices).dropna()
    prices = prices.loc[feats.index]
    if stats is None:
        tr = feats.loc[TRAIN[0]:TRAIN[1]]
        mean, std = tr.mean().values, tr.std().values + 1e-8
    else:
        mean, std = np.asarray(stats[0]), np.asarray(stats[1])
    z = np.clip((feats.values - mean) / std, -5, 5).astype(np.float32)
    fwd = (prices.shift(-1) / prices - 1).values
    return MarketData(feats.index, prices, z, feats, fwd, mean, std)
