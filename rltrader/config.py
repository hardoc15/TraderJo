"""Shared configuration for the RL trading project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Tradable universe: US equities, Nasdaq-100, long-term Treasuries, gold.
TICKERS = ["SPY", "QQQ", "TLT", "GLD"]

DOWNLOAD_START = "2005-01-01"   # extra history so 200-day features are warm
TRAIN = ("2006-01-01", "2019-12-31")
VAL = ("2020-01-01", "2021-12-31")   # used only for checkpoint / seed selection
TEST = ("2022-01-01", None)          # never seen during training or selection

TRANSACTION_COST = 0.001   # 10 bps per unit of turnover (commission + slippage)
TRADING_DAYS = 252
