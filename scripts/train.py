"""Train a PPO portfolio agent.

Checkpoints are evaluated on the VALIDATION period (2020-2021) every --eval-freq
steps; the checkpoint with the best validation Sharpe is kept. The test period
is never touched here.

Example:
    python scripts/train.py --variant logret --seed 0 --timesteps 400000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rltrader.backtest import ppo_policy, run_backtest
from rltrader.config import MODELS_DIR, TRAIN, TRANSACTION_COST, VAL
from rltrader.data import make_market_data
from rltrader.env import PortfolioEnv
from rltrader.metrics import compute_metrics

VARIANTS = {
    # name: env kwargs used during training
    "logret": dict(reward_mode="logret", cost=TRANSACTION_COST),
    "risk": dict(reward_mode="risk", risk_lambda=0.02, cost=TRANSACTION_COST),
    "nocost": dict(reward_mode="logret", cost=0.0),   # ablation: trained ignoring costs
    # lower decision frequency: trade on day 1, then hold k days
    "weekly": dict(reward_mode="logret", cost=TRANSACTION_COST, rebalance_every=5),
    "weekly_risk": dict(reward_mode="risk", risk_lambda=0.02, cost=TRANSACTION_COST, rebalance_every=5),
    "monthly": dict(reward_mode="logret", cost=TRANSACTION_COST, rebalance_every=21),
}


class ValSelector(BaseCallback):
    def __init__(self, md, val_range, eval_freq, out_dir, k=1):
        super().__init__()
        self.k = k
        self.md, self.val_range, self.eval_freq, self.out_dir = md, val_range, eval_freq, out_dir
        self.best, self.log = -np.inf, []

    def _on_step(self):
        if self.num_timesteps % self.eval_freq < self.training_env.num_envs:
            bt = run_backtest(ppo_policy(self.model, self.k), self.md, *self.val_range)
            m = compute_metrics(bt.net_return.values, bt.turnover.values)
            self.log.append({"step": self.num_timesteps, **m})
            if m["sharpe"] > self.best:
                self.best = m["sharpe"]
                self.model.save(self.out_dir / "best")
            print(f"step {self.num_timesteps:>7d}  val sharpe {m['sharpe']:+.3f}  "
                  f"ret {m['total_return']:+.3f}  turnover {m['avg_daily_turnover']:.3f}", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default="logret")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=400_000)
    ap.add_argument("--eval-freq", type=int, default=20_000)
    args = ap.parse_args()

    torch.set_num_threads(1)
    md = make_market_data()
    out = MODELS_DIR / f"{args.variant}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    tr = md.index_range(*TRAIN)
    val = md.index_range(*VAL)

    env_kwargs = VARIANTS[args.variant]
    venv = DummyVecEnv([lambda: PortfolioEnv(md, *tr, episode_len=504, **env_kwargs) for _ in range(4)])
    venv.seed(args.seed)
    model = PPO("MlpPolicy", venv, seed=args.seed, learning_rate=3e-4, n_steps=1024,
                batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.001,
                policy_kwargs=dict(net_arch=[64, 64], log_std_init=-0.5), verbose=0)
    cb = ValSelector(md, val, args.eval_freq, out, env_kwargs.get("rebalance_every", 1))
    t0 = time.time()
    model.learn(args.timesteps, callback=cb)
    model.save(out / "final")

    json.dump({"variant": args.variant, "seed": args.seed, "timesteps": args.timesteps,
               "best_val_sharpe": cb.best, "train_seconds": time.time() - t0,
               "env_kwargs": env_kwargs, "val_log": cb.log,
               "feat_mean": md.feat_mean.tolist(), "feat_std": md.feat_std.tolist()},
              open(out / "meta.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s, best val sharpe {cb.best:.3f}")


if __name__ == "__main__":
    main()
