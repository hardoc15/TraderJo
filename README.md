# TraderJo: Reinforcement-Learning Paper Trader

TraderJo is a PPO (Proximal Policy Optimization) agent that rebalances a long-only portfolio of SPY, QQQ, TLT, GLD and cash once a day. A local web dashboard shows a simulated $100k paper account and backtests. No real money is traded.

## How it works
- **Environment:** a custom Gymnasium environment ([rltrader/env.py](rltrader/env.py)). The agent observes 32 technical features plus its current holdings. Its action is mapped to portfolio weights with a softmax. The reward is the daily log return after a 10 bps transaction cost.
- **Data:** daily Yahoo Finance prices, downloaded by the code and not stored in the repo. Data is split by date: train 2006–2019, validation 2020–2021, test 2022–present. Checkpoints and seeds are chosen using validation data only.
- **Comparisons:** buy-and-hold SPY, 60/40, equal weight, a 200-day moving-average trend rule, momentum, and a random agent. We also test variants of the PPO reward (risk-penalized, and trained without costs).

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --variant logret --seed 0   # repeat for each variant and seed
python scripts/evaluate.py                          # results/ + models/deployed/
python app/app.py                                   # http://127.0.0.1:5000
python scripts/paper_trade.py                       # trade the latest close
```
To send orders to an Alpaca paper account (optional), set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` and add `--alpaca --submit`.

## Repository
`rltrader/` library · `scripts/` training, evaluation, paper trading · `app/` dashboard · `results/` metrics, plots · `slides/`, `report/` deliverables.

COMP 560 course project. Not financial advice.
