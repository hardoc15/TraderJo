# TraderJo: Reinforcement-Learning Paper Trader

A PPO agent rebalances a long-only SPY/QQQ/TLT/GLD/cash portfolio. A local dashboard runs a simulated $100k paper account and shows backtests. No real money is traded.

## Method
- **Environment** ([rltrader/env.py](rltrader/env.py)): the agent observes 32 technical features plus its current weights. A softmax maps its actions to weights. The reward is the log return after 10 bps costs. Rebalancing runs daily, weekly or monthly.
- **Data**: Yahoo Finance daily prices (downloaded, not stored). Train 2006–2019, validate 2020–2021, test 2022–present.
- **Baselines**: buy-and-hold SPY, 60/40, equal weight, SMA-200 trend, momentum, random.

## Results (test, 2022–2026)
The weekly PPO ensemble reaches Sharpe 0.62 with −20.7% max drawdown, versus 0.70 / −24.5% for SPY. Daily PPO trades about 25% of the portfolio a day, and costs cut its Sharpe to 0.23. The agent learns a nearly fixed diversified mix. Its timing adds no value over holding that mix statically (0.67). See [results/summary.md](results/summary.md).

## Run
```bash
pip install -r requirements.txt
python scripts/train.py --variant weekly --seed 0   # see VARIANTS
python scripts/evaluate.py --deploy weekly
python app/app.py                                   # http://127.0.0.1:8050
python scripts/paper_trade.py                       # trade the latest close
```
For an Alpaca paper account, set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`, then add `--alpaca --submit`.

## Layout
`rltrader/` library · `scripts/` · `app/` dashboard · `results/` · `models/deployed/` · `slides/`, `report/`

COMP 560 course project. Not financial advice.
