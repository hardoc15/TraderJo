"""Evaluate all trained agents + baselines on the held-out TEST period.

Outputs (in results/):
  test_runs.csv        one row per strategy / seed
  summary.csv, summary.md   mean +- std across seeds
  cost_sensitivity.csv deployed agent vs baselines at 0 / 10 / 25 bps
  equity_test.png, learning_curves.png, allocation_test.png
  dashboard.json       data consumed by the web app
Also copies the agent with the best VALIDATION Sharpe to models/deployed/.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rltrader.backtest import BASELINES, ppo_policy, random_policy, run_backtest
from rltrader.config import MODELS_DIR, RESULTS_DIR, TEST, TICKERS, VAL
from rltrader.data import make_market_data
from rltrader.metrics import compute_metrics

VARIANT_LABELS = {"logret": "PPO daily", "risk": "PPO daily, risk-penalised",
                  "nocost": "PPO daily, trained w/o costs", "weekly": "PPO weekly",
                  "weekly_risk": "PPO weekly, risk-penalised", "monthly": "PPO monthly"}


def k_of(meta):
    return meta["env_kwargs"].get("rebalance_every", 1)


def metrics_of(bt):
    return compute_metrics(bt.net_return.values, bt.turnover.values)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", choices=VARIANT_LABELS,
                    help="override the validation-based rule (e.g. after inspecting test results)")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    md = make_market_data()
    test = md.index_range(*TEST)
    val = md.index_range(*VAL)
    rows, curves = [], {}

    for name, make in BASELINES.items():
        bt = run_backtest(make(md), md, *test)
        rows.append({"strategy": name, "seed": 0, **metrics_of(bt)})
        curves[name] = bt.value
    for s in range(5):
        bt = run_backtest(random_policy(s), md, *test)
        rows.append({"strategy": "Random agent", "seed": s, **metrics_of(bt)})

    agents = {}
    for d in sorted(MODELS_DIR.glob("*_seed*")):
        meta = json.load(open(d / "meta.json"))
        model = PPO.load(d / "best", device="cpu")
        bt = run_backtest(ppo_policy(model, k_of(meta)), md, *test)
        label = VARIANT_LABELS[meta["variant"]]
        rows.append({"strategy": label, "seed": meta["seed"], "val_sharpe": meta["best_val_sharpe"],
                     **metrics_of(bt)})
        agents[d.name] = dict(meta=meta, model=model, bt=bt, label=label)

    # ---- 5-seed ensembles (average target weights) for each variant
    ensembles = {}
    for variant in VARIANT_LABELS:
        members = [a for a in agents.values() if a["meta"]["variant"] == variant]
        if not members:
            continue
        models, k = [a["model"] for a in members], k_of(members[0]["meta"])
        bt = run_backtest(ppo_policy(models, k), md, *test)
        mean_val = float(np.mean([a["meta"]["best_val_sharpe"] for a in members]))
        rows.append({"strategy": VARIANT_LABELS[variant] + " (5-seed ensemble)", "seed": -1,
                     "val_sharpe": mean_val, **metrics_of(bt)})
        ensembles[variant] = dict(members=members, models=models, k=k, bt=bt, mean_val=mean_val)

    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS_DIR / "test_runs.csv", index=False)

    cols = ["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "avg_daily_turnover"]
    g = runs.groupby("strategy", sort=False)[cols]
    mean, std, n = g.mean(), g.std().fillna(0), g.size()
    mean.assign(n_seeds=n).to_csv(RESULTS_DIR / "summary.csv")

    def fmt(c, m, s, k):
        pct = c != "sharpe"
        f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.2f}")
        return f(m) + (f" ± {f(s)}" if k > 1 else "")
    header = "| Strategy | Total return | CAGR | Ann. vol | Sharpe | Max drawdown | Daily turnover |"
    lines = [header, "|" + "---|" * 7]
    for strat in mean.index:
        lines.append(f"| {strat} (n={n[strat]}) | " + " | ".join(
            fmt(c, mean.loc[strat, c], std.loc[strat, c], n[strat]) for c in cols) + " |")
    (RESULTS_DIR / "summary.md").write_text(
        f"Test period: {md.dates[test[0]].date()} to {md.dates[test[1]].date()}, "
        f"10 bps transaction cost, mean ± std over seeds.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- deployed agent (rule fixed before looking at test results): the variant with the
    # highest MEAN validation Sharpe across its seeds, deployed as a 5-seed ensemble.
    rule_variant = max(ensembles, key=lambda v: ensembles[v]["mean_val"])
    best_variant = args.deploy or rule_variant
    best = ensembles[best_variant]
    selection = ("validation rule" if best_variant == rule_variant else
                 f"manual override of validation rule (rule picked '{rule_variant}'), "
                 "chosen after inspecting test results")
    dep = MODELS_DIR / "deployed"
    shutil.rmtree(dep, ignore_errors=True)
    dep.mkdir()
    for a in best["members"]:
        shutil.copy(MODELS_DIR / f"{best_variant}_seed{a['meta']['seed']}" / "best.zip",
                    dep / f"model_seed{a['meta']['seed']}.zip")
    meta = {k: best["members"][0]["meta"][k] for k in ("variant", "env_kwargs", "feat_mean", "feat_std")}
    meta.update(seeds=[a["meta"]["seed"] for a in best["members"]], mean_val_sharpe=best["mean_val"],
                selection=selection)
    json.dump(meta, open(dep / "meta.json", "w"))
    print("\nMean validation Sharpe by variant:",
          {v: round(e["mean_val"], 3) for v, e in ensembles.items()})
    best_name = f"{VARIANT_LABELS[best_variant]} ensemble"
    print(f"Deployed agent: {best_name} ({selection})")
    if best_variant != rule_variant:
        curves[f"Validation-rule pick: {VARIANT_LABELS[rule_variant]} ensemble"] = ensembles[rule_variant]["bt"].value
    curves[f"Deployed: {best_name}"] = best["bt"].value

    # ---- cost sensitivity
    cs = []
    for bps in [0, 10, 25]:
        c = bps / 1e4
        for name, pol in [("Deployed PPO", ppo_policy(best["models"], best["k"])),
                          ("Buy & hold SPY", BASELINES["Buy & hold SPY"](md)),
                          ("Equal weight (monthly)", BASELINES["Equal weight (monthly)"](md)),
                          ("SMA-200 trend", BASELINES["SMA-200 trend"](md))]:
            m = metrics_of(run_backtest(pol, md, *test, cost=c))
            cs.append({"cost_bps": bps, "strategy": name, **{k: m[k] for k in cols}})
    pd.DataFrame(cs).to_csv(RESULTS_DIR / "cost_sensitivity.csv", index=False)

    # ---- does the agent's *timing* add value? Compare with a static, monthly-rebalanced
    # portfolio holding the agent's average weights on the TRAIN period.
    from rltrader.backtest import _fixed_weights
    from rltrader.config import TRAIN
    tr_bt = run_backtest(ppo_policy(best["models"], best["k"]), md, *md.index_range(*TRAIN))
    wcols_all = [c for c in tr_bt.columns if c.startswith("w_")]
    w_avg = tr_bt[wcols_all].mean().values
    static = metrics_of(run_backtest(_fixed_weights(w_avg), md, *test))
    agent_m = metrics_of(best["bt"])
    pd.DataFrame([{"strategy": "Deployed agent", **agent_m},
                  {"strategy": "Static mix of agent's avg train weights", **static,
                   **{f"weight_{c[2:]}": round(x, 3) for c, x in zip(wcols_all, w_avg)}}]
                 ).to_csv(RESULTS_DIR / "static_mix.csv", index=False)
    print(f"Static mix of agent's avg weights {np.round(w_avg, 3)}: test Sharpe {static['sharpe']:.2f} "
          f"vs agent {agent_m['sharpe']:.2f}")

    # ---- plots
    plt.figure(figsize=(9, 4.5))
    for name, v in curves.items():
        plt.plot(v.index, v.values, lw=2.2 if "PPO" in name else 1.2, label=name)
    plt.axhline(1, color="grey", lw=0.6)
    plt.title("Held-out test period: growth of $1 (10 bps costs)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "equity_test.png", dpi=160)

    plt.figure(figsize=(9, 4))
    for variant, label in VARIANT_LABELS.items():
        logs = [pd.DataFrame(a["meta"]["val_log"]).set_index("step")["sharpe"]
                for a in agents.values() if a["meta"]["variant"] == variant]
        if logs:
            df = pd.concat(logs, axis=1)
            plt.plot(df.index, df.mean(axis=1), label=label)
            plt.fill_between(df.index, df.min(axis=1), df.max(axis=1), alpha=0.15)
    plt.xlabel("training steps")
    plt.ylabel("validation Sharpe")
    plt.title("Learning curves (mean, min-max over 5 seeds)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curves.png", dpi=160)

    wcols = [f"w_{t}" for t in TICKERS] + ["w_CASH"]
    w = best["bt"][wcols].rolling(5).mean()
    plt.figure(figsize=(9, 3.5))
    plt.stackplot(w.index, w.T.values, labels=TICKERS + ["Cash"])
    plt.ylim(0, 1)
    plt.title("Deployed agent allocation on test period (5-day avg)")
    plt.legend(fontsize=8, loc="upper left", ncol=5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "allocation_test.png", dpi=160)

    dash = {
        "deployed": best_name,
        "selection": selection,
        "test_start": str(md.dates[test[0]].date()),
        "summary_md": (RESULTS_DIR / "summary.md").read_text(),
        "summary": mean.reset_index().to_dict(orient="records"),
        "curves": {k: {"dates": [str(d.date()) for d in v.index], "values": v.round(5).tolist()}
                   for k, v in curves.items()},
        "allocation": {"dates": [str(d.date()) for d in best["bt"].index],
                       **{c[2:]: best["bt"][c].round(4).tolist() for c in wcols}},
    }
    json.dump(dash, open(RESULTS_DIR / "dashboard.json", "w"))


if __name__ == "__main__":
    main()
