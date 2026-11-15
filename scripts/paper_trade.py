"""Run the deployed agent on the simulated paper account.

    python scripts/paper_trade.py                       # trade the latest close (run once a day)
    python scripts/paper_trade.py --replay-from 2026-01-02   # fresh account, replay since date
    python scripts/paper_trade.py --alpaca              # also show orders for an Alpaca paper account
    python scripts/paper_trade.py --alpaca --submit     # ... and actually submit them
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rltrader import paper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-from")
    ap.add_argument("--no-refresh", action="store_true", help="use cached prices")
    ap.add_argument("--alpaca", action="store_true")
    ap.add_argument("--submit", action="store_true", help="submit Alpaca orders (default: dry run)")
    args = ap.parse_args()

    acct = paper.step(refresh=not args.no_refresh, replay_from=args.replay_from)
    s = paper.account_summary(acct)
    last = acct["history"][-1]
    print(f"As of {s['as_of']}: equity ${s['equity']:,.2f} ({100 * s['return']:+.2f}%)")
    print("Target weights:", {k: f"{100 * v:.1f}%" for k, v in last["weights"].items()})
    if args.alpaca:
        orders = paper.alpaca_sync(last["weights"], dry_run=not args.submit)
        print(("Submitted" if args.submit else "Dry run, would submit"), "Alpaca orders:")
        for o in orders:
            print("  ", o)


if __name__ == "__main__":
    main()
