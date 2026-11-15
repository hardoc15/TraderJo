"""Local dashboard for the RL paper-trading agent.

    python app/app.py    ->  http://127.0.0.1:8050
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, render_template, request

from rltrader import paper
from rltrader.config import RESULTS_DIR

app = Flask(__name__)
_agent = {}


def agent():
    if not _agent:
        _agent["model"], _agent["meta"] = paper.load_agent()
    return _agent["model"], _agent["meta"]


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/backtest")
def backtest():
    return jsonify(json.load(open(RESULTS_DIR / "dashboard.json")))


def _paper_payload(acct):
    if acct is None:
        return {"account": None}
    return {"account": {"history": acct["history"], "trades": acct["trades"][-200:],
                        "start_date": acct.get("start_date")},
            "summary": paper.account_summary(acct)}


@app.get("/api/paper")
def paper_state():
    return jsonify(_paper_payload(paper.load_account()))


@app.post("/api/paper/step")
def paper_step():
    model, meta = agent()
    try:
        acct = paper.step(refresh=True, model=model, meta=meta)
    except Exception as e:  # network errors etc.
        return jsonify({"error": str(e)}), 500
    return jsonify(_paper_payload(acct))


@app.post("/api/paper/replay")
def paper_replay():
    model, meta = agent()
    start = (request.get_json(silent=True) or {}).get("from", "2026-01-02")
    try:
        acct = paper.step(refresh=True, replay_from=start, model=model, meta=meta)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(_paper_payload(acct))


if __name__ == "__main__":
    app.run(debug=False, port=8050)
