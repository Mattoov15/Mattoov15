"""
Dashboard web temps réel du bot de trading.
Accessible sur http://localhost:5000 depuis n'importe quel appareil du réseau.

Lancement :
    python dashboard.py
    python dashboard.py --port 5000 --host 0.0.0.0
"""

import os
import json
import csv
import argparse
from datetime import datetime
from typing import List, Dict

from flask import Flask, render_template, jsonify

import config

app = Flask(__name__)


# ──────────────────────────────────────────────────────────────────
# Helpers de lecture des données
# ──────────────────────────────────────────────────────────────────

def _load_trades() -> List[Dict]:
    trades = []
    if not os.path.exists(config.TRADES_CSV):
        return trades
    try:
        with open(config.TRADES_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)
    except Exception:
        pass
    return trades


def _load_state() -> Dict:
    if not os.path.exists(config.STATE_FILE):
        return {}
    try:
        with open(config.STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _compute_metrics(trades: List[Dict]) -> Dict:
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0,
            "total_pnl": 0, "avg_pnl": 0,
            "wins": 0, "losses": 0,
            "max_drawdown": 0, "best_trade": 0, "worst_trade": 0,
        }

    pnls = []
    for t in trades:
        try:
            pnls.append(float(t.get("pnl") or 0))
        except (ValueError, TypeError):
            pnls.append(0.0)

    wins   = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    total  = len(pnls)

    # Drawdown max
    cumulative, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return {
        "total_trades": total,
        "win_rate":     round(wins / total * 100, 1) if total else 0,
        "total_pnl":    round(sum(pnls), 2),
        "avg_pnl":      round(sum(pnls) / total, 2) if total else 0,
        "wins":         wins,
        "losses":       losses,
        "max_drawdown": round(max_dd * 100, 2),
        "best_trade":   round(max(pnls), 2) if pnls else 0,
        "worst_trade":  round(min(pnls), 2) if pnls else 0,
    }


def _capital_curve(trades: List[Dict]) -> List[Dict]:
    """Courbe de capital au fil des trades."""
    state   = _load_state()
    capital = state.get("initial_capital", config.INITIAL_CAPITAL)
    curve = [{"x": 0, "y": round(capital, 2), "label": "Départ"}]
    for i, t in enumerate(trades, 1):
        try:
            pnl = float(t.get("pnl") or 0)
            capital += pnl
            curve.append({
                "x":      i,
                "y":      round(capital, 2),
                "label":  t.get("exit_time", "")[:16],
                "symbol": t.get("symbol", ""),
                "pnl":    round(pnl, 2),
            })
        except (ValueError, TypeError):
            pass
    return curve


def _win_rate_curve(trades: List[Dict], window: int = 20) -> List[Dict]:
    """Win rate glissant sur les N derniers trades."""
    curve = []
    for i in range(1, len(trades) + 1):
        window_trades = trades[max(0, i - window): i]
        wins = sum(1 for t in window_trades if float(t.get("pnl") or 0) > 0)
        wr = round(wins / len(window_trades) * 100, 1) if window_trades else 0
        curve.append({"x": i, "y": wr})
    return curve


# ──────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/data")
def api_data():
    trades  = _load_trades()
    state   = _load_state()
    metrics = _compute_metrics(trades)
    recent  = trades[-20:][::-1]   # 20 derniers trades, plus récent en premier

    # Statut du bot
    last_save       = state.get("last_save", "Inconnu")
    capital         = state.get("capital", config.INITIAL_CAPITAL)
    initial_capital = state.get("initial_capital", config.INITIAL_CAPITAL)
    rl_steps        = state.get("rl_steps", 0)
    epsilon         = state.get("rl_epsilon", 1.0)

    # Calcul profit/perte depuis le début
    pnl_total     = capital - initial_capital
    pnl_total_pct = round(pnl_total / initial_capital * 100, 2) if initial_capital else 0

    return jsonify({
        "metrics":      metrics,
        "capital_curve": _capital_curve(trades),
        "winrate_curve": _win_rate_curve(trades),
        "recent_trades": recent,
        "bot_status": {
            "mode":         config.TRADING_MODE.upper(),
            "capital":      round(capital, 2),
            "pnl_total":    round(pnl_total, 2),
            "pnl_total_pct": pnl_total_pct,
            "rl_steps":     rl_steps,
            "rl_epsilon":   round(epsilon, 3),
            "last_update":  last_save,
            "markets":      list(config.SYMBOLS.keys()),
        },
    })


# ──────────────────────────────────────────────────────────────────
# Lancement
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dashboard du bot de trading")
    parser.add_argument("--host", default="0.0.0.0", help="Hôte (défaut 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port (défaut 5000)")
    args = parser.parse_args()

    print(f"\n  Dashboard disponible sur :")
    print(f"  → Local   : http://localhost:{args.port}")
    print(f"  → Réseau  : http://<votre-ip>:{args.port}  (accessible depuis téléphone)\n")
    app.run(host=args.host, port=args.port, debug=False)
