"""
Gestion des logs de trades, métriques de performance et persistance CSV.
"""

import os
import csv
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

import config


def setup_logger(name: str = "trading_bot") -> logging.Logger:
    """Configure et retourne un logger rotatif."""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler fichier rotatif
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(config.LOG_DIR, f"{name}.log"),
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logger()


TRADE_FIELDS = [
    "id", "symbol", "direction", "entry_price", "exit_price",
    "quantity", "entry_time", "exit_time", "pnl", "pnl_pct",
    "sl_price", "tp_price", "exit_reason",
    "rsi_at_entry", "macd_at_entry", "bb_pct_at_entry",
    "adx_at_entry", "volume_ratio_at_entry",
    "ml_score", "rl_action",
]


class TradeLogger:
    """Enregistre et analyse l'historique des trades."""

    def __init__(self, csv_path: str = config.TRADES_CSV):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._ensure_csv()
        self.trades: List[Dict] = self._load_trades()

    # ──────────────────────────────────────────
    # Persistance CSV
    # ──────────────────────────────────────────

    def _ensure_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
                writer.writeheader()

    def _load_trades(self) -> List[Dict]:
        trades = []
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trades.append(row)
        except Exception as e:
            logger.warning(f"Impossible de charger les trades existants : {e}")
        return trades

    def log_trade(self, trade: Dict):
        """Ajoute un trade terminé dans le CSV et en mémoire."""
        # Remplir les champs manquants avec None
        complete = {field: trade.get(field, None) for field in TRADE_FIELDS}
        self.trades.append(complete)
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
                writer.writerow(complete)
            logger.info(
                f"Trade #{complete['id']} | {complete['symbol']} "
                f"{complete['direction']} | PnL: {complete['pnl']} $ "
                f"({complete['exit_reason']})"
            )
        except Exception as e:
            logger.error(f"Erreur écriture trade CSV : {e}")

    # ──────────────────────────────────────────
    # Métriques
    # ──────────────────────────────────────────

    def compute_metrics(self, last_n: Optional[int] = None) -> Dict:
        """Calcule les métriques de performance sur les N derniers trades."""
        trades = self.trades[-last_n:] if last_n else self.trades
        if not trades:
            return {}

        pnls = []
        wins = 0
        for t in trades:
            try:
                pnl = float(t.get("pnl") or 0)
                pnls.append(pnl)
                if pnl > 0:
                    wins += 1
            except (ValueError, TypeError):
                pass

        if not pnls:
            return {}

        total = len(pnls)
        win_rate = wins / total
        avg_pnl = sum(pnls) / total
        total_pnl = sum(pnls)

        # Calcul du drawdown maximum
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (simplifié, supposant rf=0)
        arr = np.array(pnls, dtype=float)
        sharpe = float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0

        metrics = {
            "total_trades": total,
            "win_rate": round(win_rate, 4),
            "avg_pnl": round(avg_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "max_drawdown": round(max_dd, 4),
            "sharpe_ratio": round(sharpe, 4),
        }
        logger.info(f"Métriques : {metrics}")
        return metrics

    def get_trades_for_ml(self) -> List[Dict]:
        """Retourne les trades avec toutes les features nécessaires pour le ML."""
        return [t for t in self.trades if t.get("pnl") is not None]

    def total_trade_count(self) -> int:
        return len(self.trades)
