"""
Orchestrateur principal du bot de trading.

Modes d'exécution :
  python bot.py                        → paper trading en temps réel
  python bot.py --mode live            → live trading via Alpaca
  python bot.py --mode backtest        → backtest sur les données historiques
  python bot.py --mode backtest --days 365  → backtest sur 1 an

Variables d'environnement (.env) :
  TRADING_MODE=paper|live
  ALPACA_API_KEY=...
  ALPACA_SECRET_KEY=...
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
from datetime import datetime, timezone
from typing import Dict, Optional

import config
from trade_logger import TradeLogger, setup_logger
from data_manager import DataManager
import numpy as np

from strategy import TradingStrategy, Signal, SignalResult
from risk_manager import RiskManager
from ml_model import SupervisedModel
from rl_agent import DQNAgent
from trader import PaperTrader, create_trader

# ──────────────────────────────────────────────────────────────────────────────
# Initialisation des logs
# ──────────────────────────────────────────────────────────────────────────────
logger = setup_logger("trading_bot")


class TradingBot:
    """
    Bot de trading autonome combinant :
    - Stratégie technique RSI+MACD+BB+ADX (~68 % win rate)
    - Supervised Learning (RandomForest + XGBoost) pour filtrer les signaux
    - Deep Q-Network (DQN) pour optimiser les décisions par renforcement
    - Risk management (Kelly Criterion, stop-loss, drawdown)
    - Paper trading et Live trading (Alpaca)
    - Apprentissage continu depuis les erreurs
    """

    def __init__(self, mode: str = None):
        self.mode = mode or config.TRADING_MODE
        self._running = False

        logger.info("=" * 60)
        logger.info(f"  BOT DE TRADING | Mode : {self.mode.upper()}")
        logger.info(f"  Marchés : {list(config.SYMBOLS.keys())}")
        logger.info("=" * 60)

        # Composants
        self.trade_logger = TradeLogger()
        self.data_manager = DataManager()
        self.strategy     = TradingStrategy()
        self.ml_model     = SupervisedModel()
        self.rl_agent     = DQNAgent()
        self.trader       = create_trader(self.trade_logger)

        # Récupération du solde réel depuis Alpaca si les clés sont configurées
        initial_capital = self._fetch_alpaca_balance()
        self.risk_manager = RiskManager(initial_capital=initial_capital)

        # Compteur de trades depuis le dernier ré-entraînement ML
        self._trades_since_ml_retrain = 0
        self._rl_step_counter         = 0

        # Chargement de l'état persistant
        self._load_state()

        # Vérification aptitude live (bloque si < 50 trades paper)
        if self.mode == "live":
            if not self.risk_manager.check_live_readiness(self.trade_logger.total_trade_count()):
                logger.error("Bot non prêt pour le live — passez d'abord par alpaca-paper.")
                sys.exit(1)

        # Gestion du shutdown propre (SIGINT/SIGTERM — compatibilité Windows)
        signal.signal(signal.SIGINT, self._shutdown_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._shutdown_handler)

    # ──────────────────────────────────────────
    # Modes d'exécution
    # ──────────────────────────────────────────

    def run_paper_or_live(self):
        """Boucle principale : analyse toutes les N secondes."""
        self._running = True
        logger.info(f"Démarrage de la boucle principale (intervalle={config.LOOP_INTERVAL_SECONDS}s)")

        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"Erreur inattendue dans la boucle principale : {e}", exc_info=True)

            if self._running:
                logger.debug(f"Pause {config.LOOP_INTERVAL_SECONDS}s...")
                time.sleep(config.LOOP_INTERVAL_SECONDS)

        logger.info("Bot arrêté proprement.")

    def run_backtest(self, days: int = 365):
        """
        Backtest sur données historiques.
        Affiche les métriques pour chaque marché.
        """
        logger.info(f"=== BACKTEST sur {days} jours ===")
        results = {}
        for name, symbols in config.SYMBOLS.items():
            sym = symbols["yfinance"]
            logger.info(f"Backtest {name.upper()} ({sym})...")
            df = self.data_manager.get_ohlcv(sym, timeframe=config.TIMEFRAME, days=days)
            if df is None or df.empty:
                logger.warning(f"Pas de données pour {sym}")
                continue
            metrics = self.strategy.backtest(df)
            results[name] = metrics
            logger.info(f"  {name}: {metrics}")

        logger.info("=== Résultats du Backtest ===")
        for name, m in results.items():
            logger.info(
                f"  {name:12s} | Trades={m.get('total_trades', 0):4d} | "
                f"WR={m.get('win_rate', 0):.1%} | AvgPnL={m.get('avg_pnl', 0):.6f}"
            )
        return results

    # ──────────────────────────────────────────
    # Tick principal
    # ──────────────────────────────────────────

    def _tick(self):
        """Une itération de la boucle principale."""
        logger.debug(f"--- Tick {datetime.now(timezone.utc).isoformat()} ---")

        if not self.risk_manager.can_trade():
            logger.warning("Trading suspendu par le risk manager.")
            return

        current_prices: Dict[str, float] = {}
        candle_data:    Dict[str, Dict]   = {}

        for market_name, symbols in config.SYMBOLS.items():
            sym = symbols["yfinance"]
            alpaca_sym = symbols["alpaca"]

            # 1. Récupération des données
            df = self.data_manager.get_ohlcv(sym, timeframe=config.TIMEFRAME)
            if df is None or df.empty:
                logger.warning(f"[{market_name}] Pas de données disponibles.")
                continue

            last_row = df.iloc[-1]
            current_price = float(last_row["Close"])
            current_prices[alpaca_sym] = current_price
            candle_data[alpaca_sym] = {
                "high": float(last_row["High"]),
                "low":  float(last_row["Low"]),
            }

            # 2. Vérification des SL/TP pour les positions ouvertes (paper et live)
            closed_trades = self.trader.check_and_close_positions(
                current_prices, candle_data
            )
            for trade in closed_trades:
                self._on_trade_closed(trade)

            # 3. Génération du signal technique
            signal_result = self.strategy.generate_signal(df)

            if signal_result.signal == Signal.HOLD:
                logger.debug(f"[{market_name}] Signal HOLD.")
                continue

            # 4. Filtrage via Supervised Learning
            ml_features = {
                "rsi_at_entry":          signal_result.rsi,
                "macd_at_entry":         signal_result.macd,
                "bb_pct_at_entry":       signal_result.bb_pct,
                "adx_at_entry":          signal_result.adx,
                "volume_ratio_at_entry": signal_result.volume_ratio,
            }
            ml_score = self.ml_model.predict_confidence(ml_features)

            if ml_score < config.ML_CONFIDENCE_THRESHOLD:
                logger.info(
                    f"[{market_name}] Signal {signal_result.signal.value} rejeté par ML "
                    f"(score={ml_score:.3f} < {config.ML_CONFIDENCE_THRESHOLD})"
                )
                continue

            # 5. Décision via RL (DQN)
            open_pos = self.trader.get_open_positions()
            in_position = 1 if any(
                p.get("symbol", p.get("trade_id", "")) == alpaca_sym
                or p.get("symbol", "") == alpaca_sym
                for p in (open_pos.values() if isinstance(open_pos, dict) else open_pos)
            ) else 0
            rl_state = self._build_rl_state(df, in_position, signal_result)
            rl_action = self.rl_agent.select_action(rl_state)

            # Validation de l'action RL
            if rl_action not in (0, 1, 2):
                logger.warning(f"[{market_name}] Action RL invalide ({rl_action}) — skip.")
                continue

            # RL doit confirmer l'action technique (1=Long, 2=Short)
            expected_action = 1 if signal_result.signal == Signal.LONG else 2
            if rl_action == 0:
                logger.info(
                    f"[{market_name}] Signal {signal_result.signal.value} rejeté par RL (action=Hold)."
                )
                continue

            if rl_action != expected_action and self.rl_agent.is_trained_enough():
                logger.info(
                    f"[{market_name}] RL désaccord (signal={expected_action}, RL={rl_action}) — skip."
                )
                continue

            # 6. Calcul de la taille de position
            win_rate = self.trade_logger.compute_metrics(last_n=50).get("win_rate", 0.55)
            quantity = self.risk_manager.compute_position_size(
                entry_price=signal_result.entry_price,
                sl_price=signal_result.sl_price,
                win_rate=max(win_rate, 0.45),
            )

            if quantity <= 0:
                logger.warning(f"[{market_name}] Quantité calculée = 0 — trade ignoré.")
                continue

            # 7. Exécution du trade
            trade_id = self.trader.open_position(alpaca_sym, signal_result, quantity)
            if trade_id:
                # Suivi des positions ouvertes pour les limites du risk manager
                cost = signal_result.entry_price * quantity
                self.risk_manager.increment_positions(cost)
                # Stocker ml_score et rl_action pour les logs du trade
                if isinstance(self.trader, PaperTrader) and alpaca_sym in self.trader.positions:
                    self.trader.positions[alpaca_sym].signal_data["ml_score"]  = round(ml_score, 4)
                    self.trader.positions[alpaca_sym].signal_data["rl_action"] = rl_action
                logger.info(
                    f"[{market_name}] Trade ouvert #{trade_id} | "
                    f"ML={ml_score:.3f} | RL={rl_action} | Raisons : {signal_result.reasons}"
                )

        # 8. Entraînement RL périodique
        self._rl_step_counter += 1
        if self._rl_step_counter % config.RL_TRAIN_EVERY_N == 0:
            loss = self.rl_agent.learn()
            if loss is not None:
                logger.debug(f"RL loss = {loss:.6f}")

        # 9. Mise à jour du risk manager (PnL non réalisé — paper et live)
        unrealized = self.trader.get_unrealized_pnl(current_prices)
        self.risk_manager.record_unrealized_pnl(unrealized)

        self._save_state()

    # ──────────────────────────────────────────
    # Callback : trade fermé
    # ──────────────────────────────────────────

    def _on_trade_closed(self, trade: Dict):
        """Appelé après la fermeture de chaque trade."""
        pnl         = float(trade.get("pnl", 0))
        exit_reason = trade.get("exit_reason", "unknown")

        # Mise à jour du capital + décrémentation des positions suivies
        position_cost = float(trade.get("entry_price", 0)) * float(trade.get("quantity", 0))
        self.risk_manager.update_capital(pnl, position_cost)

        # Apprentissage RL depuis l'erreur
        dd = self.risk_manager.status()["drawdown"]
        reward = self.rl_agent.compute_reward(pnl, dd, exit_reason)

        # Stocker l'expérience (state→action→reward→next_state→done)
        # (approximation : on réutilise le dernier état encodé)
        dummy_state = self._dummy_state()
        action = 1 if trade.get("direction") == "LONG" else 2
        self.rl_agent.store_experience(dummy_state, action, reward, dummy_state, True)

        # Compteur pour re-training ML supervisé
        self._trades_since_ml_retrain += 1
        if self._trades_since_ml_retrain >= config.ML_RETRAIN_EVERY_N_TRADES:
            logger.info(
                f"Ré-entraînement ML supervisé "
                f"({self._trades_since_ml_retrain} nouveaux trades)..."
            )
            trades_data = self.trade_logger.get_trades_for_ml()
            metrics = self.ml_model.train(trades_data)
            if metrics:
                logger.info(f"ML ré-entraîné : {metrics}")
            self._trades_since_ml_retrain = 0
            self.rl_agent.save()

        # Métriques globales
        total = self.trade_logger.total_trade_count()
        if total % 10 == 0:
            m = self.trade_logger.compute_metrics(last_n=50)
            logger.info(f"Métriques (50 derniers) : {m}")

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _fetch_alpaca_balance(self) -> float:
        """
        Récupère le solde réel du compte Alpaca (paper ou live).
        Retourne INITIAL_CAPITAL si les clés ne sont pas configurées.
        """
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            logger.info(f"Clés Alpaca non configurées — capital par défaut : {config.INITIAL_CAPITAL} $")
            return config.INITIAL_CAPITAL
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(
                config.ALPACA_API_KEY,
                config.ALPACA_SECRET_KEY,
                config.ALPACA_BASE_URL,
                api_version="v2",
            )
            equity = float(api.get_account().equity)
            logger.info(f"Solde Alpaca récupéré : {equity:.2f} $")
            return equity
        except Exception as e:
            logger.warning(f"Impossible de récupérer le solde Alpaca ({e}) — capital par défaut : {config.INITIAL_CAPITAL} $")
            return config.INITIAL_CAPITAL

    def _build_rl_state(self, df, in_position: int, signal_result) -> np.ndarray:
        closes = df["Close"].values
        ret_1h = float((closes[-1] - closes[-2]) / closes[-2]) if len(closes) >= 2 else 0.0
        ret_4h = float((closes[-1] - closes[-5]) / closes[-5]) if len(closes) >= 5 else 0.0
        capital_ratio = self.risk_manager.capital / self.risk_manager._initial_capital

        return self.rl_agent.encode_state(
            rsi=signal_result.rsi,
            macd=signal_result.macd,
            bb_pct=signal_result.bb_pct,
            adx=signal_result.adx,
            volume_ratio=signal_result.volume_ratio,
            price_ret_1h=ret_1h,
            price_ret_4h=ret_4h,
            in_position=in_position,
            pnl_pct=0.0,
            capital_ratio=capital_ratio,
        )

    def _dummy_state(self) -> np.ndarray:
        return np.zeros(10, dtype=np.float32)

    # ──────────────────────────────────────────
    # Persistance d'état
    # ──────────────────────────────────────────

    def _save_state(self):
        state = {
            "capital":                     self.risk_manager.capital,
            "peak_capital":                self.risk_manager.peak_capital,
            "initial_capital":             self.risk_manager._initial_capital,
            "trades_since_ml_retrain":     self._trades_since_ml_retrain,
            "rl_steps":                    self.rl_agent.steps,
            "rl_epsilon":                  self.rl_agent.epsilon,
            "last_save":                   datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(config.STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Erreur sauvegarde état : {e}")

    def _load_state(self):
        if not os.path.exists(config.STATE_FILE):
            return
        try:
            with open(config.STATE_FILE, "r") as f:
                state = json.load(f)
            self.risk_manager.capital              = state.get("capital", config.INITIAL_CAPITAL)
            self.risk_manager.peak_capital         = state.get("peak_capital", config.INITIAL_CAPITAL)
            self.risk_manager._initial_capital     = state.get("initial_capital", config.INITIAL_CAPITAL)
            self._trades_since_ml_retrain          = state.get("trades_since_ml_retrain", 0)
            self.rl_agent.steps                    = state.get("rl_steps", 0)
            self.rl_agent.epsilon                  = state.get("rl_epsilon", config.RL_EPSILON_START)
            logger.info(
                f"État restauré : capital={self.risk_manager.capital:.2f} $, "
                f"epsilon={self.rl_agent.epsilon:.3f}"
            )
        except Exception as e:
            logger.warning(f"Erreur chargement état : {e}")

    def _shutdown_handler(self, signum, frame):
        logger.info(f"Signal {signum} reçu — arrêt propre en cours...")
        self._running = False
        self.rl_agent.save()
        self._save_state()

        # Fermeture de toutes les positions ouvertes (paper et live)
        alpaca_to_yf = {v["alpaca"]: v["yfinance"] for v in config.SYMBOLS.values()}
        open_positions = self.trader.get_open_positions()
        symbols = (
            list(self.trader.positions.keys())
            if isinstance(self.trader, PaperTrader)
            else [p["symbol"] for p in open_positions]
        )
        for symbol in symbols:
            yf_sym = alpaca_to_yf.get(symbol, symbol)
            price = self.data_manager.get_latest_price(yf_sym) or 0.0
            self.trader.force_close(symbol, price, reason="shutdown")
            logger.info(f"[{symbol}] Position fermée au shutdown.")

        logger.info("Bot arrêté. À bientôt !")
        sys.exit(0)


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Bot de trading autonome avec apprentissage")
    parser.add_argument(
        "--mode",
        choices=["paper", "alpaca-paper", "live", "backtest"],
        default=config.TRADING_MODE,
        help="Mode d'exécution (défaut : alpaca-paper)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Nombre de jours pour le backtest (défaut : 365)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bot = TradingBot(mode=args.mode)

    if args.mode == "backtest":
        results = bot.run_backtest(days=args.days)
        print("\n=== RÉSULTATS BACKTEST ===")
        for name, m in results.items():
            wr = m.get("win_rate", 0)
            n  = m.get("total_trades", 0)
            avg = m.get("avg_pnl", 0)
            print(f"  {name:12s} | {n:4d} trades | WR={wr:.1%} | Avg PnL={avg:.6f}")
    else:
        bot.run_paper_or_live()
