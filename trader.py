"""
Exécution des ordres en mode Paper Trading (simulé) ou Live (Alpaca).
Gère les positions ouvertes, le suivi PnL et la fermeture des ordres.
"""

import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional

import config
from strategy import Signal, SignalResult
from trade_logger import TradeLogger

logger = logging.getLogger("trading_bot")


class Position:
    """Représente une position ouverte."""

    def __init__(
        self,
        trade_id:    str,
        symbol:      str,
        direction:   str,
        entry_price: float,
        quantity:    float,
        sl_price:    float,
        tp_price:    float,
        signal_data: Dict,
    ):
        self.trade_id    = trade_id
        self.symbol      = symbol
        self.direction   = direction
        self.entry_price = entry_price
        self.quantity    = quantity
        self.sl_price    = sl_price
        self.tp_price    = tp_price
        self.entry_time  = datetime.utcnow().isoformat()
        self.signal_data = signal_data   # RSI, MACD, etc. au moment de l'entrée

    def unrealized_pnl(self, current_price: float) -> float:
        if self.direction == "LONG":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def should_exit(self, current_price: float, high: float, low: float) -> Optional[str]:
        """Retourne la raison de sortie si SL/TP atteint, sinon None."""
        if self.direction == "LONG":
            if low <= self.sl_price:
                return "SL"
            if high >= self.tp_price:
                return "TP"
        else:
            if high >= self.sl_price:
                return "SL"
            if low <= self.tp_price:
                return "TP"
        return None

    def to_dict(self) -> Dict:
        return {
            "trade_id":    self.trade_id,
            "symbol":      self.symbol,
            "direction":   self.direction,
            "entry_price": self.entry_price,
            "quantity":    self.quantity,
            "sl_price":    self.sl_price,
            "tp_price":    self.tp_price,
            "entry_time":  self.entry_time,
        }


class PaperTrader:
    """
    Exécute les trades en mode simulation (paper trading).
    Aucune connexion à un exchange réel.
    """

    def __init__(self, trade_logger: TradeLogger, initial_capital: float = config.INITIAL_CAPITAL):
        self.trade_logger = trade_logger
        self.capital      = initial_capital
        self.positions: Dict[str, Position] = {}  # symbol → Position
        self._trade_counter = trade_logger.total_trade_count()

    def open_position(self, symbol: str, signal: SignalResult, quantity: float) -> Optional[str]:
        """
        Ouvre une nouvelle position simulée.
        Retourne l'ID du trade ou None si impossible.
        """
        if quantity <= 0:
            logger.warning(f"[{symbol}] Quantité invalide ({quantity}) — trade ignoré.")
            return None

        if symbol in self.positions:
            logger.info(f"[{symbol}] Position déjà ouverte — skip.")
            return None

        cost = signal.entry_price * quantity
        if cost > self.capital:
            logger.warning(
                f"[{symbol}] Capital insuffisant : coût={cost:.2f} > capital={self.capital:.2f}"
            )
            return None

        self._trade_counter += 1
        trade_id = f"T{self._trade_counter:05d}"

        self.capital -= cost
        position = Position(
            trade_id=trade_id,
            symbol=symbol,
            direction=signal.signal.value,
            entry_price=signal.entry_price,
            quantity=quantity,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            signal_data={
                "rsi_at_entry":          signal.rsi,
                "macd_at_entry":         signal.macd,
                "bb_pct_at_entry":       signal.bb_pct,
                "adx_at_entry":          signal.adx,
                "volume_ratio_at_entry": signal.volume_ratio,
            },
        )
        self.positions[symbol] = position

        logger.info(
            f"[{symbol}] OPEN {signal.signal.value} | "
            f"Prix={signal.entry_price:.4f} | Qty={quantity:.4f} | "
            f"SL={signal.sl_price:.4f} | TP={signal.tp_price:.4f} | "
            f"Coût={cost:.2f} $ | Capital restant={self.capital:.2f} $"
        )
        return trade_id

    def check_and_close_positions(
        self,
        current_prices: Dict[str, float],
        candle_data: Dict[str, Dict],
    ) -> List[Dict]:
        """
        Vérifie les SL/TP pour chaque position ouverte.
        Retourne la liste des trades fermés.
        """
        closed = []
        for symbol, position in list(self.positions.items()):
            price = current_prices.get(symbol)
            if price is None:
                continue

            candle = candle_data.get(symbol, {})
            high  = float(candle.get("high", price))
            low   = float(candle.get("low", price))

            exit_reason = position.should_exit(price, high, low)
            if exit_reason:
                exit_price = self._exit_price(position, exit_reason)
                trade_record = self._close_position(symbol, position, exit_price, exit_reason)
                closed.append(trade_record)

        return closed

    def force_close(self, symbol: str, current_price: float, reason: str = "manual") -> Optional[Dict]:
        """Force la fermeture d'une position au prix courant."""
        position = self.positions.get(symbol)
        if not position:
            return None
        return self._close_position(symbol, position, current_price, reason)

    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            price = current_prices.get(symbol)
            if price:
                total += pos.unrealized_pnl(price)
        return total

    def get_open_positions(self) -> Dict[str, Dict]:
        return {sym: pos.to_dict() for sym, pos in self.positions.items()}

    # ──────────────────────────────────────────
    # Privé
    # ──────────────────────────────────────────

    def _exit_price(self, position: Position, reason: str) -> float:
        if reason == "SL":
            return position.sl_price
        if reason == "TP":
            return position.tp_price
        return position.entry_price  # fallback

    def _close_position(
        self, symbol: str, position: Position, exit_price: float, exit_reason: str
    ) -> Dict:
        pnl = (
            (exit_price - position.entry_price) * position.quantity
            if position.direction == "LONG"
            else (position.entry_price - exit_price) * position.quantity
        )
        pnl_pct = pnl / (position.entry_price * position.quantity) if position.entry_price else 0
        self.capital += position.entry_price * position.quantity + pnl

        trade_record = {
            "id":                      position.trade_id,
            "symbol":                  symbol,
            "direction":               position.direction,
            "entry_price":             round(position.entry_price, 6),
            "exit_price":              round(exit_price, 6),
            "quantity":                round(position.quantity, 6),
            "entry_time":              position.entry_time,
            "exit_time":               datetime.utcnow().isoformat(),
            "pnl":                     round(pnl, 4),
            "pnl_pct":                 round(pnl_pct, 6),
            "sl_price":                round(position.sl_price, 6),
            "tp_price":                round(position.tp_price, 6),
            "exit_reason":             exit_reason,
            **position.signal_data,
        }

        self.trade_logger.log_trade(trade_record)
        del self.positions[symbol]

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} [{symbol}] CLOSE {position.direction} | "
            f"Raison={exit_reason} | PnL={pnl:+.2f} $ ({pnl_pct:+.2%}) | "
            f"Capital={self.capital:.2f} $"
        )
        return trade_record


class LiveTrader:
    """
    Exécute les trades en mode live via l'API Alpaca.
    Nécessite ALPACA_API_KEY et ALPACA_SECRET_KEY dans le .env.
    """

    def __init__(self, trade_logger: TradeLogger):
        self.trade_logger = trade_logger
        self._api = None
        self._connect()

    def _connect(self, max_retries: int = 4):
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            logger.error(
                "Clés Alpaca manquantes. Ajoutez ALPACA_API_KEY et ALPACA_SECRET_KEY dans .env"
            )
            return

        wait = 2
        for attempt in range(1, max_retries + 1):
            try:
                import alpaca_trade_api as tradeapi
                self._api = tradeapi.REST(
                    config.ALPACA_API_KEY,
                    config.ALPACA_SECRET_KEY,
                    config.ALPACA_BASE_URL,
                    api_version="v2",
                )
                account = self._api.get_account()
                logger.info(
                    f"Alpaca connecté ({config.TRADING_MODE.upper()}) | "
                    f"Équité : {account.equity} $ | Buying power : {account.buying_power} $"
                )
                return
            except Exception as e:
                logger.warning(f"Alpaca connexion tentative {attempt}/{max_retries} : {e}")
                if attempt < max_retries:
                    time.sleep(wait)
                    wait *= 2
        logger.error("Impossible de se connecter à Alpaca.")

    def is_connected(self) -> bool:
        return self._api is not None

    def get_buying_power(self) -> float:
        if not self.is_connected():
            return 0.0
        try:
            return float(self._api.get_account().buying_power)
        except Exception:
            return 0.0

    def open_position(self, symbol: str, signal: SignalResult, quantity: float) -> Optional[str]:
        if not self.is_connected() or quantity <= 0:
            return None

        side = "buy" if signal.signal == Signal.LONG else "sell"
        wait = 2
        for attempt in range(1, 5):
            try:
                order = self._api.submit_order(
                    symbol=symbol,
                    qty=round(quantity, 2),
                    side=side,
                    type="market",
                    time_in_force="day",
                )
                logger.info(
                    f"[{symbol}] Ordre {side.upper()} soumis | "
                    f"Qty={quantity:.2f} | OrderID={order.id}"
                )
                return order.id
            except Exception as e:
                logger.warning(f"[{symbol}] Erreur ordre tentative {attempt} : {e}")
                if attempt < 4:
                    time.sleep(wait)
                    wait *= 2
        return None

    def close_position(self, symbol: str) -> bool:
        if not self.is_connected():
            return False
        wait = 2
        for attempt in range(1, 5):
            try:
                self._api.close_position(symbol)
                logger.info(f"[{symbol}] Position fermée via Alpaca.")
                return True
            except Exception as e:
                logger.warning(f"[{symbol}] Erreur fermeture tentative {attempt} : {e}")
                if attempt < 4:
                    time.sleep(wait)
                    wait *= 2
        return False

    def get_open_positions(self) -> List[Dict]:
        if not self.is_connected():
            return []
        try:
            positions = self._api.list_positions()
            return [
                {
                    "symbol":       p.symbol,
                    "qty":          float(p.qty),
                    "unrealized_pl": float(p.unrealized_pl),
                    "current_price": float(p.current_price),
                }
                for p in positions
            ]
        except Exception as e:
            logger.warning(f"Erreur lecture positions Alpaca : {e}")
            return []


def create_trader(trade_logger: TradeLogger):
    """Factory : retourne PaperTrader ou LiveTrader selon la config."""
    if config.TRADING_MODE == "live":
        logger.info("Mode LIVE activé — connexion à Alpaca...")
        trader = LiveTrader(trade_logger)
        if not trader.is_connected():
            logger.warning("Fallback vers Paper Trading (connexion Alpaca échouée).")
            return PaperTrader(trade_logger)
        return trader
    else:
        logger.info("Mode PAPER TRADING activé.")
        return PaperTrader(trade_logger)
