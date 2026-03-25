"""
Gestion du risque : position sizing (Kelly), stop-loss,
drawdown maximum et limite de perte journalière.
"""

import logging
from datetime import date
from typing import Optional

import config

logger = logging.getLogger("trading_bot")


class RiskManager:
    """
    Contrôle le risque de chaque trade et de l'ensemble du portefeuille.
    """

    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL):
        self.capital       = initial_capital
        self.peak_capital  = initial_capital
        self.daily_loss    = 0.0
        self._last_day     = date.today()
        self._trading_halted = False

    # ──────────────────────────────────────────
    # Mise à jour de l'état
    # ──────────────────────────────────────────

    def update_capital(self, pnl: float):
        """Appelé après chaque trade fermé."""
        self._reset_daily_if_new_day()
        self.capital += pnl
        if pnl < 0:
            self.daily_loss += abs(pnl)

        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        logger.info(
            f"Capital mis à jour : {self.capital:.2f} $ | "
            f"Perte journalière : {self.daily_loss:.2f} $ | "
            f"Peak : {self.peak_capital:.2f} $"
        )
        self._check_limits()

    def record_unrealized_pnl(self, unrealized: float):
        """Pour vérification du drawdown sur position ouverte."""
        effective_capital = self.capital + unrealized
        if self.peak_capital > 0:
            dd = (self.peak_capital - effective_capital) / self.peak_capital
            if dd >= config.MAX_DRAWDOWN_PCT:
                logger.warning(
                    f"Drawdown max atteint ({dd:.1%}) — trading suspendu."
                )
                self._trading_halted = True

    # ──────────────────────────────────────────
    # Calcul de la taille de position
    # ──────────────────────────────────────────

    def compute_position_size(
        self,
        entry_price: float,
        sl_price:    float,
        win_rate:    float = 0.55,    # Taux de réussite estimé (via ML ou défaut)
        rr_ratio:    float = None,    # Ratio Risk/Reward (calculé si None)
    ) -> float:
        """
        Calcule le nombre de parts à acheter via le Critère de Kelly fractionné.

        Kelly = (win_rate × rr - loss_rate) / rr
        Position $ = Kelly_fraction × Kelly × Capital
        Quantité = Position $ / Entry_price

        Retourne 0.0 si le trade ne passe pas les contrôles de risque.
        """
        if not self.can_trade():
            return 0.0

        risk_per_share = abs(entry_price - sl_price)
        if risk_per_share <= 0:
            logger.warning("Risk per share invalide (0 ou négatif) — trade ignoré.")
            return 0.0

        if rr_ratio is None:
            reward_per_share = abs(sl_price - entry_price) * (
                config.ATR_TP_MULT / config.ATR_SL_MULT
            )
            rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 1.667

        loss_rate = 1.0 - win_rate
        kelly_raw = (win_rate * rr_ratio - loss_rate) / rr_ratio
        kelly_raw = max(0.0, kelly_raw)                 # Kelly négatif → pas de trade

        kelly_frac  = kelly_raw * config.KELLY_FRACTION
        max_position_frac = config.MAX_POSITION_PCT

        position_frac = min(kelly_frac, max_position_frac)
        position_dollars = position_frac * self.capital

        quantity = position_dollars / entry_price
        quantity = max(0.0, quantity)

        logger.debug(
            f"Kelly brut={kelly_raw:.4f} | Kelly×{config.KELLY_FRACTION}={kelly_frac:.4f} "
            f"| Position={position_frac:.2%} | ${position_dollars:.2f} | qty={quantity:.4f}"
        )
        return quantity

    # ──────────────────────────────────────────
    # Contrôles
    # ──────────────────────────────────────────

    def can_trade(self) -> bool:
        """Retourne True si le trading est autorisé."""
        self._reset_daily_if_new_day()
        if self._trading_halted:
            logger.warning("Trading suspendu (drawdown max ou perte journalière).")
            return False
        return True

    def reset_halt(self):
        """Réinitialise la suspension (à appeler manuellement ou en début de journée)."""
        self._trading_halted = False
        logger.info("Suspension de trading levée.")

    # ──────────────────────────────────────────
    # Helpers privés
    # ──────────────────────────────────────────

    def _check_limits(self):
        # Perte journalière
        daily_limit = config.DAILY_LOSS_LIMIT_PCT * self.peak_capital
        if self.daily_loss >= daily_limit:
            logger.warning(
                f"Limite de perte journalière atteinte "
                f"({self.daily_loss:.2f} $ >= {daily_limit:.2f} $) — trading suspendu."
            )
            self._trading_halted = True

        # Drawdown maximum
        if self.peak_capital > 0:
            dd = (self.peak_capital - self.capital) / self.peak_capital
            if dd >= config.MAX_DRAWDOWN_PCT:
                logger.warning(
                    f"Drawdown maximum atteint ({dd:.1%}) — trading suspendu."
                )
                self._trading_halted = True

    def _reset_daily_if_new_day(self):
        today = date.today()
        if today != self._last_day:
            self.daily_loss = 0.0
            self._last_day  = today
            # Lever la suspension de perte journalière (pas celle de drawdown)
            if self._trading_halted:
                dd = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital else 0
                if dd < config.MAX_DRAWDOWN_PCT:
                    self._trading_halted = False
                    logger.info("Nouvelle journée — suspension levée.")

    # ──────────────────────────────────────────
    # Informations
    # ──────────────────────────────────────────

    def status(self) -> dict:
        dd = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital else 0
        return {
            "capital":          round(self.capital, 2),
            "peak_capital":     round(self.peak_capital, 2),
            "daily_loss":       round(self.daily_loss, 2),
            "drawdown":         round(dd, 4),
            "trading_halted":   self._trading_halted,
        }
