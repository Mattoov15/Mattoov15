"""
Gestion du risque : position sizing (Kelly), stop-loss,
drawdown maximum et limite de perte journalière.

En mode LIVE, des limites plus strictes s'appliquent automatiquement
(voir config.py section LIVE CAPITAL LIMITS).
"""

import logging
from datetime import date
from typing import Optional

import config

logger = logging.getLogger("trading_bot")

_IS_LIVE = config.TRADING_MODE == "live"   # Les limites strictes s'appliquent uniquement en "live"


class RiskManager:
    """
    Contrôle le risque de chaque trade et de l'ensemble du portefeuille.
    Applique des limites différentes selon le mode paper / live.
    """

    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL):
        self.capital       = initial_capital
        self.peak_capital  = initial_capital
        self.daily_loss    = 0.0
        self.weekly_loss   = 0.0
        self._last_day     = date.today()
        self._last_week    = date.today().isocalendar()[1]   # numéro de semaine ISO
        self._trading_halted = False
        self.open_positions_count = 0      # positions actuellement ouvertes
        self.deployed_capital     = 0.0    # capital actuellement en marché

        # ── Sélection des limites selon le mode ──────────────────────────
        if _IS_LIVE:
            self._max_position_pct  = config.LIVE_MAX_POSITION_PCT
            self._daily_limit_pct   = config.LIVE_DAILY_LOSS_LIMIT_PCT
            self._max_drawdown_pct  = config.LIVE_MAX_DRAWDOWN_PCT
            self._kelly_fraction    = config.LIVE_KELLY_FRACTION
            self._max_positions     = config.LIVE_MAX_SIMULTANEOUS_POSITIONS
            self._weekly_limit_pct  = config.LIVE_WEEKLY_LOSS_LIMIT_PCT
            self._max_deployed_pct  = config.LIVE_MAX_DEPLOYED_CAPITAL_PCT
            logger.warning(
                "⚠️  MODE LIVE ACTIVÉ — Limites strictes appliquées : "
                f"pos≤{self._max_position_pct:.0%} | "
                f"perte/jour≤{self._daily_limit_pct:.1%} | "
                f"drawdown≤{self._max_drawdown_pct:.0%} | "
                f"Kelly×{self._kelly_fraction} | "
                f"max {self._max_positions} positions | "
                f"déployé≤{self._max_deployed_pct:.0%} | "
                f"perte/sem≤{self._weekly_limit_pct:.0%}"
            )
        else:
            self._max_position_pct  = config.MAX_POSITION_PCT
            self._daily_limit_pct   = config.DAILY_LOSS_LIMIT_PCT
            self._max_drawdown_pct  = config.MAX_DRAWDOWN_PCT
            self._kelly_fraction    = config.KELLY_FRACTION
            self._max_positions     = 999          # illimité en paper
            self._weekly_limit_pct  = None         # pas de limite hebdo en paper
            self._max_deployed_pct  = 1.0          # 100 % autorisé en paper
            logger.info("Mode PAPER TRADING — limites standard appliquées.")

    # ──────────────────────────────────────────
    # Gestion des positions ouvertes
    # ──────────────────────────────────────────

    def increment_positions(self, cost: float = 0.0):
        """Appelé à l'ouverture d'une position."""
        self.open_positions_count += 1
        self.deployed_capital     += cost

    def decrement_positions(self, cost: float = 0.0):
        """Appelé à la fermeture d'une position."""
        self.open_positions_count = max(0, self.open_positions_count - 1)
        self.deployed_capital     = max(0.0, self.deployed_capital - cost)

    # ──────────────────────────────────────────
    # Vérification d'aptitude au live
    # ──────────────────────────────────────────

    def check_live_readiness(self, trade_count: int) -> bool:
        """
        En mode live, vérifie que le bot a accumulé suffisamment de trades paper
        pour que le modèle ML soit fiable.
        Retourne False (et logue une alerte) si pas prêt.
        """
        if not _IS_LIVE:
            return True
        if trade_count < config.LIVE_MIN_PAPER_TRADES:
            logger.error(
                f"🚫 LIVE BLOQUÉ : seulement {trade_count} trades dans l'historique. "
                f"Minimum requis : {config.LIVE_MIN_PAPER_TRADES}. "
                "Continuez en paper trading jusqu'à atteindre ce seuil."
            )
            return False
        logger.info(
            f"✅ Bot prêt pour le live ({trade_count} trades dans l'historique)."
        )
        return True

    # ──────────────────────────────────────────
    # Mise à jour de l'état
    # ──────────────────────────────────────────

    def update_capital(self, pnl: float, position_cost: float = 0.0):
        """Appelé après chaque trade fermé."""
        self._reset_daily_if_new_day()
        self._reset_weekly_if_new_week()

        self.capital += pnl
        self.decrement_positions(position_cost)

        if pnl < 0:
            self.daily_loss  += abs(pnl)
            self.weekly_loss += abs(pnl)

        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        logger.info(
            f"Capital : {self.capital:.2f} $ | "
            f"Perte jour : {self.daily_loss:.2f} $ | "
            f"Perte semaine : {self.weekly_loss:.2f} $ | "
            f"Peak : {self.peak_capital:.2f} $"
        )
        self._check_limits()

    def record_unrealized_pnl(self, unrealized: float):
        """Pour vérification du drawdown sur position ouverte."""
        effective_capital = self.capital + unrealized
        if self.peak_capital > 0:
            dd = (self.peak_capital - effective_capital) / self.peak_capital
            if dd >= self._max_drawdown_pct:
                logger.warning(
                    f"Drawdown max atteint ({dd:.1%} >= {self._max_drawdown_pct:.0%}) "
                    "— trading suspendu."
                )
                self._trading_halted = True

    # ──────────────────────────────────────────
    # Calcul de la taille de position
    # ──────────────────────────────────────────

    def compute_position_size(
        self,
        entry_price: float,
        sl_price:    float,
        win_rate:    float = 0.55,
        rr_ratio:    float = None,
    ) -> float:
        """
        Calcule la quantité à acheter via le Critère de Kelly fractionné.
        Applique toutes les limites du mode actif (paper ou live).
        Retourne 0.0 si le trade ne passe pas les contrôles.
        """
        if not self.can_trade():
            return 0.0

        # ── Vérification max positions simultanées ────────────────────────
        if self.open_positions_count >= self._max_positions:
            logger.warning(
                f"Limite de positions simultanées atteinte "
                f"({self.open_positions_count}/{self._max_positions}) — trade ignoré."
            )
            return 0.0

        # ── Vérification capital déployé maximum ──────────────────────────
        max_deployable = self._max_deployed_pct * self.capital
        if self.deployed_capital >= max_deployable:
            logger.warning(
                f"Capital déployé maximum atteint "
                f"({self.deployed_capital:.2f} $ >= {max_deployable:.2f} $) — trade ignoré."
            )
            return 0.0

        risk_per_share = abs(entry_price - sl_price)
        if risk_per_share <= 0:
            logger.warning("Risk per share invalide (0 ou négatif) — trade ignoré.")
            return 0.0

        if rr_ratio is None:
            rr_ratio = config.ATR_TP_MULT / config.ATR_SL_MULT   # 1.667

        loss_rate = 1.0 - win_rate
        kelly_raw = (win_rate * rr_ratio - loss_rate) / rr_ratio
        kelly_raw = max(0.0, kelly_raw)

        kelly_frac     = kelly_raw * self._kelly_fraction
        position_frac  = min(kelly_frac, self._max_position_pct)

        # Respecter aussi le capital déployable restant
        remaining_deployable = max_deployable - self.deployed_capital
        position_dollars = min(position_frac * self.capital, remaining_deployable)
        position_dollars = max(0.0, position_dollars)

        quantity = position_dollars / entry_price

        logger.debug(
            f"Kelly brut={kelly_raw:.4f} | ×{self._kelly_fraction}={kelly_frac:.4f} "
            f"| pos={position_frac:.2%} | ${position_dollars:.2f} | qty={quantity:.4f} "
            f"| mode={'LIVE' if _IS_LIVE else 'PAPER'}"
        )
        return quantity

    # ──────────────────────────────────────────
    # Contrôles
    # ──────────────────────────────────────────

    def can_trade(self) -> bool:
        self._reset_daily_if_new_day()
        self._reset_weekly_if_new_week()
        if self._trading_halted:
            logger.warning("Trading suspendu (limite atteinte).")
            return False
        return True

    def reset_halt(self):
        self._trading_halted = False
        logger.info("Suspension de trading levée.")

    # ──────────────────────────────────────────
    # Helpers privés
    # ──────────────────────────────────────────

    def _check_limits(self):
        # Perte journalière
        daily_limit = self._daily_limit_pct * self.peak_capital
        if self.daily_loss >= daily_limit:
            logger.warning(
                f"{'⚠️ LIVE' if _IS_LIVE else ''} Limite perte/jour atteinte "
                f"({self.daily_loss:.2f} $ >= {daily_limit:.2f} $) — trading suspendu."
            )
            self._trading_halted = True

        # Perte hebdomadaire (live uniquement)
        if self._weekly_limit_pct is not None:
            weekly_limit = self._weekly_limit_pct * self.peak_capital
            if self.weekly_loss >= weekly_limit:
                logger.warning(
                    f"⚠️ LIVE Limite perte/semaine atteinte "
                    f"({self.weekly_loss:.2f} $ >= {weekly_limit:.2f} $) "
                    "— trading suspendu jusqu'au lundi suivant."
                )
                self._trading_halted = True

        # Drawdown maximum
        if self.peak_capital > 0:
            dd = (self.peak_capital - self.capital) / self.peak_capital
            if dd >= self._max_drawdown_pct:
                logger.warning(
                    f"{'⚠️ LIVE' if _IS_LIVE else ''} Drawdown max atteint "
                    f"({dd:.1%} >= {self._max_drawdown_pct:.0%}) — trading suspendu."
                )
                self._trading_halted = True

    def _reset_daily_if_new_day(self):
        today = date.today()
        if today != self._last_day:
            self.daily_loss = 0.0
            self._last_day  = today
            if self._trading_halted:
                dd = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital else 0
                if dd < self._max_drawdown_pct and self.weekly_loss < (
                    (self._weekly_limit_pct or 1.0) * self.peak_capital
                ):
                    self._trading_halted = False
                    logger.info("Nouvelle journée — suspension levée.")

    def _reset_weekly_if_new_week(self):
        current_week = date.today().isocalendar()[1]
        if current_week != self._last_week:
            self.weekly_loss = 0.0
            self._last_week  = current_week
            # Lever la suspension hebdomadaire si drawdown OK
            if self._trading_halted and _IS_LIVE:
                dd = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital else 0
                if dd < self._max_drawdown_pct:
                    self._trading_halted = False
                    logger.info("Nouvelle semaine — suspension hebdomadaire levée.")

    # ──────────────────────────────────────────
    # Informations
    # ──────────────────────────────────────────

    def status(self) -> dict:
        dd = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital else 0
        return {
            "mode":                   "live" if _IS_LIVE else "paper",
            "capital":                round(self.capital, 2),
            "peak_capital":           round(self.peak_capital, 2),
            "deployed_capital":       round(self.deployed_capital, 2),
            "open_positions":         self.open_positions_count,
            "daily_loss":             round(self.daily_loss, 2),
            "weekly_loss":            round(self.weekly_loss, 2),
            "drawdown":               round(dd, 4),
            "trading_halted":         self._trading_halted,
            "limits": {
                "max_position_pct":   self._max_position_pct,
                "daily_loss_pct":     self._daily_limit_pct,
                "max_drawdown_pct":   self._max_drawdown_pct,
                "kelly_fraction":     self._kelly_fraction,
                "max_positions":      self._max_positions,
                "weekly_loss_pct":    self._weekly_limit_pct,
                "max_deployed_pct":   self._max_deployed_pct,
            },
        }
