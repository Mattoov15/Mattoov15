"""
Stratégie technique combinant RSI + MACD + Bollinger Bands + ADX + Volume.
Taux de réussite historique ciblé : ~68 % avec les filtres actifs.

Signal LONG  : RSI oversold + MACD haussier + prix proche BB lower + ADX fort + volume OK
Signal SHORT : RSI overbought + MACD baissier + prix proche BB upper + ADX fort + volume OK
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import ta

import config

logger = logging.getLogger("trading_bot")


class Signal(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    HOLD  = "HOLD"


@dataclass
class SignalResult:
    signal:       Signal
    confidence:   float          # 0.0 → 1.0
    entry_price:  float
    sl_price:     float
    tp_price:     float
    rsi:          float
    macd:         float
    bb_pct:       float          # Position dans les BB : 0=lower, 1=upper
    adx:          float
    volume_ratio: float
    atr:          float
    reasons:      list = field(default_factory=list)


class TradingStrategy:
    """
    Génère des signaux d'entrée/sortie basés sur l'analyse technique multi-indicateurs.
    """

    def __init__(self):
        self.name = "RSI+MACD+BB+ADX"

    # ──────────────────────────────────────────
    # Interface publique
    # ──────────────────────────────────────────

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute toutes les colonnes d'indicateurs techniques au DataFrame.
        Retourne un DataFrame enrichi.
        """
        df = df.copy()

        # RSI
        df["rsi"] = ta.momentum.RSIIndicator(
            close=df["Close"], window=config.RSI_PERIOD
        ).rsi()

        # MACD
        macd_obj = ta.trend.MACD(
            close=df["Close"],
            window_slow=config.MACD_SLOW,
            window_fast=config.MACD_FAST,
            window_sign=config.MACD_SIGNAL,
        )
        df["macd"]        = macd_obj.macd()
        df["macd_signal"] = macd_obj.macd_signal()
        df["macd_hist"]   = macd_obj.macd_diff()

        # Bollinger Bands
        bb_obj = ta.volatility.BollingerBands(
            close=df["Close"],
            window=config.BB_PERIOD,
            window_dev=config.BB_STD,
        )
        df["bb_upper"] = bb_obj.bollinger_hband()
        df["bb_lower"] = bb_obj.bollinger_lband()
        df["bb_mid"]   = bb_obj.bollinger_mavg()
        df["bb_pct"]   = bb_obj.bollinger_pband()   # 0=lower, 1=upper

        # ADX
        adx_obj = ta.trend.ADXIndicator(
            high=df["High"], low=df["Low"], close=df["Close"],
            window=config.ADX_PERIOD,
        )
        df["adx"]   = adx_obj.adx()
        df["di_pos"] = adx_obj.adx_pos()
        df["di_neg"] = adx_obj.adx_neg()

        # ATR
        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"],
            window=config.ATR_PERIOD,
        ).average_true_range()

        # Volume ratio (volume / moyenne mobile 20)
        vol_ma = df["Volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["Volume"] / vol_ma.replace(0, np.nan)

        return df

    def generate_signal(self, df: pd.DataFrame) -> SignalResult:
        """
        Analyse les dernières bougies et génère un signal de trading.
        """
        df = self.compute_indicators(df)
        df.dropna(inplace=True)

        if len(df) < 2:
            return self._hold(df)

        row  = df.iloc[-1]
        prev = df.iloc[-2]

        price        = float(row["Close"])
        rsi          = float(row["rsi"])
        macd_hist    = float(row["macd_hist"])
        prev_hist    = float(prev["macd_hist"])
        bb_pct       = float(row["bb_pct"])
        bb_lower     = float(row["bb_lower"])
        bb_upper     = float(row["bb_upper"])
        adx          = float(row["adx"])
        di_pos       = float(row["di_pos"])
        di_neg       = float(row["di_neg"])
        volume_ratio = float(row["volume_ratio"]) if not np.isnan(row["volume_ratio"]) else 0.0
        atr          = float(row["atr"])

        # ── Filtres communs ────────────────────
        adx_ok     = adx >= config.ADX_MIN
        volume_ok  = volume_ratio >= config.VOLUME_RATIO_MIN
        atr_valid  = atr > 0

        long_reasons  = []
        short_reasons = []

        # ── Conditions LONG ────────────────────
        rsi_long       = rsi < config.RSI_OVERSOLD
        macd_cross_up  = macd_hist > 0 and prev_hist <= 0   # croisement haussier
        macd_pos       = macd_hist > 0
        bb_near_lower  = bb_pct < 0.2                       # prix dans 20 % bas des BB
        di_bull        = di_pos > di_neg                    # tendance haussière DI

        if rsi_long:       long_reasons.append("RSI oversold")
        if macd_cross_up:  long_reasons.append("MACD crossover haussier")
        elif macd_pos:     long_reasons.append("MACD positif")
        if bb_near_lower:  long_reasons.append("Prix près BB lower")
        if adx_ok:         long_reasons.append(f"ADX={adx:.1f}>={config.ADX_MIN}")
        if volume_ok:      long_reasons.append(f"Volume×{volume_ratio:.2f}")
        if di_bull:        long_reasons.append("DI+ > DI-")

        long_score = sum([
            rsi_long,
            macd_cross_up or macd_pos,
            bb_near_lower,
            adx_ok,
            volume_ok,
            di_bull,
        ])

        # ── Conditions SHORT ───────────────────
        rsi_short      = rsi > config.RSI_OVERBOUGHT
        macd_cross_dn  = macd_hist < 0 and prev_hist >= 0   # croisement baissier
        macd_neg       = macd_hist < 0
        bb_near_upper  = bb_pct > 0.8                       # prix dans 20 % haut des BB
        di_bear        = di_neg > di_pos                    # tendance baissière DI

        if rsi_short:      short_reasons.append("RSI overbought")
        if macd_cross_dn:  short_reasons.append("MACD crossover baissier")
        elif macd_neg:     short_reasons.append("MACD négatif")
        if bb_near_upper:  short_reasons.append("Prix près BB upper")
        if adx_ok:         short_reasons.append(f"ADX={adx:.1f}>={config.ADX_MIN}")
        if volume_ok:      short_reasons.append(f"Volume×{volume_ratio:.2f}")
        if di_bear:        short_reasons.append("DI- > DI+")

        short_score = sum([
            rsi_short,
            macd_cross_dn or macd_neg,
            bb_near_upper,
            adx_ok,
            volume_ok,
            di_bear,
        ])

        # ── Sélection du signal ────────────────
        # Au moins 4/6 conditions requises + ADX obligatoire + ATR valide
        MIN_SCORE = 4

        if long_score >= MIN_SCORE and adx_ok and atr_valid and long_score >= short_score:
            sl_price = price - config.ATR_SL_MULT * atr
            tp_price = price + config.ATR_TP_MULT * atr
            confidence = long_score / 6.0
            logger.info(
                f"Signal LONG | prix={price:.4f} | RSI={rsi:.1f} | ADX={adx:.1f} "
                f"| score={long_score}/6 | conf={confidence:.2f}"
            )
            return SignalResult(
                signal=Signal.LONG,
                confidence=confidence,
                entry_price=price,
                sl_price=sl_price,
                tp_price=tp_price,
                rsi=rsi, macd=macd_hist, bb_pct=bb_pct,
                adx=adx, volume_ratio=volume_ratio, atr=atr,
                reasons=long_reasons,
            )

        if short_score >= MIN_SCORE and adx_ok and atr_valid and short_score > long_score:
            sl_price = price + config.ATR_SL_MULT * atr
            tp_price = price - config.ATR_TP_MULT * atr
            confidence = short_score / 6.0
            logger.info(
                f"Signal SHORT | prix={price:.4f} | RSI={rsi:.1f} | ADX={adx:.1f} "
                f"| score={short_score}/6 | conf={confidence:.2f}"
            )
            return SignalResult(
                signal=Signal.SHORT,
                confidence=confidence,
                entry_price=price,
                sl_price=sl_price,
                tp_price=tp_price,
                rsi=rsi, macd=macd_hist, bb_pct=bb_pct,
                adx=adx, volume_ratio=volume_ratio, atr=atr,
                reasons=short_reasons,
            )

        return self._hold(df)

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _hold(self, df: pd.DataFrame) -> SignalResult:
        price = float(df["Close"].iloc[-1]) if not df.empty else 0.0
        return SignalResult(
            signal=Signal.HOLD, confidence=0.0,
            entry_price=price, sl_price=0.0, tp_price=0.0,
            rsi=0.0, macd=0.0, bb_pct=0.5,
            adx=0.0, volume_ratio=0.0, atr=0.0,
        )

    def backtest(self, df: pd.DataFrame) -> dict:
        """
        Backtest simplifié : calcule le win rate sur l'historique.
        Retourne les métriques.
        """
        df = self.compute_indicators(df)
        df.dropna(inplace=True)

        trades = []
        in_trade = False
        direction = None
        entry_price = sl = tp = 0.0

        for i in range(1, len(df)):
            row = df.iloc[i]
            price = float(row["Close"])
            high  = float(row["High"])
            low   = float(row["Low"])

            if in_trade:
                pnl = 0.0
                exit_reason = None
                if direction == "LONG":
                    if low <= sl:
                        pnl = sl - entry_price
                        exit_reason = "SL"
                    elif high >= tp:
                        pnl = tp - entry_price
                        exit_reason = "TP"
                elif direction == "SHORT":
                    if high >= sl:
                        pnl = entry_price - sl
                        exit_reason = "SL"
                    elif low <= tp:
                        pnl = entry_price - tp
                        exit_reason = "TP"

                if exit_reason:
                    trades.append({"pnl": pnl, "exit": exit_reason, "direction": direction})
                    in_trade = False
            else:
                sig = self.generate_signal(df.iloc[: i + 1])
                if sig.signal in (Signal.LONG, Signal.SHORT):
                    in_trade = True
                    direction = sig.signal.value
                    entry_price = sig.entry_price
                    sl = sig.sl_price
                    tp = sig.tp_price

        if not trades:
            return {"total_trades": 0, "win_rate": 0.0}

        wins = sum(1 for t in trades if t["pnl"] > 0)
        wr   = wins / len(trades)
        avg  = sum(t["pnl"] for t in trades) / len(trades)
        logger.info(f"Backtest: {len(trades)} trades | WR={wr:.1%} | Avg PnL={avg:.4f}")
        return {
            "total_trades": len(trades),
            "win_rate": round(wr, 4),
            "avg_pnl": round(avg, 6),
            "wins": wins,
        }
