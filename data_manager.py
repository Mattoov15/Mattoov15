"""
Récupération et mise en cache des données OHLCV via yfinance.
Retry automatique avec backoff exponentiel.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger("trading_bot")

# Intervalles yfinance valides
YFINANCE_INTERVALS = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "60m", "1d": "1d",
}


class DataManager:
    """Gère le téléchargement et le cache mémoire des données OHLCV."""

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._last_fetch: Dict[str, datetime] = {}
        # Cache TTL par timeframe (en secondes)
        self._ttl = {
            "1m": 60, "5m": 300, "15m": 900,
            "30m": 1800, "1h": 3600, "1d": 86400,
        }

    # ──────────────────────────────────────────
    # Interface publique
    # ──────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = config.TIMEFRAME,
        days: int = config.HISTORY_DAYS,
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Retourne un DataFrame OHLCV pour le symbole donné.
        Utilise le cache si les données sont encore fraîches.
        """
        cache_key = f"{symbol}_{timeframe}"
        ttl = self._ttl.get(timeframe, 3600)

        if not force_refresh and self._is_cache_valid(cache_key, ttl):
            return self._cache[cache_key]

        df = self._fetch_with_retry(symbol, timeframe, days)
        if df is not None and not df.empty:
            self._cache[cache_key] = df
            self._last_fetch[cache_key] = datetime.utcnow()
        return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Retourne le dernier prix de clôture disponible."""
        df = self.get_ohlcv(symbol, timeframe=config.TIMEFRAME, days=5)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
        return None

    def refresh_all(self):
        """Force le rafraîchissement de tous les symboles configurés."""
        for market_name, symbols in config.SYMBOLS.items():
            sym = symbols["yfinance"]
            self.get_ohlcv(sym, force_refresh=True)
            logger.info(f"Données rafraîchies pour {market_name} ({sym})")

    # ──────────────────────────────────────────
    # Interne : fetch + retry
    # ──────────────────────────────────────────

    def _fetch_with_retry(
        self, symbol: str, timeframe: str, days: int, max_retries: int = 4
    ) -> Optional[pd.DataFrame]:
        interval = YFINANCE_INTERVALS.get(timeframe, "60m")
        end = datetime.utcnow()
        start = end - timedelta(days=days)

        wait = 2
        for attempt in range(1, max_retries + 1):
            try:
                df = yf.download(
                    symbol,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
                if df is not None and not df.empty:
                    df = self._clean(df)
                    logger.debug(
                        f"Données OK : {symbol} | {len(df)} bougies "
                        f"({start.date()} → {end.date()})"
                    )
                    return df
                logger.warning(f"[{symbol}] DataFrame vide (tentative {attempt})")
            except Exception as e:
                logger.warning(
                    f"[{symbol}] Erreur fetch tentative {attempt}/{max_retries} : {e}"
                )
            if attempt < max_retries:
                time.sleep(wait)
                wait *= 2

        logger.error(f"[{symbol}] Impossible de récupérer les données après {max_retries} tentatives.")
        return None

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage : colonnes standardisées, suppression des NaN, tri chronologique."""
        # Aplatir les colonnes multi-index si yfinance retourne un MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(subset=["Close", "Volume"], inplace=True)
        df.sort_index(inplace=True)

        # Validation basique : prix strictement positifs
        df = df[(df["Close"] > 0) & (df["Volume"] >= 0)]
        return df

    def _is_cache_valid(self, key: str, ttl: int) -> bool:
        if key not in self._cache or key not in self._last_fetch:
            return False
        age = (datetime.utcnow() - self._last_fetch[key]).total_seconds()
        return age < ttl
