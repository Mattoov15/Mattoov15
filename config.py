"""
Configuration centrale du bot de trading.
Modifiez ce fichier ou créez un .env pour surcharger les valeurs.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# MODE : "paper" ou "live"
# ─────────────────────────────────────────────
TRADING_MODE = os.getenv("TRADING_MODE", "paper")   # "paper" | "live"

# ─────────────────────────────────────────────
# ALPACA (paper & live trading)
# ─────────────────────────────────────────────
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = (
    "https://paper-api.alpaca.markets"
    if TRADING_MODE == "paper"
    else "https://api.alpaca.markets"
)

# ─────────────────────────────────────────────
# MARCHÉS — symboles Alpaca / yfinance
# ─────────────────────────────────────────────
SYMBOLS = {
    "or":        {"alpaca": "GLD",  "yfinance": "GLD"},   # ETF Gold
    "nasdaq":    {"alpaca": "QQQ",  "yfinance": "QQQ"},   # ETF NASDAQ 100
    "dow_jones": {"alpaca": "DIA",  "yfinance": "DIA"},   # ETF Dow Jones
}

# ─────────────────────────────────────────────
# STRATÉGIE TECHNIQUE
# ─────────────────────────────────────────────
TIMEFRAME         = "1h"          # Intervalle de bougie
HISTORY_DAYS      = 90            # Jours d'historique à charger

RSI_PERIOD        = 14
RSI_OVERSOLD      = 35
RSI_OVERBOUGHT    = 65

MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9

BB_PERIOD         = 20
BB_STD            = 2.0

ADX_PERIOD        = 14
ADX_MIN           = 20            # Filtre : ADX > 20 pour valider une tendance

ATR_PERIOD        = 14
ATR_SL_MULT       = 1.5           # Stop-loss = ATR × multiplicateur
ATR_TP_MULT       = 2.5           # Take-profit = ATR × multiplicateur

VOLUME_RATIO_MIN  = 1.1           # Volume doit être > 110 % de sa moyenne

# ─────────────────────────────────────────────
# RISK MANAGEMENT
# ─────────────────────────────────────────────
INITIAL_CAPITAL        = 100_000.0   # Capital initial en $ (défaut Alpaca paper)
MAX_POSITION_PCT       = 0.10        # Max 10 % du capital par trade
DAILY_LOSS_LIMIT_PCT   = 0.03        # Arrêt si perte journalière > 3 %
MAX_DRAWDOWN_PCT       = 0.10        # Arrêt si drawdown depuis pic > 10 %
KELLY_FRACTION         = 0.5         # Fraction Kelly (0.5 = demi-Kelly, prudent)

# ─────────────────────────────────────────────
# LIMITES STRICTES POUR LE MODE LIVE
# Remplacent automatiquement les params paper ci-dessus quand TRADING_MODE == "live"
# Raisonnement :
#   5 %  position  → une erreur en live coûte réellement, moitié moins qu'en paper
#   1.5 % /jour    → force une pause et une révision humaine avant de continuer
#   6 %  drawdown  → alarme précoce, bien avant de toucher du capital irrémédiable
#   Kelly × 0.25   → quart de Kelly = standard institutionnel sur stratégie non prouvée live
#   2 positions    → GLD/QQQ/DIA sont corrélés, limiter la sur-exposition simultanée
#   3 % /semaine   → filet hebdomadaire : suspension auto jusqu'au lundi suivant
#   25 % déployés  → 75 % du capital TOUJOURS en cash comme tampon de sécurité
#   50 trades min  → le modèle ML a besoin d'au moins 50 trades pour être fiable
# ─────────────────────────────────────────────
LIVE_MAX_POSITION_PCT           = 0.05    # 5 % max du capital par trade
LIVE_DAILY_LOSS_LIMIT_PCT       = 0.015   # 1.5 % de perte/jour → arrêt immédiat
LIVE_MAX_DRAWDOWN_PCT           = 0.06    # 6 % de drawdown depuis le pic → arrêt
LIVE_KELLY_FRACTION             = 0.25    # Quart de Kelly (vs 0.5 en paper)
LIVE_MAX_SIMULTANEOUS_POSITIONS = 2       # Max 2 positions ouvertes en même temps
LIVE_WEEKLY_LOSS_LIMIT_PCT      = 0.03    # 3 % de perte/semaine → pause jusqu'au lundi
LIVE_MAX_DEPLOYED_CAPITAL_PCT   = 0.25    # Max 25 % du capital total en marché
LIVE_MIN_PAPER_TRADES           = 50      # Trades paper min avant d'autoriser le live

# ─────────────────────────────────────────────
# MACHINE LEARNING SUPERVISÉ
# ─────────────────────────────────────────────
ML_RETRAIN_EVERY_N_TRADES = 20       # Ré-entraîner toutes les N trades
ML_MIN_TRADES_FOR_TRAIN   = 30       # Minimum de trades pour un premier entraînement
ML_CONFIDENCE_THRESHOLD   = 0.55     # Score min pour valider un signal
MODEL_DIR                 = "models"

# ─────────────────────────────────────────────
# REINFORCEMENT LEARNING (DQN)
# ─────────────────────────────────────────────
RL_REPLAY_BUFFER_SIZE = 10_000
RL_BATCH_SIZE         = 32
RL_GAMMA              = 0.99
RL_LR                 = 1e-3
RL_EPSILON_START      = 1.0
RL_EPSILON_END        = 0.05
RL_EPSILON_DECAY      = 0.995
RL_TARGET_UPDATE_FREQ = 50           # Steps entre chaque mise à jour du target network
RL_TRAIN_EVERY_N      = 10           # Entraîner le DQN toutes les N steps

# ─────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────
LOOP_INTERVAL_SECONDS = 300          # Fréquence d'analyse (5 min)
FAST_LEARN_INTERVAL_S = 60           # Intervalle rapide pour ré-analyse RL (1 min)

# ─────────────────────────────────────────────
# LOGS & PERSISTANCE
# ─────────────────────────────────────────────
LOG_DIR           = "logs"
TRADES_CSV        = os.path.join(LOG_DIR, "trades_history.csv")
STATE_FILE        = "state.json"
LOG_MAX_BYTES     = 10 * 1024 * 1024   # 10 MB par fichier de log
LOG_BACKUP_COUNT  = 5
