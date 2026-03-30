"""
Composant Supervised Learning : RandomForest + XGBoost en ensemble.
Apprend à prédire si un signal technique sera profitable.
Se ré-entraîne automatiquement toutes les N trades (configurable).
"""

import os
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import config

logger = logging.getLogger("trading_bot")

os.makedirs(config.MODEL_DIR, exist_ok=True)
MODEL_PATH  = os.path.join(config.MODEL_DIR, "supervised_model.pkl")
SCALER_PATH = os.path.join(config.MODEL_DIR, "scaler.pkl")

FEATURE_COLS = [
    "rsi_at_entry", "macd_at_entry", "bb_pct_at_entry",
    "adx_at_entry", "volume_ratio_at_entry",
]


class SupervisedModel:
    """
    Modèle ensemble (RandomForest + XGBoost) entraîné sur l'historique des trades.
    Prédit la probabilité qu'un signal donné soit profitable.
    """

    def __init__(self):
        self.model: Optional[Pipeline] = None
        self._trained = False
        self._load_if_exists()

    # ──────────────────────────────────────────
    # Interface publique
    # ──────────────────────────────────────────

    def predict_confidence(self, features: Dict) -> float:
        """
        Retourne la probabilité (0.0→1.0) que le signal soit profitable.
        Si le modèle n'est pas encore entraîné, retourne 1.0 pour laisser
        passer tous les signaux techniques (le bot doit d'abord accumuler
        des trades avant que le ML puisse filtrer).
        """
        if not self._trained or self.model is None:
            return 1.0   # Pas encore entraîné → laisser passer tous les signaux

        try:
            X = self._dict_to_array(features)
            proba = self.model.predict_proba(X)[0][1]
            return float(proba)
        except Exception as e:
            logger.warning(f"Erreur de prédiction ML supervisé : {e}")
            return 0.5

    def train(self, trades: List[Dict]) -> Optional[Dict]:
        """
        Entraîne le modèle sur l'historique des trades.
        Retourne les métriques d'évaluation ou None si trop peu de données.
        """
        if len(trades) < config.ML_MIN_TRADES_FOR_TRAIN:
            logger.info(
                f"Pas assez de trades pour entraîner ({len(trades)} < "
                f"{config.ML_MIN_TRADES_FOR_TRAIN})"
            )
            return None

        X, y = self._build_dataset(trades)
        if X is None or len(X) < config.ML_MIN_TRADES_FOR_TRAIN:
            return None

        logger.info(f"Entraînement supervisé sur {len(X)} trades...")

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb)],
            voting="soft",
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ensemble),
        ])

        # Validation croisée temporelle
        tscv = TimeSeriesSplit(n_splits=3)
        try:
            scores = cross_val_score(pipe, X, y, cv=tscv, scoring="accuracy")
            cv_acc = float(scores.mean())
        except Exception:
            cv_acc = 0.0

        pipe.fit(X, y)
        self.model = pipe
        self._trained = True
        self._save()

        metrics = {
            "samples":     len(X),
            "cv_accuracy": round(cv_acc, 4),
            "pos_ratio":   round(float(y.mean()), 4),
        }
        logger.info(f"Modèle supervisé entraîné : {metrics}")
        return metrics

    def is_trained(self) -> bool:
        return self._trained

    # ──────────────────────────────────────────
    # Helpers privés
    # ──────────────────────────────────────────

    def _build_dataset(self, trades: List[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        rows = []
        labels = []
        for t in trades:
            try:
                row = [
                    float(t.get("rsi_at_entry") or 50),
                    float(t.get("macd_at_entry") or 0),
                    float(t.get("bb_pct_at_entry") or 0.5),
                    float(t.get("adx_at_entry") or 20),
                    float(t.get("volume_ratio_at_entry") or 1),
                ]
                pnl = float(t.get("pnl") or 0)
                label = 1 if pnl > 0 else 0
                rows.append(row)
                labels.append(label)
            except (ValueError, TypeError):
                continue

        if not rows:
            return None, None

        X = np.array(rows, dtype=float)
        y = np.array(labels, dtype=int)
        return X, y

    def _dict_to_array(self, features: Dict) -> np.ndarray:
        return np.array([[
            float(features.get("rsi_at_entry", 50)),
            float(features.get("macd_at_entry", 0)),
            float(features.get("bb_pct_at_entry", 0.5)),
            float(features.get("adx_at_entry", 20)),
            float(features.get("volume_ratio_at_entry", 1)),
        ]])

    def _save(self):
        try:
            joblib.dump(self.model, MODEL_PATH)
            logger.debug(f"Modèle supervisé sauvegardé : {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèle supervisé : {e}")

    def _load_if_exists(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model    = joblib.load(MODEL_PATH)
                self._trained = True
                logger.info(f"Modèle supervisé chargé depuis {MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Impossible de charger le modèle supervisé : {e}")
