"""
Agent Deep Q-Network (DQN) pour l'optimisation des décisions de trading.
Apprend par renforcement : récompenses sur les PnL, pénalités sur les drawdowns.
Actions : 0=Hold, 1=Buy/Long, 2=Sell/Short
"""

import os
import random
import logging
import collections
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config

logger = logging.getLogger("trading_bot")

os.makedirs(config.MODEL_DIR, exist_ok=True)
RL_MODEL_PATH = os.path.join(config.MODEL_DIR, "rl_dqn.pt")

# Dimensions
STATE_DIM  = 10   # Nombre de features d'état
ACTION_DIM = 3    # Hold=0, Long=1, Short=2


class DQNNetwork(nn.Module):
    """Réseau de neurones pour l'approximation de la Q-value."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Buffer circulaire d'expériences (s, a, r, s', done)."""

    def __init__(self, capacity: int = config.RL_REPLAY_BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Agent DQN avec epsilon-greedy, replay buffer et target network.
    S'améliore à chaque trade terminé en apprenant de ses erreurs.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork().to(self.device)
        self.target_net  = DQNNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.RL_LR)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss

        self.replay_buffer = ReplayBuffer()
        self.epsilon       = config.RL_EPSILON_START
        self.steps         = 0

        self._load_if_exists()

    # ──────────────────────────────────────────
    # Interface publique
    # ──────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """
        Sélectionne une action via epsilon-greedy.
        0=Hold, 1=Long, 2=Short
        """
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)

        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(t)
            return int(q_values.argmax().item())

    def store_experience(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        """Enregistre une transition dans le replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.steps += 1

        # Décroissance epsilon
        self.epsilon = max(
            config.RL_EPSILON_END,
            self.epsilon * config.RL_EPSILON_DECAY,
        )

    def learn(self) -> Optional[float]:
        """
        Effectue une étape de gradient descent.
        Retourne la loss ou None si buffer insuffisant.
        """
        if len(self.replay_buffer) < config.RL_BATCH_SIZE:
            return None

        batch = self.replay_buffer.sample(config.RL_BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Q-values courants
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)

        # Q-values cibles (Double DQN)
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q = rewards + config.RL_GAMMA * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour la stabilité
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Mise à jour du target network
        if self.steps % config.RL_TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug(f"Target network mis à jour (step {self.steps})")

        return float(loss.item())

    def is_trained_enough(self) -> bool:
        """Retourne True si l'agent a suffisamment appris pour filtrer les signaux."""
        return self.steps >= 200

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Retourne les Q-values pour un état donné (debugging/monitoring)."""
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(t).cpu().numpy()[0]

    def encode_state(
        self,
        rsi: float,
        macd: float,
        bb_pct: float,
        adx: float,
        volume_ratio: float,
        price_ret_1h: float,   # Rendement sur 1 heure
        price_ret_4h: float,   # Rendement sur 4 heures
        in_position: int,      # 0=no position, 1=long, -1=short
        pnl_pct: float,        # PnL non réalisé en %
        capital_ratio: float,  # Capital actuel / capital initial
    ) -> np.ndarray:
        """Encode les features en vecteur d'état normalisé."""
        state = np.array([
            rsi / 100.0,                  # 0-1
            np.tanh(macd),                # bornage
            bb_pct,                       # déjà 0-1
            adx / 100.0,                  # 0-1
            min(volume_ratio / 3.0, 1.0), # bornage
            np.tanh(price_ret_1h * 100),  # bornage
            np.tanh(price_ret_4h * 100),  # bornage
            float(in_position),           # -1, 0, 1
            np.tanh(pnl_pct * 10),        # bornage
            min(capital_ratio, 2.0) / 2.0,# bornage
        ], dtype=np.float32)
        return state

    def compute_reward(self, pnl: float, drawdown: float, exit_reason: str) -> float:
        """
        Calcule la récompense d'une action terminée.
        Récompense positive = profit, pénalité = perte ou drawdown.
        """
        # Récompense de base proportionnelle au PnL
        reward = np.tanh(pnl / 100.0) * 2.0

        # Pénalité pour drawdown important
        if drawdown > 0.05:
            reward -= drawdown * 2.0

        # Bonus si TP atteint (trade de qualité)
        if exit_reason == "TP":
            reward += 0.2

        # Pénalité supplémentaire si SL touché
        if exit_reason == "SL":
            reward -= 0.1

        return float(np.clip(reward, -3.0, 3.0))

    # ──────────────────────────────────────────
    # Persistance
    # ──────────────────────────────────────────

    def save(self):
        try:
            torch.save({
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "epsilon":    self.epsilon,
                "steps":      self.steps,
            }, RL_MODEL_PATH)
            logger.debug(f"DQN sauvegardé : {RL_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Erreur sauvegarde DQN : {e}")

    def _load_if_exists(self):
        if os.path.exists(RL_MODEL_PATH):
            try:
                checkpoint = torch.load(RL_MODEL_PATH, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint["policy_net"])
                self.target_net.load_state_dict(checkpoint["target_net"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.epsilon = checkpoint.get("epsilon", config.RL_EPSILON_END)
                self.steps   = checkpoint.get("steps", 0)
                logger.info(
                    f"DQN chargé depuis {RL_MODEL_PATH} "
                    f"(step={self.steps}, epsilon={self.epsilon:.3f})"
                )
            except Exception as e:
                logger.warning(f"Impossible de charger le DQN : {e}")
