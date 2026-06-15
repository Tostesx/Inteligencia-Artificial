"""
DQN Agent para Mario Bros (ALE/MarioBros-v5)
Baseado no paper: "Human-level control through deep reinforcement learning" (DeepMind, 2015)
"""

import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Rede Neural Convolucional (CNN)
# ─────────────────────────────────────────────

class DQN(nn.Module):
    """
    Arquitetura CNN idêntica ao paper original da DeepMind.
    Entrada: 4 frames em escala de cinza empilhados → (4, 84, 84)
    Saída:   Q-values para cada ação
    """

    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # → (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (64, 7, 7)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 4, 84, 84), valores em [0, 1]
        return self.fc(self.conv(x))


# ─────────────────────────────────────────────
#  Replay Buffer
# ─────────────────────────────────────────────

class ReplayBuffer:
    """Buffer de experiências com amostragem aleatória uniforme."""

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
#  Agente DQN
# ─────────────────────────────────────────────

class DQNAgent:
    """
    Agente que implementa:
    - ε-greedy exploration com decay
    - Double DQN (target network)
    - Experience Replay
    """

    def __init__(
        self,
        n_actions: int,
        device: str = "auto",
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 100_000,
        batch_size: int = 32,
        target_update_freq: int = 1_000,
        buffer_capacity: int = 100_000,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0

        # Dispositivo
        if device == "auto":
            self.device = torch.device("cpu")  # GTX 1050 Ti não suportada pelo PyTorch atual
        else:
            self.device = torch.device(device)

        print(f"[Agent] Usando dispositivo: {self.device}")

        # Redes
        self.policy_net = DQN(n_actions).to(self.device)
        self.target_net = DQN(n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    # ── Epsilon com decay exponencial ──────────────────────────────────
    def _update_epsilon(self):
        self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * np.exp(
            -self.steps / self.epsilon_decay
        )

    # ── Seleção de ação ε-greedy ────────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q = self.policy_net(s)
            return q.argmax(dim=1).item()

    # ── Armazenar experiência ───────────────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ── Passo de aprendizado ────────────────────────────────────────────
    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        self.steps += 1
        self._update_epsilon()

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Converte para tensores
        states_t      = torch.tensor(states,      device=self.device)
        actions_t     = torch.tensor(actions,     device=self.device)
        rewards_t     = torch.tensor(rewards,     device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t       = torch.tensor(dones,       device=self.device)

        # Q(s, a) com a rede de política
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Q-target (Double DQN): ação escolhida pela policy_net, valor pela target_net
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = F.huber_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Atualiza a target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ── Salvar / Carregar ───────────────────────────────────────────────
    def save(self, path: str):
        torch.save(
            {
                "policy_state": self.policy_net.state_dict(),
                "target_state": self.target_net.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "steps": self.steps,
                "epsilon": self.epsilon,
            },
            path,
        )
        print(f"[Agent] Modelo salvo em: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.steps   = ckpt["steps"]
        self.epsilon = ckpt["epsilon"]
        print(f"[Agent] Modelo carregado de: {path}  (steps={self.steps}, ε={self.epsilon:.4f})")
