"""
Wrappers de pré-processamento para o ambiente ALE/MarioBros-v5.

Pipeline:
  1. NoopResetEnv       – aleatoriedade no início do episódio
  2. FireResetEnv       – pressiona FIRE para iniciar (se necessário)
  3. MaxAndSkipEnv      – repete ação N frames e pega máximo de pixels
  4. WarpFrame          – redimensiona para 84×84 em escala de cinza
  5. ScaledFloatFrame   – normaliza pixels para [0, 1]
  6. FrameStack         – empilha 4 frames consecutivos
  7. ClipRewardEnv      – clipa reward para {-1, 0, +1}
"""
import ale_py
import numpy as np
import gymnasium as gym
gym.register_envs(ale_py)
from gymnasium import spaces
import cv2


# ─────────────────────────────────────────────
#  1. NoopResetEnv
# ─────────────────────────────────────────────

class NoopResetEnv(gym.Wrapper):
    """Executa entre 1 e `noop_max` NOOPs aleatórios no reset."""

    def __init__(self, env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0  # NOOP = 0 em todos os envs Atari

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


# ─────────────────────────────────────────────
#  2. MaxAndSkipEnv
# ─────────────────────────────────────────────

class MaxAndSkipEnv(gym.Wrapper):
    """Repete a ação por `skip` frames; retorna o máximo pixel-a-pixel
    dos 2 últimos frames (remove flickering do Atari)."""

    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        shape = env.observation_space.shape
        self._obs_buffer = np.zeros((2, *shape), dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


# ─────────────────────────────────────────────
#  3. WarpFrame
# ─────────────────────────────────────────────

class WarpFrame(gym.ObservationWrapper):
    """Converte para escala de cinza e redimensiona para 84×84."""

    def __init__(self, env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(height, width, 1),
            dtype=np.uint8,
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, np.newaxis]


# ─────────────────────────────────────────────
#  4. ScaledFloatFrame
# ─────────────────────────────────────────────

class ScaledFloatFrame(gym.ObservationWrapper):
    """Normaliza pixels para [0.0, 1.0]."""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.array(obs, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────
#  5. FrameStack
# ─────────────────────────────────────────────

class FrameStack(gym.Wrapper):
    """Empilha `k` frames consecutivos no eixo do canal.
    Retorna array com shape (k, H, W) pronto para a CNN."""

    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = k
        self._frames = None
        h, w, _ = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(k, h, w), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = obs[:, :, 0]
        self._frames = np.stack([frame] * self.k, axis=0)
        return self._frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = obs[:, :, 0]
        self._frames = np.roll(self._frames, shift=-1, axis=0)
        self._frames[-1] = frame
        return self._frames.copy(), reward, terminated, truncated, info


# ─────────────────────────────────────────────
#  6. ClipRewardEnv
# ─────────────────────────────────────────────

class ClipRewardEnv(gym.RewardWrapper):
    """Clipa o reward para {-1, 0, +1} usando np.sign."""

    def reward(self, reward: float) -> float:
        return np.sign(reward)


# ─────────────────────────────────────────────
#  Função de fábrica
# ─────────────────────────────────────────────

def make_env(
    env_id: str = "ALE/MarioBros-v5",
    render_mode: str | None = None,
    seed: int = 42,
    clip_reward: bool = True,
) -> gym.Env:
    """
    Cria e retorna o ambiente com todos os wrappers aplicados.

    Parâmetros
    ----------
    env_id : str
        ID do ambiente Gymnasium (default: ALE/MarioBros-v5).
    render_mode : str | None
        'human' para visualização em tempo real, None para treinamento.
    seed : int
        Semente aleatória.
    clip_reward : bool
        Se True, aplica ClipRewardEnv.

    Retorna
    -------
    gym.Env com observation_space de shape (4, 84, 84).
    """
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = FrameStack(env, k=4)
    if clip_reward:
        env = ClipRewardEnv(env)
    return env
