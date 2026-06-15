"""
Avaliação e visualização do agente treinado
==========================================

Uso:
  # Assiste o agente jogar (5 episódios):
  python evaluate.py --model checkpoints/best_model.pt

  # Avalia sem renderizar (mais rápido):
  python evaluate.py --model checkpoints/best_model.pt --no-render --episodes 20

  # Plota curvas de aprendizado (requer matplotlib):
  python evaluate.py --plot checkpoints/metrics.json
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

from wrappers import make_env
from agent import DQNAgent


# ──────────────────────────────────────────────────────────────────────
#  Argumentos
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Avaliação do agente DQN – Mario Bros")
    p.add_argument("--model",     type=str, default="checkpoints/best_model.pt",
                   help="Caminho para o arquivo .pt do modelo")
    p.add_argument("--episodes",  type=int, default=5,
                   help="Número de episódios de avaliação")
    p.add_argument("--no-render", action="store_true",
                   help="Desativa visualização (mais rápido)")
    p.add_argument("--plot",      type=str, default=None,
                   help="Plota métricas a partir de metrics.json")
    p.add_argument("--seed",      type=int, default=0)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
#  Avaliação
# ──────────────────────────────────────────────────────────────────────

def evaluate(model_path: str, n_episodes: int = 5, render: bool = True, seed: int = 0):
    render_mode = "human" if render else None

    env = make_env(render_mode=render_mode, seed=seed, clip_reward=False)
    n_actions = env.action_space.n

    agent = DQNAgent(n_actions=n_actions)
    agent.load(model_path)
    agent.epsilon = 0.01  # quase determinístico na avaliação

    rewards = []

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        print(f"Episódio {ep:3d} | Reward: {ep_reward:7.1f}")

    print(f"\n── Resultado ({n_episodes} episódios) ──────────────────")
    print(f"  Média:  {np.mean(rewards):.2f}")
    print(f"  Std:    {np.std(rewards):.2f}")
    print(f"  Máximo: {np.max(rewards):.2f}")
    print(f"  Mínimo: {np.min(rewards):.2f}")

    env.close()
    return rewards


# ──────────────────────────────────────────────────────────────────────
#  Plot de curvas de aprendizado
# ──────────────────────────────────────────────────────────────────────

def plot_metrics(metrics_path: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("❌ matplotlib não instalado. Execute: pip install matplotlib")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    rewards  = metrics["episode_rewards"]
    losses   = metrics["losses"]
    epsilons = metrics["epsilons"]
    episodes = range(1, len(rewards) + 1)

    # Média móvel
    def smooth(x, w=50):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("DQN – ALE/MarioBros-v5  |  Curvas de Aprendizado", fontsize=14, fontweight="bold")

    # Reward
    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.3, color="#4C9BE8", label="Reward por episódio")
    if len(rewards) >= 50:
        ax.plot(range(50, len(rewards) + 1), smooth(rewards), color="#1A5FA8", label="Média móvel (50 ep)")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Loss
    ax = axes[1]
    ax.plot(episodes, losses, alpha=0.3, color="#E8844C", label="Loss por episódio")
    if len(losses) >= 50:
        ax.plot(range(50, len(losses) + 1), smooth(losses), color="#A84B1A", label="Média móvel (50 ep)")
    ax.set_ylabel("Huber Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Epsilon
    ax = axes[2]
    ax.plot(episodes, epsilons, color="#5CB85C", label="Epsilon (ε)")
    ax.set_ylabel("Epsilon")
    ax.set_xlabel("Episódio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    out_path = Path(metrics_path).parent / "learning_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✅ Gráfico salvo em: {out_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────
#  Entrypoint
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.plot:
        plot_metrics(args.plot)
    else:
        if not Path(args.model).exists():
            print(f"❌ Modelo não encontrado: {args.model}")
            print("   Execute o treinamento primeiro: python train.py")
            sys.exit(1)
        evaluate(
            model_path=args.model,
            n_episodes=args.episodes,
            render=not args.no_render,
            seed=args.seed,
        )
