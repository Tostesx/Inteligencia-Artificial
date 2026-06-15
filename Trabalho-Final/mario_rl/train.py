"""
Treinamento do DQN Agent no ALE/MarioBros-v5
============================================

Uso:
  python train.py                    # treina com configurações padrão
  python train.py --episodes 2000    # mais episódios
  python train.py --resume           # continua de um checkpoint

Execute em terminal:
  python train.py --help
"""

import argparse
import os
import time
import json
from pathlib import Path

import numpy as np

from wrappers import make_env
from agent import DQNAgent


# ──────────────────────────────────────────────────────────────────────
#  Argumentos
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Treinamento DQN – Mario Bros ALE")
    p.add_argument("--episodes",      type=int,   default=3000,   help="Número total de episódios")
    p.add_argument("--lr",            type=float, default=1e-4,   help="Taxa de aprendizado")
    p.add_argument("--gamma",         type=float, default=0.99,   help="Fator de desconto")
    p.add_argument("--eps-start",     type=float, default=1.0,    help="Epsilon inicial")
    p.add_argument("--eps-end",       type=float, default=0.05,   help="Epsilon mínimo")
    p.add_argument("--eps-decay",     type=int,   default=150_000,help="Passos para decaimento do epsilon")
    p.add_argument("--batch-size",    type=int,   default=32,     help="Tamanho do mini-batch")
    p.add_argument("--buffer-size",   type=int,   default=100_000,help="Capacidade do replay buffer")
    p.add_argument("--target-update", type=int,   default=1_000,  help="Frequência de atualização da target net")
    p.add_argument("--warmup",        type=int,   default=10_000, help="Steps antes de começar a aprender")
    p.add_argument("--save-dir",      type=str,   default="checkpoints", help="Pasta para checkpoints")
    p.add_argument("--save-every",    type=int,   default=200,    help="Salvar a cada N episódios")
    p.add_argument("--seed",          type=int,   default=42,     help="Semente aleatória")
    p.add_argument("--resume",        type=str,   default=None,   help="Caminho do checkpoint para retomar")
    p.add_argument("--log-interval",  type=int,   default=20,     help="Exibir stats a cada N episódios")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
#  Utilitários
# ──────────────────────────────────────────────────────────────────────

def moving_average(values, window: int = 50):
    """Média móvel simples."""
    if len(values) < window:
        return np.mean(values)
    return np.mean(values[-window:])


def save_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


# ──────────────────────────────────────────────────────────────────────
#  Loop principal de treinamento
# ──────────────────────────────────────────────────────────────────────

def train(args):
    np.random.seed(args.seed)

    # Diretório de saída
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Ambiente
    env = make_env(seed=args.seed, clip_reward=True)
    n_actions = env.action_space.n
    print(f"[Env] Action space: {n_actions}  |  Obs shape: {env.observation_space.shape}")

    # Agente
    agent = DQNAgent(
        n_actions       = n_actions,
        lr              = args.lr,
        gamma           = args.gamma,
        epsilon_start   = args.eps_start,
        epsilon_end     = args.eps_end,
        epsilon_decay   = args.eps_decay,
        batch_size      = args.batch_size,
        target_update_freq = args.target_update,
        buffer_capacity = args.buffer_size,
    )

    # Retomar treinamento
    start_episode = 0
    metrics = {"episode_rewards": [], "episode_lengths": [], "losses": [], "epsilons": []}

    if args.resume:
        agent.load(args.resume)
        metrics_path = save_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            start_episode = len(metrics["episode_rewards"])
        print(f"[Train] Retomando do episódio {start_episode}")

    print("\n" + "=" * 60)
    print("  TREINAMENTO DQN – ALE/MarioBros-v5")
    print("=" * 60)
    print(f"  Episódios:     {args.episodes}")
    print(f"  Warmup steps:  {args.warmup}")
    print(f"  Buffer:        {args.buffer_size}")
    print(f"  Batch size:    {args.batch_size}")
    print("=" * 60 + "\n")

    global_steps  = agent.steps
    best_reward   = -np.inf
    total_loss    = []
    t_start       = time.time()

    for episode in range(start_episode, start_episode + args.episodes):
        obs, _ = env.reset()
        ep_reward = 0
        ep_steps  = 0
        ep_losses = []

        done = False
        while not done:
            # Fase de warmup: ações aleatórias
            if global_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, float(done))
            obs = next_obs

            # Aprendizado
            if global_steps >= args.warmup:
                loss = agent.learn()
                if loss is not None:
                    ep_losses.append(loss)

            ep_reward   += reward
            ep_steps    += 1
            global_steps += 1

        # Registra métricas
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        metrics["episode_rewards"].append(ep_reward)
        metrics["episode_lengths"].append(ep_steps)
        metrics["losses"].append(avg_loss)
        metrics["epsilons"].append(agent.epsilon)
        total_loss.append(avg_loss)

        # Salva melhor modelo
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(str(save_dir / "best_model.pt"))

        # Log periódico
        if (episode + 1) % args.log_interval == 0:
            avg_r  = moving_average(metrics["episode_rewards"])
            avg_l  = moving_average(metrics["losses"])
            elapsed = time.time() - t_start
            fps    = global_steps / elapsed

            print(
                f"Ep {episode + 1:5d}/{start_episode + args.episodes} | "
                f"Reward: {ep_reward:6.1f} | "
                f"Avg(50): {avg_r:6.1f} | "
                f"Loss: {avg_l:.4f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Steps: {global_steps:,} | "
                f"FPS: {fps:.0f}"
            )

        # Checkpoint periódico
        if (episode + 1) % args.save_every == 0:
            ckpt_path = save_dir / f"checkpoint_ep{episode + 1}.pt"
            agent.save(str(ckpt_path))
            save_metrics(metrics, str(save_dir / "metrics.json"))

    # Salva métricas finais
    save_metrics(metrics, str(save_dir / "metrics.json"))
    agent.save(str(save_dir / "final_model.pt"))

    elapsed = time.time() - t_start
    print(f"\n✅ Treinamento concluído em {elapsed / 60:.1f} minutos.")
    print(f"   Melhor reward: {best_reward:.1f}")
    print(f"   Modelos salvos em: {save_dir}/")

    env.close()
    return metrics


# ──────────────────────────────────────────────────────────────────────
#  Entrypoint
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
