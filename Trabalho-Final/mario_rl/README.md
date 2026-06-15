<div align="center">

# 🍄 Mario Bros — Deep Reinforcement Learning Agent

**Agente DQN treinado para jogar Mario Bros (Atari) via Arcade Learning Environment**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-ALE-green)](https://ale.farama.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

*Trabalho Final — Inteligência Artificial*

</div>

---

## 📖 Sobre o projeto

Este projeto implementa um agente de **aprendizado por reforço profundo** capaz de jogar o jogo **Mario Bros do Atari** de forma autônoma, aprendendo apenas a partir de pixels da tela e da pontuação do jogo — sem nenhuma regra programada manualmente.

O agente utiliza a arquitetura **Deep Q-Network (DQN)** introduzida pela DeepMind em 2015, com a melhoria **Double DQN** para estabilizar o aprendizado.

### Como funciona?

O agente observa a tela do jogo (4 frames em escala de cinza empilhados), passa por uma rede neural convolucional (CNN) e aprende quais ações maximizam a pontuação ao longo do tempo. O aprendizado ocorre por tentativa e erro através de milhares de episódios.

---

## 🎬 Demonstração

> O agente após treinamento jogando automaticamente:

```
python evaluate.py --model checkpoints/best_model.pt
```

*(abre janela gráfica com o agente em ação)*

---

## 🧠 Arquitetura

### Pipeline de pré-processamento

```
Frame RGB (210×160×3)
      ↓  Escala de cinza + resize
Grayscale (84×84×1)
      ↓  Normalização [0, 1]
Float (84×84×1)
      ↓  Empilhamento de 4 frames
Tensor (4×84×84)  ──▶  CNN
```

### Rede Neural Convolucional

```
Input: (batch, 4, 84, 84)
  Conv2d(4→32,  kernel=8, stride=4) + ReLU  →  (32, 20, 20)
  Conv2d(32→64, kernel=4, stride=2) + ReLU  →  (64,  9,  9)
  Conv2d(64→64, kernel=3, stride=1) + ReLU  →  (64,  7,  7)
  Flatten  →  Linear(3136→512) + ReLU
  Linear(512→18)  →  Q-value por ação
```

### Técnicas implementadas

| Componente | Detalhes |
|---|---|
| Algoritmo base | DQN — Mnih et al. (2015) |
| Melhoria | Double DQN — van Hasselt et al. (2016) |
| Exploração | ε-greedy com decay exponencial |
| Otimizador | Adam (lr = 1e-4) |
| Função de perda | Huber Loss |
| Replay Buffer | Uniforme — 100k transições |
| Frame skip | 4 (ação repetida por 4 frames) |
| Reward clipping | sign(r) → {−1, 0, +1} |

---

## 📁 Estrutura do repositório

```
mario_rl/
├── agent.py           # CNN + agente Double DQN
├── wrappers.py        # Wrappers de pré-processamento do ambiente
├── train.py           # Loop de treinamento com checkpoints e logs
├── evaluate.py        # Avaliação do modelo e gráficos de aprendizado
├── requirements.txt   # Dependências Python
├── checkpoints/
│   ├── best_model.pt          # Melhor modelo salvo durante o treino
│   ├── final_model.pt         # Modelo ao final do treinamento
│   └── metrics.json           # Histórico de rewards, losses e epsilon
└── README.md
```

---

## ⚙️ Instalação

### Pré-requisitos

- Python 3.10 ou superior
- pip
- (Opcional) GPU NVIDIA com CUDA — veja a seção GPU abaixo

### 1. Clone o repositório

```bash
git clone https://github.com/Tostesx/Inteligencia-Artificial/tree/main/Trabalho-Final/mario_rl.git

cd mario_rl
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate
            ou
source venv/bin/activate.fish

# Windows
venv\Scripts\activate
```

> O prompt deve mostrar `(venv)` confirmando que está ativo.

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Instale o ROM do Mario Bros

```bash
AutoROM --accept-license
```

> Isso baixa e instala automaticamente os ROMs do Atari liberados para uso livre.

---

## 🎮 Usando o modelo pré-treinado

Se você quer apenas **assistir o agente jogar** sem treinar do zero:

```bash
# Assiste 5 episódios com janela gráfica
python evaluate.py --model checkpoints/best_model.pt

# Assiste mais episódios
python evaluate.py --model checkpoints/best_model.pt --episodes 10

# Avalia 20 episódios sem abrir janela (só mostra scores)
python evaluate.py --model checkpoints/best_model.pt --no-render --episodes 20

# Plota as curvas de aprendizado do treino
python evaluate.py --plot checkpoints/metrics.json
```

> ⚠️ **Linux sem interface gráfica:** instale o SDL2 para renderização:
> - Arch Linux: `sudo pacman -S sdl2`
> - Ubuntu/Debian: `sudo apt install libsdl2-dev`

---

## 🏋️ Treinando do zero

Se quiser treinar seu próprio modelo:

```bash
# Treinamento padrão (3000 episódios)
python train.py

# Mais episódios para melhor desempenho
python train.py --episodes 10000

# Continuar de um checkpoint salvo
python train.py --resume checkpoints/best_model.pt --episodes 5000

# Ver todas as opções disponíveis
python train.py --help
```

### Parâmetros de treinamento

| Argumento | Padrão | Descrição |
|---|---|---|
| `--episodes` | 3000 | Total de episódios |
| `--lr` | 1e-4 | Taxa de aprendizado |
| `--gamma` | 0.99 | Fator de desconto γ |
| `--eps-decay` | 150000 | Steps para decay do epsilon |
| `--buffer-size` | 100000 | Capacidade do replay buffer |
| `--warmup` | 10000 | Steps aleatórios antes de aprender |
| `--target-update` | 1000 | Frequência de atualização da target net |
| `--batch-size` | 32 | Tamanho do mini-batch |
| `--save-every` | 200 | Salvar checkpoint a cada N episódios |
| `--seed` | 42 | Semente aleatória |

---

## 🖥️ Suporte a GPU

O código detecta automaticamente a GPU disponível. Para usar CUDA, instale o PyTorch compatível com sua versão de driver em [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

> **Atenção:** GPUs antigas como a GTX 1050 Ti (Compute Capability 6.1) não são suportadas pelo PyTorch 2.x. Nesse caso, o treinamento roda em **CPU**, o que é perfeitamente funcional — apenas mais lento.

Para forçar CPU explicitamente, edite `agent.py`:
```python
self.device = torch.device("cpu")
```

---

## 📊 Espaço de ações

O ambiente possui 18 ações possíveis:

| ID | Ação | ID | Ação |
|---|---|---|---|
| 0 | NOOP | 9 | DOWNLEFT |
| 1 | FIRE | 10 | UPFIRE |
| 2 | UP | 11 | RIGHTFIRE |
| 3 | RIGHT | 12 | LEFTFIRE |
| 4 | LEFT | 13 | DOWNFIRE |
| 5 | DOWN | 14 | UPRIGHTFIRE |
| 6 | UPRIGHT | 15 | UPLEFTFIRE |
| 7 | UPLEFT | 16 | DOWNRIGHTFIRE |
| 8 | DOWNRIGHT | 17 | DOWNLEFTFIRE |

---

## 💡 Dicas para melhorar o desempenho

1. **Treine por mais tempo** — DQN no Atari tipicamente precisa de 10–50 milhões de steps para atingir desempenho alto
2. **Use GPU** — Acelera o treino em 5–20× em relação à CPU
3. **Prioritized Experience Replay (PER)** — Amostras mais importantes são revisitadas com mais frequência
4. **Dueling DQN** — Separa a estimativa do valor de estado das vantagens por ação
5. **Reward shaping** — Adicione recompensas extras por matar inimigos ou avançar de fase

---

## 📚 Referências

- Mnih et al. (2015) — *Human-level control through deep reinforcement learning* — [Nature](https://www.nature.com/articles/nature14236)
- van Hasselt et al. (2016) — *Deep Reinforcement Learning with Double Q-learning* — [arXiv](https://arxiv.org/abs/1509.06461)
- [Arcade Learning Environment](https://ale.farama.org/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

<div align="center">
Feito como Trabalho Final de Inteligência Artificial
</div>
