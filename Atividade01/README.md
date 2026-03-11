# Previsão de Ações com LSTM

## Objetivo
Projeto de previsão de séries temporais financeiras utilizando redes neurais LSTM para 5 ações diferentes.

## Ações Analisadas
- TSLA (Tesla - EUA)
- AAPL (Apple - EUA)  
- PETR4.SA (Petrobras - Brasil)
- WEGE3.SA (Weg - Brasil)
- ITUB4.SA (Itaú Unibanco - Brasil)

## Período
2015-01-01 até data atual

## Metodologia
- Janela temporal: 30 dias (lookback)
- Normalização: MinMaxScaler (0-1)
- Modelo: LSTM com 50 neurônios + Dense(1)
- Treino: 80% dos dados (ordem temporal)
- Early stopping para evitar overfitting

## Resultados
| Ação | RMSE | Preço Atual | Previsão | Variação |
|------|------|-------------|----------|----------|
| TSLA | XX.XX | XXX.XX | XXX.XX | +X.XX% |
| AAPL | XX.XX | XXX.XX | XXX.XX | +X.XX% |
| ... | ... | ... | ... | ... |

## Como executar
1. Clone o repositório
2. Instale dependências: `pip install -r requirements.txt`
3. Execute o notebook