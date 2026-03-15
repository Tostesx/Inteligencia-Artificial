# Previsão de Evasão de Alunos com IA

Este projeto implementa um modelo de Machine Learning para prever evasão de alunos, com uma interface web simples em Flask.

## Objetivo
Desenvolver uma aplicação web que utiliza um modelo treinado para classificar se um aluno irá evadir ou concluir o curso.

## Dataset
Utilizado dataset fornecido pelo professor, contendo informações de alunos e a coluna alvo `status_curso` (0 = concluiu, 1 = evadiu).

## Modelo
Escolhemos Random Forest Classifier com 200 árvores, devido à sua robustez e capacidade de lidar com dados categóricos e numéricos. O modelo foi treinado após codificação das variáveis categóricas com LabelEncoder.

## Métricas de Avaliação
O modelo foi avaliado com as métricas: acurácia, precisão, recall e F1-score. O relatório de classificação é exibido durante o treinamento.

