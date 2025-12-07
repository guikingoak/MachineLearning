# Relatório – KNN

## Introdução
Este projeto tem como objetivo aplicar o algoritmo K-Nearest Neighbors (KNN) para resolver um problema de classificação com o dataset Titanic. O Projeto consiste em prever se um passageiro sobreviveu ao naufrágio com base em atributos demográficos e socioeconômicos.

## 1. Exploração dos Dados
O dataset Titanic contém features que utilizaremos como o Pclass, Sex, Age, SibSp, Parch, Fare e Embarked. A variável alvo é Survived sendo 0 = não sobreviveu e 1 = sobreviveu. A análise exploratória identificou padrões importantes, como maior sobrevivência entre mulheres e passageiros da 1ª classe. Estatísticas descritivas e visualizações realizadas ao subir carregar o dataset ajudaram a compreender melhor a distribuição dos atributos e a relação deles com a variável alvo.

## 2. Pré-processamento
O pré-processamento envolveu:
- Tratamento de valores ausentes onde para a variável Age utilizamos a mediana e para a variável Embarked utilizamos a moda.
- Conversão de Sex para valores numéricos.
- One-Hot Encoding aplicado em Embarked.
- Normalização com StandardScaler devido ao uso de distâncias.

## 3. Divisão dos Dados
O dataset foi dividido em 80% para treino e 20% para teste. A estratégia stratify=y foi utilizada para manter a proporção de classes entre as amostras.

## 4. Treinamento do Modelo
O modelo KNN foi treinado inicialmente com k = 5. Valores adicionais entre 1 e 20 de k foram testados para observar variações no desempenho.

## 5. Avaliação do Modelo
Foram utilizadas acurácia, matriz de confusão, precision, recall e F1-score. O modelo apresentou desempenho consistente, com acurácia média próxima de 80%. Valores muito baixos de k resultaram em maior sensibilidade a ruídos, enquanto valores muito altos prejudicaram a identificação de padrões locais.

## 6. Conclusão
O projeto foi bem simples e demonstrou que o KNN é capaz de realizar previsões razoáveis no dataset Titanic, desde que o pré-processamento seja bem executado. A normalização desempenhou um papel crucial durante este projeto. Como melhorias, destacam-se o uso de técnicas de balanceamento e testes com modelos mais avançados.