# Entrega Projeto II - Relatório


## 1. Introdução

Este projeto tem como objetivo descrever o processo de desenvolvimento e análise realizado, incluindo alguns dos métodos que utilizei, técnicas, resultados e propostas de melhorias.

O objetivo principal deste projeto foi desenvolver, testar e avaliar modelos computacionais a partir do conjunto de dados "housing.csv", utilizado para análise de preço de imóveis com atributos como quantidade de quartos, numero de banheiros, idade do imóvel, densidade demográfica da região, entre outras informações que possam ser utilizadas para explicar o valor final das residências.

## 2.  Pré-Processamento

Inicialmente após realizarmos o carregamento do dataset, foi visto a presença de alguns valores ausentes, a tipagem das colunas e algumas inconsistências nos registros. Apartir desta inspeção foi feita uma imputação , adotando a média das respectivas colunas para evitar reduzir o volume dos dados. Além disso, realizei uma padronização das variáveis numéricas envolvendo a verificação das faixas e distribuição com a finalidade de detectar qualquer outlier que influenciasse negativamente o desempenho do modelo.



Com base na distribuição da qualidade, o problema foi transformado em uma classificação binária. Uma nova variável, quality_category, foi criada, onde vinhos com nota maior ou igual a 6 foram classificados como "Bons" (1) e os demais como "Regulares" (0). Esta abordagem simplifica o modelo e cria um problema de classificação mais balanceado.

O dataset não apresentou valores ausentes.

Por fim, todas as características preditoras foram padronizadas utilizando o StandardScaler do Scikit-learn. Esta etapa é fundamental para o desempenho de algoritmos baseados em distância, como KNN e K-Means, garantindo que todas as features contribuam de forma equitativa para o modelo.


## 3. Desenvolvimento do Modelo

Para este projeto utilizamos um modelo baseado em Regressão Linear, Random Forest e o Gradient Boosting.
A regressão linear foi utilizada pois não há necessidade de usar nenhum hiperparâmetro complexo e o ajuste é baseado em minimizar o erro quadrático, gerando uma linha para se utilizar como referencia inicial. Em seguida utilizamos o Random Forest para tentar captar as relações que não são lineares, utilizamos cerca de 100 árvores de decissão através do n_estimator com a intenção de reduzir a variância e melhorar a generalização, evitando que o modelo se torne sensível a ruidos visto no conjunto de treino. Por último utilizamos o Gradient Boosting para garantir o treinamento das árvores de forma sequencial, garantindo que com cada nova árvore criada, ela corrigirá os erros cometidos pelas anteriores, resultando em um modelo final otimizado. Utilizamos o próprio padrão do modelo onde a Learning rate foi de 0.1 e o n_estimators também foi de 100 com a profundidade de 3.

Após a construção dos três modelos, cada um deles foi treinado usando o conjunto de treino já processado e em seguida foram todos avaliados através da MAE e RMSE, permitindo com que a gente comparasse diretamente o desempenho entre abordagens lineares e não lineares. De modo geral, como foi esperado os modelos baseados em árvores, em especifico o Gradient Boosting , apresentou um desempenho superior ao da regressão linear normal, refletindo a capacidade que este modelo tem em lidar com relações mais complexas entre atributos.

# 4.  Resultados obtidos

Após o treinamento dos três modelos, foi possível comparar diretamente o desempenho de cada um deles através da métrica de erro. Os resultados a seguir mostram as diferenças entre as abordagens:

A Regressão Linear usada como modelo de referência, apresentou o desempenho mais modesto. Seus resultados foram: MAE de 50.670,49, MSE de 4.908.290.571,35, RMSE de 70.059,19 e R² de 0,6254. Esses valores indicam que o modelo conseguiu capturar apenas a tendência geral do comportamento dos preços, mas houve dificuldade para representar relações mais complexas existentes dentro dataset. O valor alto de RMSE mostra que o erro médio das previsões em relação ao valor real do imóvel ainda era bastante alto, reforçando a limitação da suposição de linearidade.

Em seguida o Random Forest Regressor apresentou o melhor desempenho entre os três modelos. Os resultados foram: MAE de 31.471,47, MSE de 2.382.459.089,47, RMSE de 48.810,44 e R² de 0,8182. Esse modelo conseguiu reduzir muito o erro em relação à regressão linear e apresentou o maior valor de R², mostrando que conseguiu explicar mais de 81% da variabilidade dos preços dos imóveis. A combinação de múltiplas árvores permitiu capturar relações não lineares e diminuir o impacto dos ruídos, o que explica a melhora no desempenho.

Já o Gradient Boosting Regressor, embora também tenha superado a regressão linear, ficou atrás do Random Forest nos resultados finais. Seus valores foram: MAE de 38.276,87, MSE de 3.125.879.704,30, RMSE de 55.909,57 e R² de 0,7615. O modelo apresentou boa capacidade de ajuste, mas o desempenhodo Random Forest indica que para esse dataset específico, o boosting sequencial não capturou as interações de forma tão eficaz quanto a agregação de múltiplas árvores independentes.

Estes resultados sugerem que há espaço para melhoras futuras, como técnicas de feature engineering mais avançadas e otimizações nos hiperparâmetros de boosting. Ainda assim, para o conjunto de modelos utilizados, a evidência mostra que o Random Forest Regressor oferece o melhor equílibrio entre precisão , generalização e estabilidade.

# 5. Conclusão

O desenvolvimento deste projeto revelou limitações naturais do próprio dataset, como a presença de variáveis altamente correlacionadas, ruídos e grande diferença entre regiões, o que impacta a capacidade explicativa de qualquer técnica aplicada. Esses aspectos apontam para a relevância de aprofundar etapas como feature engineering, enriquecimento do conjunto de dados e estratégias avançadas de validação.

O trabalho também contribuiu para consolidar habilidades práticas na comparação de algoritmos, interpretação de métricas, organização de pipeline e avaliação crítica dos resultados. A principio houve um grande destaque do papel de métodos ensemble como alternativas quando a tarefa exige equilíbrio entre robustez, estabilidade e sensibilidade a padrões escondidos nos dados.

Em síntese, mais do que identificar qual modelo é melhor neste dataset, o projeto demonstrou o valor de um processo analítico estruturado, da experimentação guiada e da escolha de técnicas adequadas ao contexto. Os resultados abrem espaço para melhorias futuras e para a aplicação de abordagens mais avançadas, ao mesmo tempo em que solidificam uma base consistente de entendimento sobre modelage preditiva em problemas reais.