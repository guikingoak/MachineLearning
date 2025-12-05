# PageRank – Rede soc-Epinions1

O dataset soc-Epinions1 representa uma rede de confiança em que cada aresta dirigida `A → B` significa que o usuário A confia em B. A rede possui:

- 75.879 usuários  
- 508.837 arestas  

O objetivo deste exercício foi implementar o PageRank manualmente, comparar com a função nativa do NetworkX e identificar os usuários mais influentes.

## Implementação

A versão manual utiliza iteração de potência, com:

- fator de amortecimento: **0.85**  
- tolerância: **1e-6**  
- máximo de **200** iterações  

Também foi aplicado o tratamento clássico de dangling nodes. Para comparação, utilizou-se `networkx.pagerank` com a mesma configuração de α.

## Resultados

### Dataset carregado

O arquivo foi lido corretamente, totalizando 508.837 relações.

### Convergência

- convergiu em **53 iterações**  
- erro final: **9.78e-07**  
- tempo aproximado: **10.51s**  

### Top 10 (Implementação Manual)

1 18 0.0045350
2 737 0.0031504
3 118 0.0021220
4 1719 0.0020781
5 136 0.0019870
6 790 0.0019688
7 143 0.0019568
8 40 0.0018248
9 1619 0.0015362
10 725 0.0014960



### Top 10 (NetworkX)

1 18 0.0045361
2 737 0.0031504
3 118 0.0021224
4 1719 0.0020793
5 136 0.0019878
6 790 0.0019700
7 143 0.0019572
8 40 0.0018245
9 1619 0.0015369
10 725 0.0014953


Os valores são praticamente idênticos.

### Correlação entre os métodos

Correlação Pearson: 0.9999978918929292
Correlação Spearman: 0.9999997531328336


As duas implementações produzem praticamente o mesmo vetor de PageRank.

### Gráfico de Comparação

A imagem compara os primeiros 1000 nós:

![Comparação PageRank](pagerank_comparison.png)

O alinhamento com a diagonal confirma consistência entre as implementações.

## Conclusão

O PageRank manual apresentou resultados equivalentes ao método nativo do NetworkX. O nó **18** aparece como o mais influente da rede. A convergência ocorreu de forma estável e os valores de correlação confirmam a correção da implementação.
