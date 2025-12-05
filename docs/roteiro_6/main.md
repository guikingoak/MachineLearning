# Análise do Algoritmo PageRank na Rede soc-Epinions1

## 1. Descrição geral

O PageRank é um método de avaliação de importância em grafos dirigidos baseado na distribuição de probabilidade de uma caminhada aleatória. Cada nó recebe um valor proporcional à probabilidade de um agente estar posicionado na mesma posição após um grande número de passos. Esse valor reflete não apenas quantas conexões o nó recebe, mas também a relevância dos nós que apontam para ele.

Em nosso estudo, o algoritmo foi aplicado à rede **soc-Epinions1**, composta por relações de confiança entre usuários em uma plataforma de avaliações. O PageRank se ajusta naturalmente a esse tipo de estrutura, capturando padrões globais de reputação.

---

## 2. Estrutura dos dados e construção do grafo

O dataset contém pares de valores inteiros representando relações de confiança do tipo:

\[
u \rightarrow v
\]

em que **u confia em v**. Após a leitura do arquivo:

- nós identificados: **75.879**  
- arestas dirigidas: **508.837**

A modelagem considerou apenas a sua estrutura original, sem transformações ou novas filtragens. O grafo foi construído como **dirigido**, preservando a direção e permitindo a avaliação de caminhos assimétricos de reputação.

As características estruturais indicam uma rede grande e esparsa, com distribuição desigual de graus e presença de usuários com papel de hubs de confiança. Esses padrões influenciam diretamente o comportamento do PageRank.

---

## 3. Implementação do PageRank

### 3.1 Formulação adotada

A implementação segue a forma iterativa padrão:

\[
PR(i) = \frac{1-d}{N} + d \sum_{j \in \text{In}(i)} \frac{PR(j)}{\text{outdeg}(j)},
\]

com:

- **N**: número total de nós,  
- **d = 0.85**: fator de amortecimento,  
- **In(i)**: conjunto de nós que apontam para i,  
- **outdeg(j)**: grau de saída do nó j.

### 3.2 Estratégias utilizadas

- Inicialização uniforme do vetor de PageRank.  
- Computação iterativa usando listas de adjacência, evitando a construção de matriz completa.  
- Redistribuição do PageRank acumulado em nós sem arestas de saída (dangling nodes).  
- Controle de convergência pela diferença entre vetores consecutivos.  
- Critério de parada: erro inferior a \(10^{-6}\) ou limite de 200 iterações.

Essa abordagem é adequada para grafos grandes, reduz o consumo de memória e mantém o custo proporcional ao uso de nós e arestas.

### 3.3 Implementação de referência

Para verificar consistência, os mesmos parâmetros foram aplicados ao método nativo do NetworkX (`networkx.pagerank`). Os resultados dos dois métodos foram comparados quantitativamente.

---

## 4. Resultados obtidos

### 4.1 Convergência

A implementação manual convergiu de forma estável:

- iterações necessárias: **53**  
- erro final: **9.78 × 10⁻⁷**  
- tempo de execução (ambiente local): **≈ 10.5 s**

Esses valores são esperados para grafos desse porte usando iteração de potência com tolerância fina.

---

### 4.2 Nós com maior PageRank

Os dez nós com maior importância na rede, segundo a implementação manual, são:

| Rank | Nó   | PageRank |
|-----:|-----:|---------:|
| 1 | 18   | 0.0045350 |
| 2 | 737  | 0.0031504 |
| 3 | 118  | 0.0021220 |
| 4 | 1719 | 0.0020781 |
| 5 | 136  | 0.0019870 |
| 6 | 790  | 0.0019689 |
| 7 | 143  | 0.0019568 |
| 8 | 40   | 0.0018248 |
| 9 | 1619 | 0.0015362 |
|10 | 725  | 0.0014960 |

A execução com NetworkX teve como resultado a mesma lista de nós e a mesma ordenação, com pequenas diferenças numéricas na sexta casa decimal.

---

### 4.3 Comparação entre as duas implementações

Para avaliar similaridade quantitativa, foram calculadas as correlações entre os vetores completos de PageRank:

- **Pearson:** 0.9999978919  
- **Spearman:** 0.9999997531  

Os dois indicadores apontam equivalência entre os métodos: a estrutura de ordenação e a escala dos valores foram preservadas.

O gráfico de dispersão gerado (`pagerank_comparison.png`) mostra os pontos concentrados ao longo da diagonal, indicando alinhamento direto entre as duas soluções.

---

## 5. Interpretação dos resultados

Os usuários no topo do ranking possuem valores de PageRank bem maiores que a média da rede. Isso sugere:

- A presença de usuários que costumam receber confiança de múltiplas regiões do grafo.  
- forte acumulação de importância de alguns nós centrais.
- reforço através de conexões de alta qualidade (nós bem classificados apontando para outros nós centrais).

O PageRank costuma capturar este comportamento de uma forma mais informativa que o grau de entrada simples, pois considera a relevância dos emissores das arestas. Assim, há a possibilidade de  um nó com menor quantidade de entradas poder superar outro mais conectado se receber ligações de usuários igualmente relevantes.

Essa característica costuma ser adequada para redes de confiança, em que o valor das conexões costuma ser tão importante quanto a quantidade.

---

## 6. Conclusão

A aplicação do PageRank ao grafo soc-Epinions1 nos mostrou que a estrutura da rede apresenta forte concentração de importância em poucos usuários, compatível com os padrões típicos de redes sociais dirigidas.

A implementação manual reproduziu corretamente os comportamentos esperados do algoritmo, confirmado pela comparação com o NetworkX. A convergência ocorreu sem nenhuma instabilidade e os resultados dos dois métodos apresentaram praticamente a mesma coisa no quesito ordenação de nós.

A análise final indica que o PageRank é um método apropriado para identificar usuários centrais em redes de confiança, oferecendo uma medida de relevância estrutural que vai além do número de conexões recebidas e reflete a propagação global de reputação ao longo do grafo.
