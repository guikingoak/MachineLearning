import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from scipy import stats

# ----------------------------------------------------------
# 1. Leitura do dataset local
# ----------------------------------------------------------

EDGE_FILE = "https://raw.githubusercontent.com/guikingoak/MachineLearning/main/docs/PageRank/soc-Epinions1.txt"

df = pd.read_csv(
    EDGE_FILE,
    sep=r"\s+",
    comment="#",
    header=None,
    names=["source", "target"]
)

print("Dataset carregado.")
print(df.head())
print(df.shape)

# ----------------------------------------------------------
# 2. Construção do grafo
# ----------------------------------------------------------

G = nx.DiGraph()
G.add_edges_from(df.values.tolist())

print(f"Grafo criado: {G.number_of_nodes():,} nós, {G.number_of_edges():,} arestas")

# ----------------------------------------------------------
# 3. Implementação do PageRank manual
# ----------------------------------------------------------

def pagerank_scratch(G, d=0.85, tol=1e-6, max_iter=200):
    nodes = list(G.nodes())
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}

    pr = np.ones(N) / N
    out_deg = np.array([G.out_degree(n) for n in nodes], dtype=float)
    teleport = (1 - d) / N

    incoming = [[] for _ in range(N)]
    for u, v in G.edges():
        incoming[idx[v]].append(idx[u])

    for iteration in range(max_iter):
        pr_new = np.zeros(N)

        for i in range(N):
            s = 0
            for j in incoming[i]:
                if out_deg[j] > 0:
                    s += pr[j] / out_deg[j]
            pr_new[i] = teleport + d * s

        dangling = pr[out_deg == 0].sum()
        if dangling > 0:
            pr_new += d * dangling / N

        err = np.abs(pr_new - pr).sum()
        if err < tol:
            return {nodes[i]: float(pr_new[i]) for i in range(N)}, iteration, err

        pr = pr_new

    return {nodes[i]: float(pr[i]) for i in range(N)}, max_iter, err

# ----------------------------------------------------------
# 4. Execução do PageRank manual
# ----------------------------------------------------------

print("Executando PageRank manual...")
start = time.time()
pr_manual, iters, err = pagerank_scratch(G, d=0.85)
end = time.time()

print(f"Convergiu em {iters} iterações, erro {err:.2e}")
print(f"Tempo: {end - start:.2f}s")

# ----------------------------------------------------------
# 5. PageRank NetworkX
# ----------------------------------------------------------

print("Executando PageRank (networkx)...")
pr_nx = nx.pagerank(G, alpha=0.85, tol=1e-8)

# ----------------------------------------------------------
# 6. Top-10 manual e NetworkX
# ----------------------------------------------------------

def topk(d, k=10):
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

top10_manual = topk(pr_manual, 10)
top10_nx = topk(pr_nx, 10)

print("\nTOP 10 (manual):")
for r, (n, s) in enumerate(top10_manual, 1):
    print(r, n, s)

print("\nTOP 10 (networkx):")
for r, (n, s) in enumerate(top10_nx, 1):
    print(r, n, s)

# ----------------------------------------------------------
# 7. Correlação entre os métodos
# ----------------------------------------------------------

nodes_sorted = sorted(pr_manual.keys())
v_manual = np.array([pr_manual[n] for n in nodes_sorted])
v_nx = np.array([pr_nx[n] for n in nodes_sorted])

pearson = np.corrcoef(v_manual, v_nx)[0, 1]
spearman, _ = stats.spearmanr(v_manual, v_nx)

print("\nCorrelação Pearson:", pearson)
print("Correlação Spearman:", spearman)

# ----------------------------------------------------------
# 8. Gráfico de comparação
# ----------------------------------------------------------

plt.figure(figsize=(8,6))
plt.scatter(v_manual[:1000], v_nx[:1000], s=10, alpha=0.5)
plt.xlabel("PageRank manual")
plt.ylabel("PageRank NetworkX")
plt.title("Comparação PageRank (1000 primeiros nós)")
plt.grid(True)
plt.tight_layout()
plt.savefig("pagerank_comparison.png")
plt.close()

print("\nGráfico salvo: pagerank_comparison.png")
print("Concluído.")
