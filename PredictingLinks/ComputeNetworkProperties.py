import networkx as nx
import networkit as nk
import matplotlib.pyplot as plt
import numpy as np
import gzip

# data preprocessing
with gzip.open('PredictingLinks/wiki-topcats.txt.gz', 'rt') as file:
    G_nx = nx.read_edgelist(file, create_using=nx.DiGraph())

print(f"Graph: {G_nx.number_of_nodes()} nodes, {G_nx.number_of_edges()} edges.")

# convert NetworkX graph to Networkit graph for efficiency
G_nk = nk.graph.Graph(n=G_nx.number_of_nodes(), directed=True)
node_map = {node: i for i, node in enumerate(G_nx.nodes())}
for u, v in G_nx.edges():
    G_nk.addEdge(node_map[u], node_map[v])
G_nk.indexEdges()

# average path length and diameter
sample_nodes = np.random.choice(list(node_map.values()), 
                                size=min(1000, G_nk.numberOfNodes()), # sample size can be changed
                                replace=False)
total_path_length = 0
count = 0

for node in sample_nodes:
    sp = nk.distance.BFS(G_nk, node).run().getDistances()
    valid_paths = [dist for dist in sp if dist > 0]
    total_path_length += sum(valid_paths)
    count += len(valid_paths)

approx_avg_path_length = total_path_length / count
print(f"Approximate Average Shortest Path Length (using 1000 random nodes): {approx_avg_path_length:.4f}")

# diameter
diameter_approx = nk.distance.Diameter(G_nk, algo=1).run().getDiameter() # 1 for exact, 2 for estimated range
print(f"Approximate Diameter: {diameter_approx}")
print(f"Actual Diameter (longest shortest path): 9")

# average clustering coefficient - nx version
avg_clustering_coeff = nx.average_clustering(G_nx)
print(f"Average Clustering Coefficient: {avg_clustering_coeff:.4f}")
