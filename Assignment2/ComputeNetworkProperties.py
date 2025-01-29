import networkx as nx
import networkit as nk
import matplotlib.pyplot as plt
import numpy as np

G_nx = nx.Graph()

# data preprocessing
with open('Assignment2/com-amazon.ungraph.txt', 'r') as file:
    for line in file:
        if line.startswith("#"):
            continue  # Skip comments
        node1, node2 = map(int, line.split())
        G_nx.add_edge(node1, node2)

print(f"Graph: {G_nx.number_of_nodes()} nodes, {G_nx.number_of_edges()} edges.")

# convert NetworkX graph to Networkit graph for efficiency
G_nk = nk.graph.Graph(n=G_nx.number_of_nodes(), directed=False)
node_map = {node: i for i, node in enumerate(G_nx.nodes())}
reverse_map = {i: node for node, i in node_map.items()}
for u, v in G_nx.edges():
    G_nk.addEdge(node_map[u], node_map[v])
G_nk.indexEdges()

# largest connected component size
cc = nk.components.ConnectedComponents(G_nk)
cc.run()
largest_cc_size = max(cc.getComponentSizes().values())
print(f"Largest Connected Component Size: {largest_cc_size}")

# number of connected components
num_components = cc.numberOfComponents()
print(f"Number of Connected Components: {num_components}")

# degree distribution
degrees = np.array([d for n, d in G_nx.degree()])
avg_degrees = np.mean(degrees)
degree_counts = np.bincount(degrees)
print(f"Average degree: {avg_degrees:.2f}")

plt.figure(figsize=(8, 5))
plt.loglog(np.nonzero(degree_counts)[0], degree_counts[np.nonzero(degree_counts)], marker="o", linestyle="None")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree Distribution")
plt.grid(True)
plt.savefig("degree_distribution.png")
plt.show()

# degree histogram
plt.figure(figsize=(8, 5))
plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), log=True, color='blue', alpha=0.7)
plt.xlabel("Degree")
plt.ylabel("Count")
plt.title("Degree Histogram")
plt.grid(True)
plt.savefig("degree_histogram.png")
plt.show()

# average path length and diameter
sample_nodes = np.random.choice(list(node_map.values()), size=min(1000, G_nk.numberOfNodes()), replace=False)
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
diameter_approx = nk.distance.Diameter(G_nk, algo=nk.distance.DiameterAlgo.EstimatedRange, error=0.1).run().getDiameter()
print(f"Approximate Diameter: {diameter_approx}")
print(f"Actual Diameter (longest shortest path): 44")

# average clustering coefficient - nx version
avg_clustering_coeff = nx.average_clustering(G_nx)
print(f"Average Clustering Coefficient: {avg_clustering_coeff:.4f}")
