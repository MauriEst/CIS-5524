import numpy as np
import networkx as nx

# Initialize an empty graph
G = nx.Graph()

# Read the text file and add edges to the graph
with open('Assignment2/com-amazon.ungraph.txt', 'r') as file:
    for line in file:
        node1, node2 = map(int, line.strip().split())
        G.add_edge(node1, node2)

# compute global network properties
def compute_network_properties(G):
    # size and diameter of the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    size = len(largest_cc)
    diameter = nx.diameter(subgraph) if nx.is_connected(subgraph) else float('inf')
    
    # degree distribution
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees)
    
    # average path length
    avg_path_length = nx.average_shortest_path_length(subgraph) if nx.is_connected(subgraph) else float('inf')
    
    # average clustering coefficient
    avg_clustering_coeff = nx.average_clustering(G)
    cluster_coeff = nx.clustering(G)
    
    print(f"-----------Amazon Product Graph Properties-------------")
    print(f"Size of largest connected component: {size}")
    print(f"Diameter of largest connected component: {diameter}")
    print(f"Average degree: {avg_degree}")
    print(f"Average path length: {avg_path_length}")
    print(f"Average clustering coefficient: {avg_clustering_coeff}\n")
    print(f" clustering coefficient: {cluster_coeff}\n")

# Call the function to compute and print network properties
compute_network_properties(G)