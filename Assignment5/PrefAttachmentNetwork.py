import networkx as nx
import random
import matplotlib.pyplot as plt

def scale_free(n, m):
    if m < 1 or  m >= n: 
        raise nx.NetworkXError("Preferential attactment algorithm must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n)) 
    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.empty_graph(m)
    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
     # Track degrees of specific nodes
    degrees_over_time = {4: [], 100: [], 1000: [], 5000: []} # nodes 4, 100, 1000, 5000
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Record degrees at key milestones
        if source >= 100:
            degrees_over_time[100].append(G.degree(100))
        if source >= 1000:
            degrees_over_time[1000].append(G.degree(1000))
        if source >= 5000:
            degrees_over_time[5000].append(G.degree(5000))

        degrees_over_time[4].append(G.degree(4))

        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachement)
        targets = random.sample(repeated_nodes, k=m)
        source += 1

    return G, degrees_over_time

# 1 way to plot
def plot_degree_distribution(G, title):
    degree_sequence = sorted([d for n, d in G.degree()], reverse = True)
    degree_count = {degree: degree_sequence.count(degree) for degree in degree_sequence}
    degrees, counts = zip(*degree_count.items())
    
    plt.figure(figsize=(8, 6))
    plt.loglog(degrees, counts, 'b.', alpha=0.7)
    plt.xlabel("Degree (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title(title)
    plt.show()

# Generate and plot networks for 100, 1,000, and 10,000 nodes
sizes = [100, 1000, 10000]
m = 4

for size in sizes:
    G_nx, degrees_over_time = scale_free(size, m)
    clustering_coefficient = nx.average_clustering(G_nx)
    print(f"Clustering coefficient for Scale Free graph of size {size} is {clustering_coefficient}")
    if size == 100:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G_nx)
        nx.draw_networkx_nodes(G_nx, pos, node_size=100, node_color='lightblue')
        nx.draw_networkx_edges(G_nx, pos, alpha=0.5)
        nx.draw_networkx_labels(G_nx, pos, font_size=8)
        plt.title("Preferential Attachment Network")
        plt.show()
    plot_degree_distribution(G_nx, f"Degree Distribution (Log-Log Scale) for {size} Nodes")

# Plot degree dynamics
plt.figure(figsize=(10, 6))
for node, degrees in degrees_over_time.items():
    plt.plot(degrees, label=f'Node {node}')
plt.xlabel('Time')
plt.ylabel('Degree')
plt.title('Degree Dynamics of Specified Nodes')
plt.legend()
plt.show()

degreeOfNode4 = G_nx.degree(4)
degreeOfNode100 = G_nx.degree(100)
degreeOfNode5000 = G_nx.degree(5000)
print(f"Degree of node 4 is {degreeOfNode4} and node 100 is {degreeOfNode100} and node 5000 is {degreeOfNode5000}")