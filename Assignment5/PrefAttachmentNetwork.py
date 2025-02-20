import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

def scale_free(n, m):
    if m < 1 or  m >=n: 
        raise nx.NetworkXError("Preferential attactment algorithm must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n)) 
    # Add m initial nodes (m0 in barabasi-speak)
    G=nx.empty_graph(m)

    # Target nodes for new edges
    targets=list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes=[]
    # Start adding the other n-m nodes. The first node is m.
    source=m
    while source<n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source]*m,targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source]*m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachement)
        targets = random.sample(repeated_nodes, m)
        source += 1
    return G

# 1 way to plot
def plot_degree_distribution(G, title):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
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
    G_nx = scale_free(size, m)
    plot_degree_distribution(G_nx, f"Degree Distribution (Log-Log Scale) for {size} Nodes")

# G_nx = scale_free(10000, 4)

# 2nd way to plot
degrees = np.array([d for n, d in G_nx.degree()])
degree_counts = np.bincount(degrees)

plt.figure(figsize=(8, 5))
plt.loglog(np.nonzero(degree_counts)[0], degree_counts[np.nonzero(degree_counts)], marker="o", linestyle="None")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree Distribution")
plt.grid(True)
plt.savefig("Assignment5/degree_distribution.png")
plt.show()

# Draw network
# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G_nx)
# nx.draw_networkx_nodes(G_nx, pos, node_size=100, node_color='lightblue')
# nx.draw_networkx_edges(G_nx, pos, alpha=0.5)
# nx.draw_networkx_labels(G_nx, pos, font_size=8)

# plt.title("Preferential Attachment Network")
# plt.show()

# ------------------------------------------------------------- #
# # Creating a graph with 4 nodes and 3 edges, initial nodes (m0)
# G_nx = nx.Graph()
# G_nx.add_edge(0, 1)
# G_nx.add_edge(3, 0)
# G_nx.add_edge(2, 0)

# # Function to expand existing graph using preferential attachment
# def pref_attachment(G, n):
#     """
#     Expands the existing graph using preferential attachment.

#     Parameters:
#     G (networkx.Graph): The existing graph.
#     n (int): The number of new nodes to add.

#     Returns:
#     networkx.Graph: The updated graph with new nodes added.
#     """
#     for i in range(n):
#         new_node = len(G.nodes)
#         targets = list(G.nodes)
#         degrees = [G.degree(node) for node in targets]
#         total_degree = sum(degrees)
#         probabilities = [degree / total_degree for degree in degrees]
        
#         # Select a node based on the calculated probabilities
#         selected_node = nx.utils.random_sequence.discrete_sequence(1, probabilities)[0]
        
#         # Add the new node with an edge to the selected node
#         G.add_edge(new_node, selected_node)
    
#     return G

# G_nx = pref_attachment(G_nx, 100)

# # Draw network
# plt.figure(figsize=(12, 8))  # Set the figure size
# pos = nx.spring_layout(G_nx)  # Use the spring layout for better positioning
# nx.draw_networkx_nodes(G_nx, pos, node_size=100, node_color='lightblue')  # Customize node appearance
# nx.draw_networkx_edges(G_nx, pos, alpha=0.5)  # Customize edge appearance
# nx.draw_networkx_labels(G_nx, pos, font_size=8)  # Add labels to nodes

# plt.title("Preferential Attachment Network 1")
# plt.show()


# def initialize_graph(m0):
#     """Create an initial small graph with m0 nodes and random connections."""
#     G = nx.Graph()
    
#     # Add nodes
#     for i in range(m0):
#         G.add_node(i)
    
#     # Add random edges to ensure connectivity
#     for i in range(m0):
#         possible_targets = list(set(range(m0)) - {i})  # Exclude self-loops
#         num_edges = random.randint(1, m0 - 1)  # Each node gets at least 1 edge
#         chosen_targets = random.sample(possible_targets, num_edges)
#         for target in chosen_targets:
#             G.add_edge(i, target)
    
#     return G

# def add_node(G, degree_dict, m):
#     """Add a new node to the graph using the preferential attachment rule."""
#     existing_nodes = list(G.nodes())
#     degrees = [degree_dict[node] for node in existing_nodes]
    
#     # Select m existing nodes based on their degree probability
#     targets = random.choices(existing_nodes, weights=degrees, k=m)
    
#     # Add new node and connect it
#     new_node = max(G.nodes()) + 1
#     G.add_node(new_node)
#     for target in targets:
#         G.add_edge(new_node, target)
#         degree_dict[new_node] = degree_dict.get(new_node, 0) + 1
#         degree_dict[target] += 1

# def generate_preferential_attachment_graph(N, m0, m):
#     """Generate a network using preferential attachment."""
#     G = initialize_graph(m0)
#     degree_dict = {node: G.degree(node) for node in G.nodes()}

#     # Add nodes one by one
#     for _ in range(N - m0):
#         add_node(G, degree_dict, m)
    
#     return G, degree_dict

# # Parameters
# m0 = 5     # Initial small network size
# N = 100  # Total number of nodes
# m = 3      # Number of edges each new node adds

# # Generate the network
# G, degree_dict = generate_preferential_attachment_graph(N, m0, m)

# # Plot the degree distribution (log-log scale)
# degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
# plt.figure(figsize=(8, 6))
# plt.hist(degree_sequence, bins=50, log=True, color='b', alpha=0.7)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel("Degree (log scale)")
# plt.ylabel("Frequency (log scale)")
# plt.title("Degree Distribution (Log-Log Scale)")
# plt.show()

# # Draw network
# plt.figure(figsize=(12, 8))  # Set the figure size
# pos = nx.spring_layout(G_nx)  # Use the spring layout for better positioning
# nx.draw_networkx_nodes(G_nx, pos, node_size=100, node_color='lightblue')  # Customize node appearance
# nx.draw_networkx_edges(G_nx, pos, alpha=0.5)  # Customize edge appearance
# nx.draw_networkx_labels(G_nx, pos, font_size=8)  # Add labels to nodes

# plt.title("Preferential Attachment Network 2")
# plt.show()