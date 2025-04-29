import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_wikilink_graph(file_path):
    """Load the WikiLinkGraphs CSV file into a NetworkX directed graph."""
    try:
        # Read the first 10 rows of the CSV file for inspection
        df_preview = pd.read_csv(file_path, sep='\t', nrows=10)
        print("Top 10 rows of the dataset:")
        print(df_preview)

        # Save the preview to a CSV file
        preview_file = "top_10_rows_preview.csv"
        df_preview.to_csv(preview_file, index=False)
        print(f"Top 10 rows saved to {preview_file}")

        # Create a directed graph
        G = nx.DiGraph()

        # Read the CSV file in chunks, skipping bad lines
        for chunk in pd.read_csv(file_path, chunksize=1000, sep='\t', on_bad_lines='skip'):
            # Add edges from the dataframe
            for _, row in chunk.iterrows():
                source = row['page_id_from']
                target = row['page_id_to']
                G.add_edge(source, target)
        
        return G
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None

def save_graph(G, file_path):
    """Save the graph to a file using pickle."""
    with open(file_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved to {file_path}")

def load_graph(file_path):
    """Load the graph from a file using pickle."""
    if Path(file_path).exists():
        with open(file_path, 'rb') as f:
            G = pickle.load(f)
        print(f"Graph loaded from {file_path}")
        return G
    else:
        print(f"No saved graph found at {file_path}")
        return None

def explore_graph(G):
    """Perform basic exploration of the graph."""
    if G is None:
        print("No graph to explore.")
        return
    
    # Basic statistics
    print(f"Number of nodes: {G.number_of_nodes():,}")
    print(f"Number of edges: {G.number_of_edges():,}")
    avg_degree = sum(dict(G.out_degree()).values()) / G.number_of_nodes()
    print(f"Average out-degree: {avg_degree:.2f}")
    
    # Plot degree distribution
    degrees = [d for _, d in G.out_degree()]
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=50, log=True, density=True)
    plt.title("Out-Degree Distribution (Log Scale)")
    plt.xlabel("Out-Degree")
    plt.ylabel("Density")
    plt.yscale('log')
    plt.savefig('degree_distribution.png')
    plt.close()
    print("Degree distribution plot saved as 'degree_distribution.png'")

def visualize_subgraph(G, num_nodes=500, output_file="graph_visualization.png"):
    """Visualize the graph and save it as an image."""
    if G is None:
        print("No graph to visualize.")
        return
    
    subgraph = G.subgraph(list(G.nodes)[:num_nodes])
    print(f"Visualizing a subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges.")
    
    
    plt.figure(figsize=(12, 8))  # Set the figure size
    pos = nx.spring_layout(subgraph, seed=42)  # Use the spring layout for better visualization
    nx.draw_networkx_nodes(subgraph, pos, node_size=10, node_color='blue', alpha=0.7)
    nx.draw_networkx_edges(subgraph, pos, edge_color='gray', alpha=0.5)
    plt.title("Graph Visualization")
    plt.axis("off")  # Turn off the axis
    plt.savefig(output_file, dpi=300)  # Save the visualization as an image
    plt.close()
    print(f"Graph visualization saved as '{output_file}'")



def visualize_lcc(G, num_nodes=500, output_file="lcc_visualization.png"):
    """Visualize the largest connected component of the graph."""
    if G is None:
        print("No graph to visualize.")
        return
    
    # Extract the largest weakly connected component
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    lcc = G.subgraph(largest_wcc)
    print(f"The largest connected component has {len(lcc.nodes)} nodes and {len(lcc.edges)} edges.")
    
    # Extract a smaller subgraph from the LCC
    subgraph = lcc.subgraph(list(lcc.nodes)[:num_nodes])
    print(f"Visualizing a subgraph of the LCC with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges.")
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw_networkx_nodes(subgraph, pos, node_size=10, node_color='blue', alpha=0.7)
    nx.draw_networkx_edges(subgraph, pos, edge_color='gray', alpha=0.5)
    plt.title("Largest Connected Component (LCC) Subgraph Visualization")
    plt.axis("off")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"LCC subgraph visualization saved as '{output_file}'")

def aggregate_and_visualize_statistics(G):
    """Aggregate and visualize graph statistics."""
    if G is None:
        print("No graph to analyze.")
        return
    
    # Degree distribution
    degrees = [d for _, d in G.out_degree()]
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=50, log=True, density=True)
    plt.title("Out-Degree Distribution (Log Scale)")
    plt.xlabel("Out-Degree")
    plt.ylabel("Density")
    plt.yscale('log')
    plt.savefig("degree_distribution.png")
    plt.close()
    print("Degree distribution plot saved as 'degree_distribution.png'")

def visualize_article_and_neighbors(G, article_id, num_neighbors=20, output_file="article_neighbors.png"):
    """Visualize an article and its nearest neighbors."""
    if G is None:
        print("No graph to visualize.")
        return
    
    if article_id not in G:
        print(f"Article ID {article_id} not found in the graph.")
        return
    
    # Get the neighbors of the article
    neighbors = list(G.neighbors(article_id))[:num_neighbors]
    nodes_to_include = [article_id] + neighbors
    
    # Create the subgraph
    subgraph = G.subgraph(nodes_to_include)
    print(f"Visualizing article {article_id} and its {len(neighbors)} nearest neighbors.")
    
    # Visualize the subgraph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw_networkx_nodes(subgraph, pos, node_size=300, node_color='blue', alpha=0.8)
    nx.draw_networkx_edges(subgraph, pos, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(subgraph, pos, font_size=10, font_color='black')
    plt.title(f"Article {article_id} and its Nearest Neighbors")
    plt.axis("off")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Visualization saved as '{output_file}'")
    
def main():
    # Define file parameters
    csv_file = "enwiki.wikilink_graph.2005-03-01.csv"
    graph_file = "wikilink_graph.gpickle"
    
    # Try to load the graph from a saved file
    G = load_graph(graph_file)
    
    # If the graph is not saved, create it from the CSV
    if G is None:
        print(f"Loading graph from {csv_file}...")
        G = load_wikilink_graph(csv_file)
        save_graph(G, graph_file)

    # Visualize an article and its neighbors
    article_id = 12  # Replace with the ID of the article you want to visualize
    visualize_article_and_neighbors(G, article_id, num_neighbors=5)
    
    # Explore the graph
    # print("Exploring graph properties...")
    # explore_graph(G)

    # Visualize a subgraph
    # print("Visualizing a subgraph...")
    # visualize_subgraph(G, num_nodes=500)
    
    # Visualize the largest connected component
    # print("Visualizing the largest connected component...")
    # visualize_lcc(G, num_nodes=500)
    
    # Aggregate and visualize statistics
    # print("Aggregating and visualizing statistics...")
    # aggregate_and_visualize_statistics(G)

if __name__ == "__main__":
    main()