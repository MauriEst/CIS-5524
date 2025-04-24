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
    
def custom_common_neighbors(G, u, v):
    """Compute common neighbors for a directed graph using out-neighbors."""
    neighbors_u = set(G.successors(u))  # Out-neighbors of u
    neighbors_v = set(G.successors(v))  # Out-neighbors of v
    return len(neighbors_u & neighbors_v)



def compute_structural_features(G, edge_pairs):
    """Compute structural features for a list of edge pairs."""
    features = []
    for u, v in edge_pairs:
        # Common Neighbors
        common_neigh = custom_common_neighbors(G, u, v)
        
        # Jaccard Similarity
        neighbors_u = set(G[u])
        neighbors_v = set(G[v])
        union_size = len(neighbors_u | neighbors_v)
        jaccard = common_neigh / union_size if union_size > 0 else 0
        
        # Preferential Attachment
        pref_attach = G.out_degree(u) * G.out_degree(v)
        
        features.append([common_neigh, jaccard, pref_attach])
    
    return np.array(features)

def compute_textual_features(G, edge_pairs):
    """Compute textual features (cosine similarity of page titles) for edge pairs."""
    # Collect all titles
    titles = [G.nodes[node].get('title', '') for node in G.nodes()]
    vectorizer = CountVectorizer(binary=True)
    title_vectors = vectorizer.fit_transform(titles)
    
    # Map node IDs to their index in the title list
    node_to_index = {node: i for i, node in enumerate(G.nodes())}
    
    # Compute cosine similarity for each pair
    text_features = []
    for u, v in edge_pairs:
        u_idx = node_to_index.get(u, -1)
        v_idx = node_to_index.get(v, -1)
        if u_idx != -1 and v_idx != -1:
            sim = cosine_similarity(title_vectors[u_idx], title_vectors[v_idx])[0][0]
        else:
            sim = 0
        text_features.append([sim])
    
    return np.array(text_features)

def prepare_link_prediction_data(G, sample_size=10000):
    """Prepare data for link prediction by sampling positive and negative edges."""
    # Sample positive edges (existing)
    pos_edges = random.sample(list(G.edges()), sample_size)
    
    # Sample negative edges (non-existing)
    all_nodes = list(G.nodes())
    neg_edges = []
    while len(neg_edges) < sample_size:
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if u != v and not G.has_edge(u, v):
            neg_edges.append((u, v))
    
    # Combine and label
    all_edges = pos_edges + neg_edges
    labels = [1] * sample_size + [0] * sample_size
    
    # Compute features
    structural_features = compute_structural_features(G, all_edges)
    textual_features = compute_textual_features(G, all_edges)
    features = np.hstack([structural_features, textual_features])
    
    return features, labels, all_edges

def train_and_evaluate_model(G):
    """Train and evaluate a link prediction model."""
    # Prepare data
    print("Preparing link prediction data...")
    features, labels, _ = prepare_link_prediction_data(G, sample_size=10000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Train Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print("\nLink Prediction Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Save results to a file
    with open('link_prediction_results.txt', 'w') as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"AUC-ROC: {auc:.4f}\n")
    print("Results saved to 'link_prediction_results.txt'")

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

    # Perform link prediction
    print("Starting link prediction...")
    train_and_evaluate_model(G)

if __name__ == "__main__":
    main()