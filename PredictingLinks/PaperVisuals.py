import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import random
import pickle
from pathlib import Path

# Set Seaborn style for clean, professional plots
sns.set_style("whitegrid")

def plot_model_performance(models, auc_roc, filename='model_performance.svg'):
    """
    Create a bar chart comparing AUC-ROC scores across different models with unique colors and a legend.
    
    Args:
        models (list): List of model names.
        auc_roc (list): List of AUC-ROC scores corresponding to the models.
        filename (str): Output filename for the SVG plot.
    """
    plt.figure(figsize=(8, 6))
    
    # Define unique colors for each bar
    colors = ['skyblue', 'orange', 'green']
    
    # Create the bar chart
    bars = plt.bar(models, auc_roc, color=colors)
    
    # Add a legend
    plt.legend(bars, models, title="Models", loc="lower right")
    
    # Add labels and title
    plt.xlabel('Models')
    plt.ylabel('AUC-ROC')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)

    # Zoom in on the y-axis to focus on the range of AUC-ROC scores
    min_auc = min(auc_roc) - 0.01
    max_auc = max(auc_roc) + 0.01
    plt.ylim(min_auc, max_auc)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(filename, format='svg')
    plt.close()
    print(f"Saved model performance plot to {filename}")

def plot_feature_importance(features, importances, filename='feature_importance.svg'):
    """
    Create a horizontal bar plot showing feature importances.
    
    Args:
        features (list): List of feature names.
        importances (list): List of importance scores.
        filename (str): Output filename for the SVG plot.
    """
    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(features))
    plt.barh(y_pos, importances, align='center', color='orange')
    plt.yticks(y_pos, features)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance Analysis')
    plt.tight_layout()
    plt.savefig(filename, format='svg')
    plt.close()
    print(f"Saved feature importance plot to {filename}")

def plot_textual_features(pairs, similarities, filename='textual_features.svg'):
    """
    Create a bar plot illustrating cosine similarities for textual features.
    
    Args:
        pairs (list): List of page title pairs (e.g., 'Physics-Mechanics').
        similarities (list): List of cosine similarity scores.
        filename (str): Output filename for the SVG plot.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(pairs, similarities, color='green')
    plt.xlabel('Page Title Pairs')
    plt.ylabel('Cosine Similarity')
    plt.title('Textual Feature Illustration')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(filename, format='svg')
    plt.close()
    print(f"Saved textual features plot to {filename}")

def visualize_subgraph(G, num_nodes=1000, output_file="subgraph.svg"):
    """Visualize the graph and save it as an image."""
    if G is None:
        print("No graph to visualize.")
        return
    
    subgraph = G.subgraph(list(G.nodes)[:num_nodes]).copy()
    subgraph.remove_edges_from(nx.selfloop_edges(subgraph)) # Remove self-loops from the subgraph
    print(f"Visualizing a subgraph with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges.")
    
    nx.draw(subgraph, pos=nx.spring_layout(subgraph))
    plt.show()
    
    plt.figure(figsize=(12, 8))  # Set the figure size
    pos = nx.spring_layout(subgraph, seed=42)  # Use the spring layout for better visualization
    nx.draw_networkx_nodes(subgraph, pos, node_size=100, node_color='red', alpha=0.9)
    nx.draw_networkx_edges(subgraph, pos, edge_color='black', alpha=0.8)
    plt.title("Graph Visualization")
    plt.axis("off")  # Turn off the axis
    plt.savefig(output_file, dpi=300)  # Save the visualization as an image
    plt.close()
    print(f"Graph visualization saved as '{output_file}'")

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

def main():

    graph_file = "wikilink_graph.gpickle"
    G = load_graph(graph_file)

    # Sample data for model performance
    models = ['Random Forest', 'Logistic Regression', 'RF + Node2Vec']
    auc_roc = [0.9236, 0.9417, 0.9418]
    plot_model_performance(models, auc_roc)

    # Sample data for feature importance
    features = ['Common Neighbors', 'Jaccard', 'Node2Vec']
    importances = [0.35, 0.25, 0.40]
    plot_feature_importance(features, importances)

    # Sample data for textual features
    pairs = ['Physics-Mechanics', 'Quantum-Particle']
    similarities = [0.85, 0.92]
    plot_textual_features(pairs, similarities)

    visualize_subgraph(G)

if __name__ == "__main__":
    main()