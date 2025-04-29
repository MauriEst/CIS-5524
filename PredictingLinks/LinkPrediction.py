import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
from node2vec import Node2Vec

def load_wikilink_graph(file_path):
    """Load the WikiLinkGraphs CSV file into a NetworkX directed graph."""
    try:
        # Read the first 10 rows of the CSV file for inspection
        df_preview = pd.read_csv(file_path, sep=',', nrows=10, comment='#')
        print("Top 10 rows of the dataset:")
        print(df_preview.to_string(index=False))

        # Save the preview to a CSV file
        preview_file = "top_10_rows_preview.csv"
        df_preview.to_csv(preview_file, index=False)
        print(f"Top 10 rows saved to {preview_file}")

        # Create a directed graph
        G = nx.DiGraph()

        # Read the CSV file in chunks, skipping bad lines
        for chunk in pd.read_csv(file_path, chunksize=1000, sep=',', comment='#', on_bad_lines='skip'):
            # Add edges from the dataframe
            for _, row in chunk.iterrows():
                source = row['page_id_from']
                target = row['page_id_to']
                G.add_edge(source, target)
                # Store page titles as node attributes, handle missing titles
                G.nodes[source]['title'] = str(row['page_title_from']) if pd.notna(row['page_title_from']) else ''
                G.nodes[target]['title'] = str(row['page_title_to']) if pd.notna(row['page_title_to']) else ''
        
        # Check for nodes with missing titles
        missing_titles = sum(1 for node in G.nodes() if not G.nodes[node]['title'])
        print(f"Nodes with missing or empty titles: {missing_titles}")
        
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
        neighbors_u = set(G.successors(u))
        neighbors_v = set(G.successors(v))
        union_size = len(neighbors_u | neighbors_v)
        jaccard = common_neigh / union_size if union_size > 0 else 0
        
        # Preferential Attachment
        pref_attach = G.out_degree(u) * G.out_degree(v)
        
        features.append([common_neigh, jaccard, pref_attach])
    
    return np.array(features)

def compute_textual_features(G, edge_pairs):
    """Compute textual features (cosine similarity of page titles) for edge pairs."""
    titles = [G.nodes[node].get('title', '') for node in G.nodes()]
    valid_titles = [t if t.strip() else 'unknown' for t in titles]
    
    if not any(t != 'unknown' for t in valid_titles):
        print("Warning: No valid titles found. Using zero similarity for all pairs.")
        return np.zeros((len(edge_pairs), 1))
    
    try:
        vectorizer = CountVectorizer(binary=True, stop_words=None, min_df=1)
        title_vectors = vectorizer.fit_transform(valid_titles)
        node_to_index = {node: i for i, node in enumerate(G.nodes())}
        
        text_features = []
        for u, v in edge_pairs:
            u_idx = node_to_index.get(u, -1)
            v_idx = node_to_index.get(v, -1)
            if u_idx != -1 and v_idx != -1 and valid_titles[u_idx] != 'unknown' and valid_titles[v_idx] != 'unknown':
                sim = cosine_similarity(title_vectors[u_idx], title_vectors[v_idx])[0][0]
            else:
                sim = 0
            text_features.append([sim])
        
        return np.array(text_features)
    except Exception as e:
        print(f"Error computing textual features: {e}. Using zero similarity for all pairs.")
        return np.zeros((len(edge_pairs), 1))

def compute_node2vec_features(G, edge_pairs, dimensions=8, walk_length=10, num_walks=5):
    """Compute Node2Vec embedding-based features for edge pairs."""
    model_file = "node2vec_model.pkl"
    
    # Check if cached model exists
    if Path(model_file).exists():
        print(f"Loading cached Node2Vec model from {model_file}...")
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    else:
        print("Generating Node2Vec embeddings...")
        # Convert node IDs to strings (Node2Vec expects strings)
        G_str = nx.relabel_nodes(G, {node: str(node) for node in G.nodes()})
        
        # Generate embeddings
        node2vec = Node2Vec(
            G_str,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=8,  # Adjust based on your CPU cores
            quiet=True,
            p=1.0,
            q=1.0
        )
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # Save model
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Node2Vec model saved to {model_file}")
    
    # Compute cosine similarity of embeddings for each pair
    embedding_features = []
    for u, v in edge_pairs:
        u_str, v_str = str(u), str(v)
        if u_str in model.wv and v_str in model.wv:
            emb_u = model.wv[u_str]
            emb_v = model.wv[v_str]
            sim = np.dot(emb_u, emb_v) / (np.linalg.norm(emb_u) * np.linalg.norm(emb_v) + 1e-8)
        else:
            sim = 0
        embedding_features.append([sim])
    
    return np.array(embedding_features)

def prepare_link_prediction_data(G, sample_size=5000, use_node2vec=False):
    """Prepare data for link prediction by sampling positive and negative edges."""
    pos_edges = random.sample(list(G.edges()), min(sample_size, len(list(G.edges()))))
    all_nodes = list(G.nodes())
    neg_edges = []
    while len(neg_edges) < sample_size:
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if u != v and not G.has_edge(u, v):
            neg_edges.append((u, v))
    
    all_edges = pos_edges + neg_edges
    labels = [1] * len(pos_edges) + [0] * len(neg_edges)
    
    # Compute features
    structural_features = compute_structural_features(G, all_edges)
    textual_features = compute_textual_features(G, all_edges)
    features = np.hstack([structural_features, textual_features])
    
    if use_node2vec:
        node2vec_features = compute_node2vec_features(G, all_edges)
        features = np.hstack([features, node2vec_features])
    
    return features, labels, all_edges

def train_and_evaluate_model(G):
    """Train and evaluate multiple link prediction models."""
    # Prepare data
    print("Preparing link prediction data (Random Forest and Logistic Regression)...")
    features, labels, _ = prepare_link_prediction_data(G, sample_size=5000, use_node2vec=False)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_proba)
    
    # Train and evaluate Logistic Regression
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    lr_precision = precision_score(y_test, lr_pred)
    lr_recall = recall_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_proba)
    
    # Prepare data with Node2Vec features
    print("Preparing link prediction data (Node2Vec)...")
    node2vec_features, node2vec_labels, _ = prepare_link_prediction_data(G, sample_size=5000, use_node2vec=True)
    X_train_nv, X_test_nv, y_train_nv, y_test_nv = train_test_split(
        node2vec_features, node2vec_labels, test_size=0.2, random_state=42
    )
    
    # Train and evaluate Random Forest with Node2Vec
    print("Training Random Forest with Node2Vec features...")
    nv_model = RandomForestClassifier(n_estimators=100, random_state=42)
    nv_model.fit(X_train_nv, y_train_nv)
    nv_pred = nv_model.predict(X_test_nv)
    nv_proba = nv_model.predict_proba(X_test_nv)[:, 1]
    
    nv_precision = precision_score(y_test_nv, nv_pred)
    nv_recall = recall_score(y_test_nv, nv_pred)
    nv_f1 = f1_score(y_test_nv, nv_pred)
    nv_auc = roc_auc_score(y_test_nv, nv_proba)
    
    # Print results
    print("\nLink Prediction Results:")
    print("Random Forest:")
    print(f"Precision: {rf_precision:.4f}, Recall: {rf_recall:.4f}, F1-Score: {rf_f1:.4f}, AUC-ROC: {rf_auc:.4f}")
    print("Logistic Regression:")
    print(f"Precision: {lr_precision:.4f}, Recall: {lr_recall:.4f}, F1-Score: {lr_f1:.4f}, AUC-ROC: {lr_auc:.4f}")
    print("Random Forest with Node2Vec:")
    print(f"Precision: {nv_precision:.4f}, Recall: {nv_recall:.4f}, F1-Score: {nv_f1:.4f}, AUC-ROC: {nv_auc:.4f}")
    
    # Save results to a file
    with open('link_prediction_results.txt', 'w') as f:
        f.write("Random Forest:\n")
        f.write(f"Precision: {rf_precision:.4f}, Recall: {rf_recall:.4f}, F1-Score: {rf_f1:.4f}, AUC-ROC: {rf_auc:.4f}\n")
        f.write("Logistic Regression:\n")
        f.write(f"Precision: {lr_precision:.4f}, Recall: {lr_recall:.4f}, F1-Score: {lr_f1:.4f}, AUC-ROC: {lr_auc:.4f}\n")
        f.write("Random Forest with Node2Vec:\n")
        f.write(f"Precision: {nv_precision:.4f}, Recall: {nv_recall:.4f}, F1-Score: {nv_f1:.4f}, AUC-ROC: {nv_auc:.4f}\n")
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
    
    # Create a smaller subgraph
    print("Creating smaller subgraph...")
    # Get top 25,000 nodes by out-degree
    out_degrees = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, degree in out_degrees[:25000]]
    # Add 25,000 random nodes
    random_nodes = random.sample(list(G.nodes()), 25000)
    # Combine and remove duplicates
    selected_nodes = list(set(top_nodes + random_nodes))
    # Create induced subgraph
    G = G.subgraph(selected_nodes)
    print(f"Using subgraph with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
    
    # Perform link prediction
    print("Starting link prediction...")
    train_and_evaluate_model(G)

if __name__ == "__main__":
    main()