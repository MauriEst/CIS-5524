from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pymnet
import networkx as nx

"""
This script reads the data from assignment 1 CSV file and creates adjacency 
matrices for each layer of the multilayer network. Then we use pymnet to
create a multilayer network and visualize it. We also compute global network
properties for each layer.
"""

data = pd.read_csv('Assignment1/Assignment_1_Data.csv')
data.columns = data.columns.str.strip() # remove leading/trailing whitespaces
data = data.rename(columns={
    'Do you have some background on Data Mining (CIS 4523/5523) or Machine Learning?': 'Data Mining/ML',
    'Do you have some background on Python or R Programming?': 'Python/R',
    'Do you have some background on Graphs or Statistics?': 'Graphs/Statistics'
})

students = data['Student Name'].unique()
num_students = len(students)

# initialize adjacency matrices for each layer
layer1 = np.zeros((num_students, num_students), dtype=int)
layer2 = np.zeros((num_students, num_students), dtype=int)
layer3 = np.zeros((num_students, num_students), dtype=int)

# create a mapping from student names to indices
student_index = {student: idx for idx, student in enumerate(students)} # {student_name: index}

# Fill the adjacency matrices
for _, row in data.iterrows():
    student = row['Student Name']
    idx = student_index[student]
    
    for other_student in students:
        other_idx = student_index[other_student]
        
        if row['Data Mining/ML'] == data[data['Student Name'] == other_student]['Data Mining/ML'].values[0]:
            layer1[idx, other_idx] = 1
        if row['Python/R'] == data[data['Student Name'] == other_student]['Python/R'].values[0]:
            layer2[idx, other_idx] = 1
        if row['Graphs/Statistics'] == data[data['Student Name'] == other_student]['Graphs/Statistics'].values[0]:
            layer3[idx, other_idx] = 1

np.savetxt('Assignment1/ML.csv', layer1, delimiter=',', fmt='%d')
np.savetxt('Assignment1/PythonR.csv', layer2, delimiter=',', fmt='%d')
np.savetxt('Assignment1/GraphStat.csv', layer3, delimiter=',', fmt='%d')

print("Adjacency matrices saved to ML.csv, PythonR.csv, and GraphStat.csv")

# create a multilayer network using pymnet
ml_net = pymnet.MultilayerNetwork(aspects=1)

# add nodes and edges for each layer
for i, student in enumerate(students):
    for j, other_student in enumerate(students):
        if layer1[i, j] == 1:
            ml_net[student, other_student, 'Data Mining/ML'] = 1
        if layer2[i, j] == 1:
            ml_net[student, other_student, 'Python/R'] = 1
        if layer3[i, j] == 1:
            ml_net[student, other_student, 'Graphs/Statistics'] = 1

# compute global network properties
def compute_network_properties(G, layer_name):
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
    
    print(f"Layer: {layer_name}")
    print(f"Size of largest connected component: {size}")
    print(f"Diameter of largest connected component: {diameter}")
    print(f"Average degree: {avg_degree}")
    print(f"Average path length: {avg_path_length}")
    print(f"Average clustering coefficient: {avg_clustering_coeff}\n")

# create networkx graphs for each layer for computation
G1 = nx.from_numpy_array(layer1)
G2 = nx.from_numpy_array(layer2)
G3 = nx.from_numpy_array(layer3)

# call compute_network_properties for each layer
compute_network_properties(G1, 'Data Mining/ML')
compute_network_properties(G2, 'Python/R')
compute_network_properties(G3, 'Graphs/Statistics')

# visualize the multilayer network
pymnet.draw(ml_net, 
            layerLabelDict={0: 'Data Mining/ML', 1: 'Python/R', 2: 'Graphs/Statistics'},
            layout='spring', 
            show=True)
