import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# data preprocessing
data = pd.read_csv('Assignment_1_Data.csv')
data.columns = data.columns.str.strip() # remove leading/trailing whitespaces
data = data.rename(columns={
    'Do you have some background on Data Mining (CIS 4523/5523) or Machine Learning?': 'Data Mining/ML',
    'Do you have some background on Python or R Programming?': 'Python/R',
    'Do you have some background on Graphs or Statistics?': 'Graphs/Statistics'
})

# create a bipartite graph
B = nx.Graph()
courses = data.columns[1:]
B.add_nodes_from(courses, bipartite=1) # first set of nodes are courses

# iterate over data to add student nodes and edges
for _, row in data.iterrows():
    student = row['Student Name']
    B.add_node(student, bipartite=0)
    for course in courses:
        if row[course] == 'Yes':
            B.add_edge(student, course)

# drawing bipartite graph
plt.figure(figsize=(12, 8))
plt.title('Bipartite Network of CIS5524 Students and Pre Course Work')

pos = nx.drawing.layout.bipartite_layout(B, courses)
nx.draw(B, pos, with_labels=True, 
        node_color=['blue' if node in courses else 'green' for node in B.nodes()],
        node_size=500, edge_color='black', 
        font_size=10, font_weight='bold')

plt.show()

# create csv file for edge list
bipartite_edges = nx.to_pandas_edgelist(B)
bipartite_edges.to_csv("bipartite_students_courses.csv", index=False)
print(bipartite_edges)
