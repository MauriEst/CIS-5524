import pandas as pd
from itertools import combinations

"""
This script reads the data from the CSV file and creates an edge list of students who 
share at least one skill.
"""

csv_file = 'Assignment_1_Data.csv'
data = pd.read_csv(csv_file)

data.columns = [
    "Student Number",
    "Background in Data Mining/ML",
    "Background in Python/R",
    "Background in Graphs/Statistics",
]

binary_data = data.copy()
binary_data.replace({'Yes': 1, 'No': 0}, inplace=True)

edge_list = []
students = binary_data['Student Number']
skills = binary_data.iloc[:, 1:]

for student1, student2 in combinations(range(len(students)), 2):
    shared_skills = (skills.iloc[student1] & skills.iloc[student2]).sum()
    if shared_skills > 0:
        edge_list.append((students.iloc[student1], students.iloc[student2], shared_skills))

edge_list_df = pd.DataFrame(edge_list, columns=['Source', 'Target', 'Weight'])
num_edges = len(edge_list_df)
num_nodes = len(students)

output_file = 'EdgeList.csv'
with open(output_file, 'w') as f:
    f.write(f'# Nodes: {num_nodes} Edges: {num_edges}\n')
edge_list_df.to_csv(output_file, mode='a', index=False)

print(f'Edge list saved to {output_file}')