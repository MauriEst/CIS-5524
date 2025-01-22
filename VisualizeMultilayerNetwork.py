import csv
import pymnet
from pymnet import *

graph = pymnet.MultilayerNetwork(aspects=1)

with open('EdgeList.csv', 'r') as f:
    next(f)
    reader = csv.DictReader(f)
    for row in reader:
        source, target, weight = row['Source'], row['Target'], int(row['Weight'])

        # Define thresholds for assigning edges to layers
        if weight == 1:
            graph[source, target, 'Data Mining/ML'] = 1
        elif weight == 2:
            graph[source, target, 'Python/R'] = 1
        elif weight == 3:
            graph[source, target, 'Graphs/Statistics'] = 1

        # graph[source, target, 'Data Mining/ML'] = int(weight)
        # graph[source, target, 'Python/R'] = int(weight)
        # graph[source, target, 'Graphs/Statistics'] = int(weight)

fig = pymnet.draw(graph, figsize=(10, 8), #layershape='circle', 
            # nodeLabelRule={},
            # layerLabelRule={},
            # layout='circular',
            # nodeColorDict={(0,0):'r', (1,0): 'b', (0,1):'g'},
            # nodeSizeRule={'rule': 'degree', 'propscale': 0.5},
            show=True)

fig.savefig('net.pdf')

