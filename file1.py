import networkx as nx
import matplotlib.pyplot as plt
import community

nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
edges = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('C', 'F'), ('C', 'G'), ('D', 'H'), ('E', 'I'), ('F', 'J'), ('G', 'K')]

graph = nx.Graph()

graph.add_nodes_from(nodes)
graph.add_edges_from(edges)

partition = community.best_partition(graph)

node_colors = [partition.get(node) for node in graph.nodes()]

plt.figure(figsize=(8, 6))
nx.draw(graph, with_labels=True, node_color=node_colors, cmap=plt.cm.RdYlBu)
plt.title("Complex Network Graph with Node Clustering")
plt.show()
