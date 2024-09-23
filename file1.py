import networkx as nx
import matplotlib.pyplot as plt
import community

# Sample data for nodes and edges
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
edges = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('C', 'F'), ('C', 'G'), ('D', 'H'), ('E', 'I'), ('F', 'J'), ('G', 'K')]

# Create a graph object
graph = nx.Graph()

# Add nodes and edges to the graph
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)

# Perform node clustering using the Louvain algorithm
partition = community.best_partition(graph)

# Create a dictionary to store node colors based on their cluster
node_colors = [partition.get(node) for node in graph.nodes()]

# Visualize the graph with node clustering
plt.figure(figsize=(8, 6))
nx.draw(graph, with_labels=True, node_color=node_colors, cmap=plt.cm.RdYlBu)
plt.title("Complex Network Graph with Node Clustering")
plt.show()
