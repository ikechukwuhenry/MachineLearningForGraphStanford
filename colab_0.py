import networkx as nx

# importing matplotlib.pyplot
import matplotlib.pyplot as plt

# Create an undirected graph G
G = nx.Graph()
print(G.is_directed())

# Create a directed graph H
H = nx.DiGraph()
print(H.is_directed())

# Add graph level attribute
G.graph["Name"] = "Bar"
print(G.graph)


G.nodes(data=True)


# Add multiple nodes with attribute
G.add_nodes_from([
    (1, {"feature": 1, "label": 1}),
    (2, {"feature": 2, "label": 2})
]
) # (node, attrdict)

# Loop through all the nodes
# Set data=True will return node attributes
for node in G.nodes(data=True):
    print(node)

# Get number of nodes
num_nodes = G.number_of_nodes()
print("G has {} nodes".format(num_nodes))

# Add one edge with edge weight 0.5
G.add_edge(0, 1, weight=0.5)

# Get attributes of the edge (0,1)
edge_0_1_attr = G.edges[(0,1)]
print("Edge (0,1) has the attributes {}".format(edge_0_1_attr))

# Add multiple edges with edge weights
G.add_edges_from([
    (1, 2, {"weight": 0.3}),
    (2, 0, {"weight": 0.1})
])

# Get number of edges
num_edges = G.number_of_edges()
print("G has {} edges".format(num_edges))

# Draw the graph
nx.draw(G, with_labels = True)
plt.show()

node_id = 1

# Degree of node 1
print("Node {} has degree {}".format(node_id, G.degree[node_id]))

# Get neighbor of node 1
for neighbor in G.neighbors(node_id):
    print("Node {} has {}".format(node_id, neighbor))

nx.path_graph(num_edges)

num_nodes = 4
# Create a new path like graph and change it to a directed graph
G = nx.DiGraph(nx.path_graph(num_nodes))
nx.draw(G, with_labels = True)
plt.show()

# Get the PageRank
pr = nx.pagerank(G, alpha=0.8)