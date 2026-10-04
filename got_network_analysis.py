import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
from networkx.algorithms.community import greedy_modularity_communities, modularity

# Step 2: Load local CSV
fname = "stormofswords.csv"
df = pd.read_csv(fname)
df.columns = [c.strip().lower() for c in df.columns]
print(f"Loaded {fname} with columns: {list(df.columns)}")

# Step 3: Build graph
G = nx.from_pandas_edgelist(
    df,
    source="source",
    target="target",
    edge_attr="weight" if "weight" in df.columns else None
)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Connected:", nx.is_connected(G))

def find(name):
    return [n for n in G.nodes() if name.lower() in str(n).lower()]

jon_matches = find("jon")
cersei_matches = find("cersei")
print("Found Jon matches:", jon_matches)
print("Found Cersei matches:", cersei_matches)

# Set node names based on dataset
start = jon_matches[0] if jon_matches else "Jon"
target = cersei_matches[0] if cersei_matches else "Cersei"
print(f"Using start='{start}' and target='{target}'")

# Step 5: Graph traversal
t0 = time.perf_counter()
bfs_order = list(nx.bfs_tree(G, source=start).nodes())
bfs_time = time.perf_counter() - t0

t0 = time.perf_counter()
dfs_order = list(nx.dfs_preorder_nodes(G, source=start))
dfs_time = time.perf_counter() - t0

print("BFS first 15:", bfs_order[:15])
print("DFS first 15:", dfs_order[:15])
print(f"BFS time: {bfs_time:.6f}s | DFS time: {dfs_time:.6f}s")

layers = dict(enumerate(nx.bfs_layers(G, start)))
for depth, nodes in layers.items():
    print(f"Depth {depth}: {len(nodes)} nodes")

path = nx.shortest_path(G, start, target)
print("Shortest path:", path, "| length:", len(path) - 1)

for u, v, d in G.edges(data=True):
    d["distance"] = 1 / d.get("weight", 1)
wpath = nx.dijkstra_path(G, start, target, weight="distance")
print("Weighted (Dijkstra) path:", wpath)

# Step 6: Graph measures
GC = G.subgraph(max(nx.connected_components(G), key=len))
print("Density:", nx.density(G))
print("Diameter:", nx.diameter(GC))
print("Avg path length:", nx.average_shortest_path_length(GC))
print("Avg clustering:", nx.average_clustering(G))

deg = nx.degree_centrality(G)
btw = nx.betweenness_centrality(G)
clo = nx.closeness_centrality(G)

metrics = pd.DataFrame({"degree": deg, "betweenness": btw, "closeness": clo})
print("\nTop 10 Nodes by Betweenness Centrality:")
print(metrics.sort_values("betweenness", ascending=False).head(10))

# Step 7: Community detection
comms = greedy_modularity_communities(G)
print("\nCommunities:", len(comms), "| Modularity:", modularity(G, comms))

# Step 8: Plot graph
node_comm = {n: i for i, c in enumerate(comms) for n in c}
plt.figure(figsize=(12, 9))
pos = nx.spring_layout(G, seed=42, k=0.3)
nx.draw_networkx_nodes(G, pos, node_size=[3000 * btw[n] + 30 for n in G],
                       node_color=[node_comm[n] for n in G], cmap="tab10")
nx.draw_networkx_edges(G, pos, alpha=0.2)
nx.draw_networkx_labels(G, pos, labels={n: n for n in G if btw[n] > 0.03}, font_size=8)
plt.axis("off")
plt.title("NetworkX: size = betweenness, color = community")
plt.savefig("networkx_plot.png", dpi=200, bbox_inches="tight")
plt.close()

# Step 9: Export files locally
nx.set_node_attributes(G, deg, "nx_degree")
nx.set_node_attributes(G, btw, "nx_betweenness")
nx.set_node_attributes(G, clo, "nx_closeness")
nx.set_node_attributes(G, node_comm, "nx_community")
for depth, nodes in layers.items():
    for n in nodes:
        G.nodes[n]["bfs_depth"] = depth

nx.write_gexf(G, "got_network.gexf")
metrics.to_csv("networkx_metrics.csv")
print("\nSuccessfully saved outputs locally:")
print(" - got_network.gexf")
print(" - networkx_metrics.csv")
print(" - networkx_plot.png")
