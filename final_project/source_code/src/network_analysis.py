import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
# Hàm 1: Vẽ đồ thị mạng HVG từ file CSV
def draw_network_graph(csv_path, output_path=OUTPUT_DIR / 'graph_network.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()

    # Thêm các cạnh vào đồ thị với trọng số
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']), weight=row['weight'])

    pos = nx.spring_layout(G, seed=42)  # Tùy chọn bố trí
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', edge_color='gray', arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{w:.2f}" for (u, v), w in edge_labels.items()})
    plt.title("Network Graph (HVG)")
    plt.savefig(output_path)
    plt.close()

# Hàm 2: Vẽ đồ thị dao động theo thời gian
def plot_daodong_timeseries(csv_path, output_path=OUTPUT_DIR / 'graph_daodong.png'):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df['node'], df['daodong'], marker='o', linestyle='-', color='blue')
    plt.xlabel('Ngày giao dịch (node)')
    plt.ylabel('Dao động')
    plt.title('Biểu đồ dao động theo thời gian')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

# Hàm 3: Vẽ histogram trọng số cạnh
def plot_weight_histogram(csv_path, output_path=OUTPUT_DIR / 'hist_weight.png'):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.hist(df['weight'], bins=30, color='green', edgecolor='black')
    plt.title('Phân bố trọng số cạnh')
    plt.xlabel('Trọng số')
    plt.ylabel('Số lượng')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

# Hàm 4: Vẽ phân bố bậc vào và bậc ra
def plot_degree_distribution(csv_path, output_path=OUTPUT_DIR / 'degree_distribution.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']))
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    plt.figure(figsize=(10, 5))
    plt.hist(in_degrees, bins=range(max(in_degrees)+1), alpha=0.6, label='Bậc vào', color='blue')
    plt.hist(out_degrees, bins=range(max(out_degrees)+1), alpha=0.6, label='Bậc ra', color='orange')
    plt.legend()
    plt.title('Phân bố bậc vào và bậc ra')
    plt.xlabel('Bậc')
    plt.ylabel('Số lượng node')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

# Hàm 5: Vẽ subgraph khuếch tán từ một node
def plot_diffusion_subgraph(csv_path, source_node, output_path=OUTPUT_DIR / 'diffusion_from_node.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']), weight=row['weight'])

    # Lấy các node reachable từ source_node
    descendants = nx.descendants(G, source_node)
    sub_nodes = list(descendants) + [source_node]
    SG = G.subgraph(sub_nodes)

    pos = nx.spring_layout(SG, seed=42)
    plt.figure(figsize=(10, 7))
    nx.draw(SG, pos, with_labels=True, node_color='lightcoral', edge_color='gray', node_size=400, arrows=True)
    edge_labels = nx.get_edge_attributes(SG, 'weight')
    nx.draw_networkx_edge_labels(SG, pos, edge_labels={(u, v): f"{w:.2f}" for (u, v), w in edge_labels.items()})
    plt.title(f"Lan truyền từ node {source_node}")
    plt.savefig(output_path)
    plt.close()

def plot_shortest_path_distribution(csv_path, output_path=OUTPUT_DIR / 'shortest_path_distribution.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']), weight=row['weight'])

    # Tính shortest path length (không dùng trọng số để tính theo số bước)
    path_lengths = []
    for source in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, source)
        for target, length in lengths.items():
            if source != target:
                path_lengths.append(length)

    # Vẽ histogram độ dài đường đi ngắn nhất
    plt.figure(figsize=(10, 6))
    plt.hist(path_lengths, bins=range(1, max(path_lengths)+2), color='purple', edgecolor='black', align='left')
    plt.title('Phân bố độ dài đường đi ngắn nhất (Shortest Path Lengths)')
    plt.xlabel('Số bước')
    plt.ylabel('Số lượng cặp node')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


# Hàm 7: Vẽ tất cả các đường đi độc lập (không trùng cạnh) từ source đến target
def plot_edge_disjoint_paths(csv_path, source, target, output_path=OUTPUT_DIR / 'edge_disjoint_paths.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']), weight=row['weight'])

    # Tìm tất cả các đường đi edge-disjoint từ source đến target
    paths = list(nx.edge_disjoint_paths(G, source, target))

    if not paths:
        print(f"Không tồn tại đường đi độc lập từ {source} đến {target}.")
        return

    # Vẽ subgraph chứa tất cả các đường đi độc lập
    path_edges = set()
    for path in paths:
        path_edges.update(zip(path[:-1], path[1:]))

    SG = nx.DiGraph()
    SG.add_edges_from(path_edges)

    pos = nx.spring_layout(SG, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(SG, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=600, arrows=True)

    # Tô màu các đường đi khác nhau
    colors = ['red', 'blue', 'orange', 'purple', 'green', 'black']
    for i, path in enumerate(paths):
        edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(SG, pos, edgelist=edges, edge_color=colors[i % len(colors)], width=2, arrows=True)

    plt.title(f"Tất cả đường đi edge-disjoint từ {source} đến {target} (số lượng: {len(paths)})")
    plt.savefig(output_path)
    plt.close()

# Hàm 8.1: Vẽ biểu đồ phân bố kích thước các SCC
def plot_scc_size_distribution(csv_path, output_path=OUTPUT_DIR / 'scc_size_distribution.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']))

    sccs = list(nx.strongly_connected_components(G))
    scc_sizes = [len(scc) for scc in sccs]

    plt.figure(figsize=(10, 6))
    plt.hist(scc_sizes, bins=range(1, max(scc_sizes)+2), edgecolor='black', color='teal', align='left')
    plt.title('Phân bố kích thước các thành phần liên thông mạnh (SCC)')
    plt.xlabel('Kích thước SCC')
    plt.ylabel('Số lượng SCC')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

# Hàm 8.2: Vẽ subgraph SCC lớn nhất
def plot_largest_scc_subgraph(csv_path, output_path=OUTPUT_DIR / 'largest_scc_subgraph.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']), weight=row['weight'])

    sccs = list(nx.strongly_connected_components(G))
    largest_scc = max(sccs, key=len)

    SG = G.subgraph(largest_scc).copy()

    pos = nx.spring_layout(SG, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(SG, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=600, arrows=True)
    edge_labels = nx.get_edge_attributes(SG, 'weight')
    nx.draw_networkx_edge_labels(SG, pos, edge_labels={(u, v): f"{w:.2f}" for (u, v), w in edge_labels.items()})
    plt.title(f"Thành phần liên thông mạnh lớn nhất (kích thước: {len(largest_scc)})")
    plt.savefig(output_path)
    plt.close()


# Hàm 9.1: Tìm và in tập cắt cạnh (edge cut set) giữa 2 node
def plot_edge_cut(csv_path, source, target, output_path=OUTPUT_DIR / 'edge_cut.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']), weight=row['weight'])

    try:
        cut_edges = nx.minimum_edge_cut(G, source, target)
        if not cut_edges:
            print(f"Không có tập cắt cạnh giữa {source} và {target}.")
            return

        # Vẽ toàn bộ mạng, highlight các cạnh bị cắt
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightgray', edge_color='gray', arrows=True)

        # Tô đỏ tập cắt
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='red', width=2)

        plt.title(f"Tập cắt cạnh (edge cut) giữa {source} và {target}")
        plt.savefig(output_path)
        plt.close()
    except nx.NetworkXError as e:
        print(f"Lỗi: {e}")

# Hàm 9.2: Tìm và in tập cắt node (node cut set) giữa 2 node
def plot_node_cut(csv_path, source, target, output_path=OUTPUT_DIR / 'node_cut.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']), weight=row['weight'])

    try:
        cut_nodes = nx.minimum_node_cut(G, source, target)
        if not cut_nodes:
            print(f"Không có tập cắt node giữa {source} và {target}.")
            return

        # Vẽ toàn bộ mạng, highlight các node bị cắt
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(12, 8))
        node_colors = ['red' if node in cut_nodes else 'lightgray' for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, edge_color='gray', arrows=True)

        plt.title(f"Tập cắt node (node cut) giữa {source} và {target}")
        plt.savefig(output_path)
        plt.close()
    except nx.NetworkXError as e:
        print(f"Lỗi: {e}")

# Hàm 10.x: Phân tích 3.7 -> 3.13 các chỉ số trung tâm và hệ số mạng

# 3.7 Degree Centrality
def plot_degree_centrality(csv_path, output_path=OUTPUT_DIR / 'degree_centrality.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']))
    
    centrality = nx.degree_centrality(G)
    top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    nodes, scores = zip(*top)

    plt.figure(figsize=(10, 6))
    plt.bar(nodes, scores, color='coral')
    plt.title('Top 10 Node có Degree Centrality cao nhất')
    plt.xlabel('Node')
    plt.ylabel('Degree Centrality')
    plt.savefig(output_path)
    plt.close()


# 3.8 Eigenvector Centrality
def plot_eigenvector_centrality(csv_path, output_path=OUTPUT_DIR / 'eigenvector_centrality.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']))

    try:
        centrality = nx.eigenvector_centrality(G, max_iter=1000)
        top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        nodes, scores = zip(*top)

        plt.figure(figsize=(10, 6))
        plt.bar(nodes, scores, color='skyblue')
        plt.title('Top 10 Node có Eigenvector Centrality cao nhất')
        plt.xlabel('Node')
        plt.ylabel('Eigenvector Centrality')
        plt.savefig(output_path)
        plt.close()
    except nx.NetworkXException as e:
        print(f"Lỗi tính eigenvector centrality: {e}")


# 3.9 Closeness Centrality
def plot_closeness_centrality(csv_path, output_path=OUTPUT_DIR / 'closeness_centrality.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']))

    centrality = nx.closeness_centrality(G)
    top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    nodes, scores = zip(*top)

    plt.figure(figsize=(10, 6))
    plt.bar(nodes, scores, color='green')
    plt.title('Top 10 Node có Closeness Centrality cao nhất')
    plt.xlabel('Node')
    plt.ylabel('Closeness Centrality')
    plt.savefig(output_path)
    plt.close()


# 3.10 Betweenness Centrality
def plot_betweenness_centrality(csv_path, output_path=OUTPUT_DIR / 'betweenness_centrality.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']))

    centrality = nx.betweenness_centrality(G)
    top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    nodes, scores = zip(*top)

    plt.figure(figsize=(10, 6))
    plt.bar(nodes, scores, color='purple')
    plt.title('Top 10 Node có Betweenness Centrality cao nhất')
    plt.xlabel('Node')
    plt.ylabel('Betweenness Centrality')
    plt.savefig(output_path)
    plt.close()


# 3.11 K-core subgraph
def plot_k_core(csv_path, k=2, output_path=OUTPUT_DIR / 'k_core.png'):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']))

    core_subgraph = nx.k_core(G.to_undirected(), k=k)
    pos = nx.spring_layout(core_subgraph, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(core_subgraph, pos, with_labels=True, node_size=600, node_color='orange', edge_color='gray', arrows=False)
    plt.title(f'K-core Subgraph (k={k})')
    plt.savefig(output_path)
    plt.close()


# 3.12 Transitivity (tính bắc cầu) - hệ số toàn mạng
def compute_transitivity(csv_path):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']))

    trans = nx.transitivity(G.to_undirected())
    print(f"Hệ số bắc cầu (Transitivity) toàn mạng: {trans}")
    return trans


# 3.13 Clustering coefficient
def compute_clustering_coefficient(csv_path):
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node']), int(row['node_link']))

    clustering = nx.average_clustering(G.to_undirected())
    print(f"Hệ số phân cụm trung bình: {clustering}")
    return clustering


"""
The following example calls are commented. Use DATA_DIR/OUTPUT_DIR for paths.
"""
draw_network_graph(DATA_DIR / "data_final.csv", OUTPUT_DIR / "graph_network.png")
# → Vẽ và lưu đồ thị mạng có hướng & trọng số

plot_daodong_timeseries(DATA_DIR / "data_final.csv", OUTPUT_DIR / "graph_daodong.png")
# → Vẽ đồ thị dao động theo thời gian

plot_weight_histogram(DATA_DIR / "data_final.csv", OUTPUT_DIR / "hist_weight.png")
# → Vẽ histogram trọng số các cạnh

plot_degree_distribution(DATA_DIR / "data_final.csv", OUTPUT_DIR / "degree_distribution.png")
# → Vẽ phân bố bậc vào và bậc ra

plot_diffusion_subgraph(DATA_DIR / "data_final.csv", source_node=272, output_path=OUTPUT_DIR / "diffusion_from_node272.png")
# → Vẽ mạng khuếch tán từ node cụ thể (VD: node 272)

plot_shortest_path_distribution(DATA_DIR / "data_final.csv", OUTPUT_DIR / "shortest_path_distribution.png")
# → Vẽ phân bố độ dài đường đi ngắn nhất

plot_edge_disjoint_paths(DATA_DIR / "data_final.csv", source=1, target=100, output_path=OUTPUT_DIR / "edge_disjoint_paths_1_10.png")
# → Vẽ các đường đi không chung cạnh từ node 1 đến node 100

plot_scc_size_distribution(DATA_DIR / "data_final.csv", OUTPUT_DIR / "scc_size_distribution.png")
# → Vẽ phân bố kích thước các thành phần liên thông mạnh

plot_largest_scc_subgraph(DATA_DIR / "data_final.csv", OUTPUT_DIR / "largest_scc_subgraph.png")
# → Vẽ thành phần liên thông mạnh lớn nhất

plot_edge_cut(DATA_DIR / "data_final.csv", source=1, target=10, output_path=OUTPUT_DIR / "edge_cut_1_10.png")
# → Vẽ cắt cạnh từ node 1 đến node 10

plot_degree_centrality(DATA_DIR / "data_final.csv", OUTPUT_DIR / "degree_centrality.png")
# → Vẽ phân bố trung tâm bậc

plot_eigenvector_centrality(DATA_DIR / "data_final.csv", OUTPUT_DIR / "eigenvector_centrality.png")
# → Vẽ phân bố trung tâm riêng

plot_closeness_centrality(DATA_DIR / "data_final.csv", OUTPUT_DIR / "closeness_centrality.png")
# → Vẽ phân bố trung tâm gần gũi

plot_betweenness_centrality(DATA_DIR / "data_final.csv", OUTPUT_DIR / "betweenness_centrality.png")
# → Vẽ phân bố trung tâm trung gian

plot_k_core(DATA_DIR / "data_final.csv", k=2, output_path=OUTPUT_DIR / "k_core.png")
# → Vẽ phân bố k-core

transitivity = compute_transitivity(DATA_DIR / "data_final.csv")  # In giá trị transitivity
clustering = compute_clustering_coefficient(DATA_DIR / "data_final.csv")  # In hệ số phân cụm
