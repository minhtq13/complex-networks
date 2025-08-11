import pandas as pd
import networkx as nx
import numpy as np
from itertools import permutations
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Đọc dữ liệu từ file CSV
df = pd.read_csv(DATA_DIR / 'data_final.csv')

# Hiển thị một số dòng đầu tiên để kiểm tra
print("Dữ liệu đầu vào (5 dòng đầu):")
print(df.head())

# Tạo đồ thị có hướng
G = nx.DiGraph()

# Thêm các cạnh từ dữ liệu
for _, row in df.iterrows():
    G.add_edge(int(row['node']), int(row['node_link']), weight=row['weight'])

print("\n--- THÔNG TIN ĐỒ THỊ ---")
print(f"Số nút: {G.number_of_nodes()}")
print(f"Số cạnh: {G.number_of_edges()}")

# 1. Tính Geodesic path (đường đi ngắn nhất)
print("\n--- GEODESIC PATHS ---")

# Chọn một số cặp nút để hiển thị đường đi ngắn nhất
sample_nodes = np.random.choice(list(G.nodes()), min(5, len(G.nodes())), replace=False)
sample_pairs = [(sample_nodes[i], sample_nodes[j])
                for i in range(len(sample_nodes))
                for j in range(i + 1, len(sample_nodes))]

print(f"Hiển thị đường đi ngắn nhất cho {len(sample_pairs)} cặp nút mẫu:")

for source, target in sample_pairs:
    try:
        path = nx.shortest_path(G, source=source, target=target, weight='weight')
        length = nx.shortest_path_length(G, source=source, target=target, weight='weight')
        print(f"Đường đi ngắn nhất từ {source} đến {target}: {path}")
        print(f"  Độ dài: {length:.4f}")
    except nx.NetworkXNoPath:
        print(f"Không có đường đi từ {source} đến {target}")

# Minh họa: vẽ ví dụ một đường đi ngắn nhất (nếu tìm được)
def plot_shortest_path_example(graph: nx.DiGraph, path_nodes, save_path: Path):
    if not path_nodes or len(path_nodes) < 2:
        return
    sub_nodes = set(path_nodes)
    SG = graph.subgraph(sub_nodes).copy()
    pos = nx.spring_layout(SG, seed=42)
    plt.figure(figsize=(10, 7))
    nx.draw(SG, pos, with_labels=True, node_color='lightgray', edge_color='gray', node_size=500, arrows=True)
    path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
    nx.draw_networkx_edges(SG, pos, edgelist=path_edges, edge_color='red', width=2.5, arrows=True)
    plt.title('Ví dụ đường đi ngắn nhất (tô đỏ)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu ví dụ shortest path: {save_path}")

# Tìm một cặp có đường đi để vẽ
shortest_example_saved = False
for source, target in sample_pairs:
    try:
        example_path = nx.shortest_path(G, source=source, target=target, weight='weight')
        plot_shortest_path_example(G, example_path, OUTPUT_DIR / 'shortest_path_example.png')
        shortest_example_saved = True
        break
    except nx.NetworkXNoPath:
        continue

# Minh họa: histogram độ dài đường đi ngắn nhất (trọng số) theo mẫu nguồn
def plot_weighted_shortest_path_hist(graph: nx.DiGraph, k_sources: int = 20, save_path: Path = OUTPUT_DIR / 'shortest_path_weighted_hist.png'):
    nodes = list(graph.nodes())
    if not nodes:
        return
    k = min(k_sources, len(nodes))
    sources = list(np.random.choice(nodes, k, replace=False))
    lengths = []
    for s in sources:
        dist = nx.single_source_dijkstra_path_length(graph, source=s, weight='weight')
        # loại bỏ tự thân và vô cực
        for t, d in dist.items():
            if t != s and np.isfinite(d):
                lengths.append(d)
    if not lengths:
        print("Không thu được độ dài đường đi (có thể đồ thị quá rời)")
        return
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, color='purple', edgecolor='black')
    plt.xlabel('Độ dài (trọng số)')
    plt.ylabel('Số lượng cặp')
    plt.title('Phân bố độ dài đường đi ngắn nhất (có trọng số, mẫu nguồn)')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu histogram shortest path (weighted): {save_path}")

plot_weighted_shortest_path_hist(G)

# Tính đường đi ngắn nhất trung bình
try:
    avg_path = nx.average_shortest_path_length(G, weight='weight')
    print(f"\nĐộ dài đường đi ngắn nhất trung bình: {avg_path:.4f}")
except nx.NetworkXError:
    print("\nĐồ thị không liên thông, không thể tính độ dài đường đi trung bình")
    # Tính cho các thành phần liên thông mạnh
    components = list(nx.strongly_connected_components(G))
    print(f"Số thành phần liên thông mạnh: {len(components)}")

    if len(components) > 0:
        largest_cc = max(components, key=len)
        print(f"Kích thước thành phần liên thông mạnh lớn nhất: {len(largest_cc)}")
        if len(largest_cc) > 1:
            largest_subgraph = G.subgraph(largest_cc).copy()
            avg_path_largest = nx.average_shortest_path_length(largest_subgraph, weight='weight')
            print(f"Độ dài đường đi ngắn nhất trung bình trong thành phần lớn nhất: {avg_path_largest:.4f}")

# Minh họa: thành phần liên thông mạnh lớn nhất
def plot_largest_scc(graph: nx.DiGraph, save_path: Path = OUTPUT_DIR / 'largest_scc_subgraph.png'):
    sccs = list(nx.strongly_connected_components(graph))
    if not sccs:
        return
    largest_scc = max(sccs, key=len)
    if len(largest_scc) == 0:
        return
    SG = graph.subgraph(largest_scc).copy()
    pos = nx.spring_layout(SG, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(SG, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=600, arrows=True)
    edge_labels = nx.get_edge_attributes(SG, 'weight')
    nx.draw_networkx_edge_labels(SG, pos, edge_labels={(u, v): f"{w:.2f}" for (u, v), w in edge_labels.items()}, font_size=8)
    plt.title(f"SCC lớn nhất (kích thước: {len(largest_scc)})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu largest SCC: {save_path}")

plot_largest_scc(G)

# 2. Kiểm tra và tìm Hamilton path
print("\n--- HAMILTON PATH ---")


# Hàm kiểm tra một thứ tự các nút có tạo thành Hamilton path không
def is_hamilton_path(graph, path):
    if len(path) != len(graph.nodes()):
        return False

    for i in range(len(path) - 1):
        if not graph.has_edge(path[i], path[i + 1]):
            return False
    return True


# Đối với đồ thị lớn, chúng ta chỉ kiểm tra Hamilton path cho một phần nhỏ
def find_hamilton_path_sample(graph, max_nodes=10):
    if len(graph) <= max_nodes:
        nodes = list(graph.nodes())
        for path in permutations(nodes):
            if is_hamilton_path(graph, path):
                return list(path)
        return None
    else:
        print(f"Đồ thị quá lớn ({len(graph)} nút) để kiểm tra đầy đủ Hamilton path")
        print("Thử kiểm tra trên một tập con...")

        # Lấy một tập con nhỏ của đồ thị
        sample_nodes = list(graph.nodes())[:max_nodes]
        subgraph = graph.subgraph(sample_nodes).copy()

        for path in permutations(sample_nodes):
            if is_hamilton_path(subgraph, path):
                return list(path)
        return None


# Nếu đồ thị quá lớn, chỉ tìm Hamilton path trên thành phần liên thông lớn nhất
hamilton_path_result = None
hamilton_subgraph = None

if len(G) > 20:  # Ngưỡng kích thước đồ thị để áp dụng thuật toán đầy đủ
    components = list(nx.strongly_connected_components(G))
    if len(components) > 0:
        largest_cc = max(components, key=len)
        if len(largest_cc) <= 10:  # Chỉ thử nếu thành phần liên thông < 10 nút
            print(f"Tìm Hamilton path trên thành phần liên thông mạnh lớn nhất ({len(largest_cc)} nút)")
            subgraph = G.subgraph(largest_cc).copy()
            hamilton_path = find_hamilton_path_sample(subgraph)
            if hamilton_path:
                print(f"Tìm thấy Hamilton path trong thành phần liên thông: {hamilton_path}")
                hamilton_path_result = hamilton_path
                hamilton_subgraph = subgraph
            else:
                print("Không tìm thấy Hamilton path trong thành phần liên thông lớn nhất")
        else:
            print(
                f"Thành phần liên thông mạnh lớn nhất ({len(largest_cc)} nút) quá lớn để kiểm tra đầy đủ Hamilton path")
            print("Thử một phương pháp xấp xỉ bằng cách lấy mẫu...")

            # Sử dụng thuật toán xấp xỉ hoặc heuristic ở đây
            sample_nodes = list(largest_cc)[:8]  # Lấy 8 nút đầu tiên để thử
            sub = G.subgraph(sample_nodes).copy()
            sample_path = find_hamilton_path_sample(sub)
            if sample_path:
                print(f"Tìm thấy Hamilton path trên tập con 8 nút: {sample_path}")
                hamilton_path_result = sample_path
                hamilton_subgraph = sub
            else:
                print("Không tìm thấy Hamilton path trên tập con")
else:
    hamilton_path = find_hamilton_path_sample(G)
    if hamilton_path:
        print(f"Tìm thấy Hamilton path: {hamilton_path}")
        hamilton_path_result = hamilton_path
        hamilton_subgraph = G
    else:
        print("Không tìm thấy Hamilton path trong đồ thị")

# Thuật toán hiệu quả hơn để phát hiện Hamilton path
print("\nSử dụng phương pháp hiệu quả hơn để kiểm tra khả năng tồn tại Hamilton path:")

# Điều kiện cần để tồn tại Hamilton path
is_strongly_connected = nx.is_strongly_connected(G)
print(f"Đồ thị có liên thông mạnh không?: {is_strongly_connected}")

if not is_strongly_connected:
    print("Đồ thị không liên thông mạnh, nên không thể có Hamilton path cho toàn bộ đồ thị")

    # Kiểm tra các thành phần liên thông mạnh
    components = list(nx.strongly_connected_components(G))
    print(f"Số thành phần liên thông mạnh: {len(components)}")
    print(f"Kích thước các thành phần lớn nhất: {sorted([len(c) for c in components], reverse=True)[:5]}")

    # Nếu có một thành phần liên thông mạnh đủ lớn
    if any(len(c) > len(G.nodes()) * 0.8 for c in components):
        print("Có một thành phần liên thông mạnh lớn, có thể tồn tại Hamilton path trong thành phần này")
else:
    # Kiểm tra điều kiện Dirac: Nếu đồ thị có n nút và mỗi nút có bậc ≥ n/2
    n = G.number_of_nodes()
    min_degree = min(dict(G.degree()).values())

    if min_degree >= n / 2:
        print(f"Thỏa mãn điều kiện Dirac (mỗi nút có bậc ≥ {n / 2}), có thể tồn tại Hamilton path")
    else:
        print(f"Không thỏa mãn điều kiện Dirac, bậc nhỏ nhất = {min_degree} < {n / 2}")
        print("Tuy nhiên, đây chỉ là điều kiện đủ, đồ thị vẫn có thể có Hamilton path")

# Minh họa: nếu tìm thấy Hamilton path, vẽ subgraph và tô đỏ đường đi
def plot_hamilton_path(subgraph: nx.DiGraph, path_nodes, save_path: Path = OUTPUT_DIR / 'hamilton_path_example.png'):
    if subgraph is None or not path_nodes or len(path_nodes) < 2:
        return
    pos = nx.spring_layout(subgraph, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(subgraph, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=600, arrows=True)
    path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
    nx.draw_networkx_edges(subgraph, pos, edgelist=path_edges, edge_color='red', width=2.5, arrows=True)
    plt.title('Minh họa Hamilton path (nếu tồn tại)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu Hamilton path example: {save_path}")

if hamilton_path_result is not None:
    plot_hamilton_path(hamilton_subgraph, hamilton_path_result)