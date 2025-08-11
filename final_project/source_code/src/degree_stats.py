import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

# Đọc dữ liệu từ file CSV
df = pd.read_csv(DATA_DIR / 'data_final.csv')

# Tạo đồ thị có hướng
G = nx.DiGraph()

# Thêm các cạnh từ dữ liệu
for _, row in df.iterrows():
    G.add_edge(int(row['node']), int(row['node_link']), weight=row['weight'])

# Tính các chỉ số về bậc
in_degrees = [d for n, d in G.in_degree()]
out_degrees = [d for n, d in G.out_degree()]
total_degrees = [d for n, d in G.degree()]

# Tính các thống kê
min_in_degree = min(in_degrees)
max_in_degree = max(in_degrees)
min_out_degree = min(out_degrees)
max_out_degree = max(out_degrees)
median_total_degree = np.median(total_degrees)

# In kết quả
print(f"Thống kê về bậc của đồ thị:")
print(f"---------------------------")
print(f"Bậc vào nhỏ nhất: {min_in_degree}")
print(f"Bậc vào lớn nhất: {max_in_degree}")
print(f"Bậc ra nhỏ nhất: {min_out_degree}")
print(f"Bậc ra lớn nhất: {max_out_degree}")
# print(f"Bậc vào ra (tổng) trung vị: {median_total_degree}")

# Thông tin bổ sung về đồ thị
print(f"\nThông tin bổ sung:")
print(f"---------------------------")
print(f"Số đỉnh: {G.number_of_nodes()}")
print(f"Số cạnh: {G.number_of_edges()}")

# Danh sách các đỉnh có bậc vào nhỏ nhất
min_in_nodes = [n for n, d in G.in_degree() if d == min_in_degree]
print(f"\nCác đỉnh có bậc vào nhỏ nhất ({min_in_degree}): {min_in_nodes}")

# Danh sách các đỉnh có bậc vào lớn nhất
max_in_nodes = [n for n, d in G.in_degree() if d == max_in_degree]
print(f"Các đỉnh có bậc vào lớn nhất ({max_in_degree}): {max_in_nodes}")

# Danh sách các đỉnh có bậc ra nhỏ nhất
min_out_nodes = [n for n, d in G.out_degree() if d == min_out_degree]
print(f"Các đỉnh có bậc ra nhỏ nhất ({min_out_degree}): {min_out_nodes[:10]}{'...' if len(min_out_nodes) > 10 else ''}")

# Danh sách các đỉnh có bậc ra lớn nhất
max_out_nodes = [n for n, d in G.out_degree() if d == max_out_degree]
print(f"Các đỉnh có bậc ra lớn nhất ({max_out_degree}): {max_out_nodes}")

# Minh họa bằng đồ thị: phân bố bậc và top-10 node
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Histogram phân bố bậc vào/ra
plt.figure(figsize=(10, 5))
bins_in = range(0, max(in_degrees) + 2)
bins_out = range(0, max(out_degrees) + 2)
plt.hist(in_degrees, bins=bins_in, alpha=0.6, label='Bậc vào', color='steelblue', edgecolor='black')
plt.hist(out_degrees, bins=bins_out, alpha=0.6, label='Bậc ra', color='orange', edgecolor='black')
plt.xlabel('Bậc')
plt.ylabel('Số lượng node')
plt.title('Phân bố bậc vào/ra')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
dist_path = OUTPUT_DIR / 'degree_distribution_stats.png'
plt.tight_layout()
plt.savefig(dist_path, dpi=300)
plt.close()
print(f"✅ Đã lưu biểu đồ phân bố bậc: {dist_path}")

# 2) Bar chart top-10 bậc vào/ra
in_deg_dict = dict(G.in_degree())
out_deg_dict = dict(G.out_degree())
top_in = sorted(in_deg_dict.items(), key=lambda x: x[1], reverse=True)[:10]
top_out = sorted(out_deg_dict.items(), key=lambda x: x[1], reverse=True)[:10]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

if top_in:
    nodes_in, vals_in = zip(*top_in)
    axes[0].bar(range(len(nodes_in)), vals_in, color='steelblue')
    axes[0].set_xticks(range(len(nodes_in)))
    axes[0].set_xticklabels(nodes_in, rotation=45, ha='right', fontsize=8)
    axes[0].set_title('Top-10 bậc vào')
    axes[0].set_xlabel('Node')
    axes[0].set_ylabel('Bậc')

if top_out:
    nodes_out, vals_out = zip(*top_out)
    axes[1].bar(range(len(nodes_out)), vals_out, color='orange')
    axes[1].set_xticks(range(len(nodes_out)))
    axes[1].set_xticklabels(nodes_out, rotation=45, ha='right', fontsize=8)
    axes[1].set_title('Top-10 bậc ra')
    axes[1].set_xlabel('Node')

for ax in axes:
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

plt.tight_layout()
top_path = OUTPUT_DIR / 'top_degree_nodes.png'
plt.savefig(top_path, dpi=300)
plt.close()
print(f"✅ Đã lưu biểu đồ top-degree: {top_path}")