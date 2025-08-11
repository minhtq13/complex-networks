import datetime
import networkx as nx
from matplotlib import pyplot as plt
from scipy.integrate import quad
import pandas as pd
from pathlib import Path

# Resolve project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

def dientich(m,c,t,d):
    def f1(x):
        return (c-m)*x + (m-t)
    def f2(x):
        return (t-c)*x + (2*c-2*t)
    def f3(x):
        return (d-t)*x + (2*t-2*d)

    result_1, error = quad(f1, 0, 1)
    result_2, error = quad(f2, 1, 2)
    result_3, error = quad(f3, 2, 3)

    return round(result_1+ result_2 + result_3, 6)

def dientich_csv(file_path):
    df = pd.read_excel(file_path)
    kq = []
    for _, row in df.iterrows():
        m = row['open']
        c = row['high']
        t = row['low']
        d = row['close']

        area = dientich(m, c, t, d)

        date = row['time']
        if isinstance(date, datetime.datetime):
            formatted_date = f"{date.day}-{date.month}-{date.year}"
        else:
            formatted_date = date

        kq.append((formatted_date, area))
    return kq

def create_ptl():

    # Danh sách các điểm bạn có thể chỉnh sửa
    points = {
        'A': (0, 4),
        'B': (1, 4),
        'C': (2, 0.5),
        'D': (3, 4)
    }

    # Tách tọa độ và nhãn
    labels = list(points.keys())
    x_vals = [points[label][0] for label in labels]
    y_vals = [points[label][1] for label in labels]

    # Vẽ biểu đồ
    plt.figure(figsize=(6, 5))
    plt.plot(x_vals, y_vals, '-o', color='black', markerfacecolor='red', linewidth=2)

    # Vẽ đường thẳng đứng từ điểm xuống trục hoành
    for x, y in zip(x_vals, y_vals):
        plt.vlines(x, 0, y, colors='black', linestyles='solid', linewidth=1)

    # Ghi nhãn từng điểm
    for label, x, y in zip(labels, x_vals, y_vals):
        plt.text(x-0.25, y + 0.1, label, fontsize=12, ha='center')

    # Vẽ trục
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    # Hiện các ticks trục x (giống ảnh vẽ tay)
    plt.xticks(x_vals, [str(x) for x in x_vals], fontsize=12)
    plt.yticks([])  # Ẩn trục y nếu muốn gọn

    # Giới hạn và định dạng
    plt.xlim(min(x_vals) - 0.5, max(x_vals) + 0.5)
    plt.ylim(0, max(y_vals) + 1)
    plt.box(False)
    #plt.title("Biểu đồ với đường thẳng đứng từ điểm", fontsize=14)

    plt.tight_layout()
    plt.show()


def ve_do_thi_dien_tich(file_csv, x_limit):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(file_csv)

    # Lấy x_limit dòng đầu tiên
    df_plot = df.head(x_limit)

    # Vẽ đồ thị
    plt.figure(figsize=(14, 6))
    plt.plot(df_plot["date"], df_plot["daodong"], marker='o', linestyle='-', linewidth=2)
    plt.xticks(rotation=45)
    plt.xlabel("Ngày")
    plt.ylabel("Giá trị dao động (Diện tích)")
    plt.title(f"Biến động diện tích trong {x_limit} ngày đầu")
    plt.grid(True)
    plt.tight_layout()

    # Lưu ảnh
    image_path = OUTPUT_DIR / "do_thi_dien_tich.png"
    plt.savefig(image_path)
    plt.close()
    print(f"✅ Đã lưu ảnh đồ thị tại: {image_path}")


def ve_do_thi_dien_tich_cot(file_csv, x_limit):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(file_csv)

    # Lấy x_limit dòng đầu tiên
    df_plot = df.head(x_limit).reset_index(drop=True)

    # Tạo trục x là số thứ tự 1, 2, 3, ...
    x_vals = list(range(1, x_limit + 1))
    y_vals = df_plot["daodong"]

    # Vẽ đồ thị
    plt.figure(figsize=(14, 6))

    # Vẽ điểm
    plt.scatter(x_vals, y_vals, color='red', zorder=3)

    # Vẽ đường dóng xuống Ox
    for x, y in zip(x_vals, y_vals):
        plt.vlines(x, 0, y, colors='gray', linestyles='solid', linewidth=0.8, zorder=1)

    # Thiết lập trục
    plt.xticks(x_vals)
    plt.xlabel("Ngày (thứ tự)")
    plt.ylabel("Giá trị dao động (Diện tích)")
    plt.title(f"Biến động trong {x_limit} ngày đầu")

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Lưu ảnh
    image_path = OUTPUT_DIR / "do_thi_danh_so.png"
    plt.savefig(image_path, dpi=300)
    plt.close()
    print(f"✅ Đã lưu ảnh đồ thị tại: {image_path}")


def convert_date_to_sequence(input_file, output_file):
    """
    Đọc file CSV và chuyển đổi cột date thành chuỗi số từ 1 đến hết

    Args:
        input_file (str): Đường dẫn đến file CSV đầu vào
        output_file (str): Đường dẫn đến file CSV đầu ra
    """
    # Đọc file CSV
    df = pd.read_csv(input_file)

    # Tạo cột mới với giá trị từ 1 đến số lượng hàng
    df['node'] = range(1, len(df) + 1)

    # Tạo DataFrame mới chỉ với cột date_sequence và cột daodong
    new_df = df[['node', 'daodong']]

    # Lưu DataFrame mới ra file CSV
    new_df.to_csv(output_file, index=False)

    print(f"Đã chuyển đổi thành công và lưu kết quả vào file {output_file}")


def tao_mang_visibility(input_file, output_file):
    """
    Hàm tạo mạng Horizontal Visibility từ file CSV chứa cột 'daodong'.
    Đầu vào:
        input_file: đường dẫn đến file CSV đầu vào (có cột 'daodong')
        output_file: đường dẫn đến file CSV đầu ra chứa mạng
    Đầu ra:
        Lưu file CSV chứa các cột: node, daodong, node_link, daodong_node_link, weight
    """
    # Đọc file
    df = pd.read_csv(input_file)

    # Thêm cột node (số thứ tự)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'node'}, inplace=True)

    # Tạo danh sách cạnh theo nguyên tắc HVG
    edges = []
    for i in range(len(df)):
        y_i = df.loc[i, 'daodong']
        for j in range(i + 1, len(df)):
            y_j = df.loc[j, 'daodong']
            # Lấy các giá trị trung gian giữa i và j
            intermediate_values = df.loc[i + 1:j - 1, 'daodong']
            if all(intermediate_values < min(y_i, y_j)):
                weight = round((y_i * y_j)/(j-i), 6)
                edges.append({
                    'node': i,
                    'daodong': y_i,
                    'node_link': j,
                    'daodong_node_link': y_j,
                    'weight': weight
                })

    # Ghi ra file CSV kết quả
    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(output_file, index=False)
    print(f"✅ Đã tạo file mạng visibility tại: {output_file}")


def tao_mang_co_huong_trong_so(input_file):
    """
    Tạo mạng có hướng, có trọng số từ file CSV theo quy tắc visibility graph:
    Trọng số: (y_i * y_j) / (j - i)

    Parameters:
        input_file (str): Đường dẫn file CSV đã tạo từ visibility rule (có các cột:
                          node, daodong, node_link, daodong_node_link, weight ban đầu nếu có)

    Returns:
        G (networkx.DiGraph): Đồ thị đã xây dựng
    """
    df = pd.read_csv(input_file)

    # Tạo đồ thị có hướng
    G = nx.DiGraph()

    # Thêm node và các thuộc tính
    nodes = pd.concat([
        df[['node', 'daodong']].rename(columns={'node': 'id'}),
        df[['node_link', 'daodong_node_link']].rename(columns={'node_link': 'id', 'daodong_node_link': 'daodong'})
    ]).drop_duplicates(subset='id')

    for _, row in nodes.iterrows():
        G.add_node(row['id'], daodong=row['daodong'])

    # Thêm cạnh có hướng với trọng số mới
    for _, row in df.iterrows():
        i = row['node']
        j = row['node_link']
        y_i = row['daodong']
        y_j = row['daodong_node_link']
        weight = round((y_i * y_j) / (j - i), 6)
        G.add_edge(i, j, weight=weight)

    print(f"✅ Mạng đã tạo xong với {G.number_of_nodes()} node và {G.number_of_edges()} cạnh.")
    return G

def ve_mang(G, save_path=None ,title="Mạng"):
    """
    Hàm vẽ mạng có hướng với trọng số.

    Parameters:
        G (networkx.DiGraph): mạng đã tạo
        title (str): tiêu đề đồ thị
    """
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)  # định vị node

    # Vẽ node
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')

    # Vẽ cạnh có hướng
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)

    # Vẽ nhãn node
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Hiển thị trọng số cạnh
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_size=7)

    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=500)
        print(f"✅ Đã lưu ảnh mạng tại: {save_path}")
    plt.show()


def ve_mang_to_mau(G, save_path=None, title="Mạng theo mức độ ảnh hưởng"):
    """
    Vẽ mạng có hướng, bỏ trọng số, tô màu & chỉnh size theo out-degree.

    Parameters:
        G (networkx.DiGraph): mạng đã tạo
        save_path (str): đường dẫn lưu ảnh (nếu cần)
        title (str): tiêu đề đồ thị
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Tính out-degree
    out_deg = dict(G.out_degree())
    node_sizes = [200 + 60 * out_deg.get(n, 0) for n in G.nodes()]
    node_colors = [out_deg.get(n, 0) for n in G.nodes()]

    pos = nx.spring_layout(G, seed=42)

    # Vẽ node
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax,
                                   node_size=node_sizes,
                                   node_color=node_colors,
                                   cmap=plt.cm.viridis,
                                   alpha=0.85)

    # Vẽ cạnh và nhãn
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', edge_color='gray', alpha=0.6)
    # nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    labels = {n: str(int(n)) if float(n).is_integer() else str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)

    # Gắn colorbar vào figure
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])  # bắt buộc để tránh lỗi
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Out-degree (mức ảnh hưởng)')

    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=500)
        print(f"✅ Đã lưu ảnh mạng tại: {save_path}")
    plt.show()

G = tao_mang_co_huong_trong_so(str(DATA_DIR / "data_final.csv"))

ve_mang_to_mau(G, save_path=str(OUTPUT_DIR / "network_mau.png"))

# Pipeline tạo dữ liệu từ OHLCV -> daodong -> HVG (có kiểm tra tồn tại file)
# 1) Tạo file daodong thô từ Excel nếu có
xlsx_path = DATA_DIR / "gia_lich_su_ohlcv_ACB.xlsx"
raw_daodong_csv = DATA_DIR / "dien_tich_ACB.csv"
if xlsx_path.exists():
    kq = dientich_csv(xlsx_path)
    output_df = pd.DataFrame(kq, columns=["date", "daodong"])
    output_df.to_csv(raw_daodong_csv, index=False, encoding='utf-8-sig')
    print(f"✅ Đã lưu file: {raw_daodong_csv}")
else:
    if not raw_daodong_csv.exists():
        print(f"⚠️ Không tìm thấy dữ liệu daodong: {raw_daodong_csv} (bỏ qua bước tạo từ Excel)")

# 2) Chuyển date -> node sequence
seq_csv = DATA_DIR / "dientich_ACB.csv"
if raw_daodong_csv.exists():
    convert_date_to_sequence(raw_daodong_csv, seq_csv)

    # 3) Vẽ đồ thị cột trên chuỗi daodong (đã đánh số)
    ve_do_thi_dien_tich_cot(seq_csv, 30)

    # 4) Tạo mạng HVG từ chuỗi daodong đã đánh số
    final_csv = DATA_DIR / "data_final.csv"
    tao_mang_visibility(seq_csv, final_csv)
else:
    print(f"⚠️ Bỏ qua tạo HVG vì không có file: {raw_daodong_csv}")

# 5) Tùy chọn: vẽ hình minh họa PTL đơn giản
# create_ptl()

# 6) Vẽ đồ thị theo ngày nếu có file (date, daodong)
if raw_daodong_csv.exists():
    ve_do_thi_dien_tich(raw_daodong_csv, 30)
else:
    print(f"⚠️ Không thể vẽ ve_do_thi_dien_tich vì thiếu file: {raw_daodong_csv}")