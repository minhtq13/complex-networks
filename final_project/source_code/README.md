## ComplexNetwork

Author: Tạ Quang Minh - 20242537
Phân tích mạng từ chuỗi thời gian dao động (Horizontal Visibility Graph - HVG) và các chỉ số mạng trên dữ liệu cổ phiếu. Dự án này là final project cho môn Complex Network.

## Cấu trúc thư mục

- `src/`
  - `hvg_builder.py`: Hàm xử lý/biến đổi dữ liệu, tạo HVG, vẽ mạng tô màu theo out-degree.
  - `network_analysis.py`: Bộ hàm trực quan hóa và phân tích mạng, ghi ảnh vào thư mục `output/`.
  - `degree_stats.py`: Thống kê nhanh bậc vào/ra, số node/cạnh từ `data_final.csv`.
  - `paths_and_hamilton.py`: Demo đường đi ngắn nhất có trọng số và kiểm tra Hamilton path (mẫu nhỏ), thống kê liên thông mạnh.
- `data/`
  - `data_final.csv`: Dữ liệu cạnh HVG gồm các cột: `node, node_link, weight` (có thể kèm `daodong, daodong_node_link` nếu tạo bằng `tao_mang_visibility`).
  - Các file dữ liệu gốc: `gia_lich_su_ohlcv_ACB.xlsx`, `dien_tich_ACB.csv`, `dientich_ACB.csv`, `macophieu.csv`, `TCBS_532025.csv`, `VCI_532025.csv`.
- `notebooks/`
  - `data_stock.ipynb`: Notebook thử nghiệm/phân tích dữ liệu.
- `output/`
  - Lưu các hình ảnh kết quả phân tích.
- `requirements.txt`, `README.md`.

## Thiết lập môi trường

- Python 3.9+ (khuyến nghị)
- Cài thư viện:

```bash
pip install -r requirements.txt
```

## Luồng xử lý dữ liệu

1. Từ OHLCV → “dao động” (tùy chọn, nếu bạn đã có `data/dien_tich_ACB.csv` có thể bỏ qua):
   - Dùng `dientich_csv` trong `src/my_f2.py` để xuất `date, daodong` từ `data/gia_lich_su_ohlcv_ACB.xlsx`.
2. Đánh số node theo thời gian (tùy chọn):
   - `convert_date_to_sequence(input_csv, output_csv)` để thêm cột `node` (1..n) và giữ `daodong`.
3. Tạo HVG (nếu cần tạo mới `data_final.csv`):
   - `tao_mang_visibility(input_csv, data/data_final.csv)` để sinh các cạnh, đảm bảo có cột `daodong` cho cả hai đầu mút.
4. Phân tích/Trực quan:
   - Dùng `src/my_f.py` và/hoặc `src/my_f2.py` để sinh biểu đồ/ảnh trong `output/`.

## Cách chạy

- Vẽ mạng tô màu theo out-degree và lưu `output/network_mau.png`:

```bash
python src/hvg_builder.py
```

- Sinh các biểu đồ phân tích mạng (centrality, k-core, SCC, paths...) vào `output/`:

```bash
python src/network_analysis.py
```

- Thống kê nhanh bậc vào/ra, số node/cạnh:

```bash
python src/degree_stats.py
```

- Khảo sát đường đi ngắn nhất có trọng số và Hamilton path (mẫu):

```bash
python src/paths_and_hamilton.py
```

## Đầu ra tiêu biểu (có sẵn trong `output/`)

- `graph_network.png`, `graph_daodong.png`, `hist_weight.png`, `degree_distribution.png`
- `diffusion_from_node272.png`, `shortest_path_distribution.png`
- `edge_disjoint_paths_1_10.png`, `edge_cut_1_10.png`, `node_cut_1_10.png`
- `scc_size_distribution.png`, `largest_scc_subgraph.png`
- `degree_centrality.png`, `eigenvector_centrality.png`, `closeness_centrality.png`, `betweenness_centrality.png`, `k_core.png`

## Hướng phát triển

- Tối ưu hiệu năng tạo HVG (hiện O(n^2)): dùng kỹ thuật stack/segment tree để giảm độ phức tạp (tiệm cận O(n log n)).
- Chuẩn hóa schema dữ liệu: đảm bảo `data_final.csv` luôn có `daodong` và `daodong_node_link` khi cần tái tính trọng số, kèm mô tả cột rõ ràng.
- Tham số hóa CLI: thêm argparse cho các script để cấu hình đường dẫn input/output, tham số k của k-core, top-k centrality, node nguồn/đích cho path.
- Lưu Graph ra định dạng chuẩn (graphml/gexf) để dùng với Gephi/Neo4j; kèm script import/export.
- Phân tích nhạy cảm: so sánh các công thức trọng số khác (ví dụ w = |y_i - y_j|, hay chỉ dùng 1/(j-i)) và tác động lên centrality/kết cấu mạng.
- Validation thống kê: bootstrap trên các phân đoạn thời gian khác nhau; kiểm thử tính ổn định của thứ hạng centrality.
- Notebook: minh họa end-to-end (OHLCV → daodong → HVG → phân tích) để tái lập thí nghiệm nhanh.
