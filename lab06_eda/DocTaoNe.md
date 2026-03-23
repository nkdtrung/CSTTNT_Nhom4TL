# Lab 06 - Exploratory Data Analysis (EDA)

## Dataset: Pima Indians Diabetes

---

## 📁 Cấu Trúc Thư Mục

```
LAB06_EDA/
├── Data/
│   ├── pima-indians-diabetes.csv       # Dataset chính (768 mẫu, 9 cột)
│   └── pima-indians-diabetes.names     # Mô tả ý nghĩa từng cột
├── [Tutorial] EDA-Python.ipynb         # File mẫu (dataset Iris - chỉ tham khảo)
├── ex1_FeatureEngineering.ipynb        # Ví dụ xử lý và biến đổi features
├── ex2_MissingValues.ipynb             # Ví dụ xử lý missing values
├── ex3_eg_plot_3_features.ipynb        # Ví dụ vẽ biểu đồ
└── Lab3_EDA_Pima_Diabetes.ipynb        # ✅ FILE CHÍNH - Code EDA của
```

---

## 📊 Thông Tin Dataset

| Thông tin       | Chi tiết                                        |
| --------------- | ----------------------------------------------- |
| **Tên dataset** | Pima Indians Diabetes                           |
| **Nguồn**       | https://archive.ics.uci.edu/dataset/34/diabetes |
| **Số mẫu**      | 768 bệnh nhân                                   |
| **Số cột**      | 9 (8 features + 1 nhãn)                         |

### Ý nghĩa các cột:

| Cột                      | Ý nghĩa                                      |
| ------------------------ | -------------------------------------------- |
| Pregnancies              | Số lần mang thai                             |
| Glucose                  | Nồng độ glucose trong máu                    |
| BloodPressure            | Huyết áp tâm trương (mm Hg)                  |
| SkinThickness            | Độ dày da cơ tam đầu (mm)                    |
| Insulin                  | Insulin huyết thanh 2 giờ (mu U/ml)          |
| BMI                      | Chỉ số khối cơ thể                           |
| DiabetesPedigreeFunction | Chỉ số tiền sử gia đình tiểu đường           |
| Age                      | Tuổi                                         |
| **Outcome**              | **0 = Không tiểu đường / 1 = Có tiểu đường** |

---

## 📋 Nội Dung File Lab3_EDA_Pima_Diabetes.ipynb

File gồm **7 sections** chính:

| Section                       | Nội dung                                          | Kết quả                      |
| ----------------------------- | ------------------------------------------------- | ---------------------------- |
| **1. Load Data & Tổng Quan**  | Load CSV, xem shape, head, info                   | Bảng dữ liệu 768 x 9         |
| **2. Thống Kê Mô Tả**         | describe(), kiểu dữ liệu, thống kê cơ bản         | Bảng thống kê min/max/mean   |
| **3. Missing Values Ẩn**      | Phát hiện giá trị 0 bất hợp lệ, xử lý bằng median | Bảng % missing values        |
| **4. Phân Tích Phân Phối**    | Histogram từng cột                                | Biểu đồ phân phối 8 features |
| **5. Boxplot Outliers**       | Phát hiện ngoại lệ                                | Biểu đồ boxplot              |
| **6. Heatmap Tương Quan**     | Ma trận tương quan giữa các cột                   | Heatmap màu sắc              |
| **7. Phân Tích theo Outcome** | So sánh nhóm bệnh vs không bệnh                   | Biểu đồ so sánh 0 vs 1       |

---

## 🔍 Phát Hiện Quan Trọng Trong EDA

### Missing Values Ẩn:

Các cột sau chứa giá trị **0 không hợp lệ** về mặt y tế:

| Cột           | Số giá trị 0 | Tỷ lệ |
| ------------- | ------------ | ----- |
| Insulin       | 374          | ~49%  |
| SkinThickness | 227          | ~30%  |
| BloodPressure | 35           | ~5%   |
| BMI           | 11           | ~1.4% |
| Glucose       | 5            | ~0.7% |

→ Xử lý: thay thế bằng **median** của từng cột
