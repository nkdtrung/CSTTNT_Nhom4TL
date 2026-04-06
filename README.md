# Machine Learning Classification Projects

Dự án tập trung vào Exploratory Data Analysis (EDA), Data Preprocessing, và Binary Classification sử dụng các thuật toán Machine Learning hiện đại. Bộ sưu tập này bao gồm ba phần chính: dự án cuối kỳ về phân loại tiền giả, lab thực hành EDA, và thực hành về preprocessing dữ liệu.

---

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Cấu trúc dự án](#cấu-trúc-dự-án)
3. [Chi tiết từng phần](#chi-tiết-từng-phần)
4. [Công nghệ sử dụng](#công-nghệ-sử-dụng)
5. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
6. [Kết quả chính](#kết-quả-chính)
7. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Tổng quan

Dự án này là phần của khóa học về Khoa học Dữ liệu và Machine Learning. Nó bao gồm các bài thực hành liên quan đến:

- **Exploratory Data Analysis (EDA)**: Phân tích và khám phá các dataset để hiểu rõ đặc trưng dữ liệu
- **Data Preprocessing**: Xử lý dữ liệu bị thiếu, chuẩn hóa, và lựa chọn đặc trưng
- **Binary Classification**: Áp dụng nhiều thuật toán Machine Learning và so sánh hiệu suất
- **Model Tuning**: Tối ưu hóa siêu tham số (hyperparameter) để nâng cao độ chính xác

---

## Cấu trúc dự án

```
CSTTNT_Nhom4TL/
│
├── README.md                                  # File này
│
├── CSTTNT_DeAnCuoiKy/                        # Dự án cuối kỳ - Phân loại tiền giả
│   ├── banknote_authentication.csv            # Dataset chính (1372 mẫu)
│   ├── banknote_classification.py             # Script Python chính
│   ├── baseline_results.csv                   # Kết quả 10 mô hình baseline
│   ├── tuned_results.csv                      # Kết quả sau khi fine-tuning
│   ├── Dashboard_KetQua.ipynb                # Notebook dashboard kết quả
│   ├── NhatKy_ThucNghiem.ipynb                # Nhật ký thực nghiệm
│   ├── eda_distributions.png                  # Biểu đồ phân phối đặc trưng
│   ├── eda_boxplot.png                        # Biểu đồ hộp (Boxplot)
│   ├── eda_correlation.png                    # Ma trận tương quan
│   ├── eda_pairplot.png                       # Pairplot - tất cả đặc trưng
│   ├── confusion_matrices.png                 # Ma trận nhầm lẫn top 3 mô hình
│   ├── model_comparison.png                   # So sánh 10 mô hình baseline
│   ├── final_comparison_f1.png                # So sánh F1-Score sau tuning
│   └── roc_curve.png                          # Đường cong ROC
│
├── lab06_eda/                                 # Lab 06 - EDA với Pima Diabetes
│   ├── [Tutorial] EDA-Python.ipynb            # File hướng dẫn với dataset Iris
│   ├── Lab3_EDA_Pima_Diabetes.ipynb           # Notebook EDA chính của lab
│   ├── ex1_FeatureEngineering.ipynb           # Ví dụ feature engineering
│   ├── ex2_MissingValues.ipynb                # Ví dụ xử lý missing values
│   ├── ex3_eg_plot_3_features.ipynb           # Ví dụ vẽ biểu đồ
│   ├── DocTaoNe.md                            # Hướng dẫn về dataset
│   └── Data/
│       ├── pima-indians-diabetes.csv          # Dataset Pima Diabetes (768 bệnh nhân)
│       └── pima-indians-diabetes.names        # Mô tả chi tiết các cột dữ liệu
│
└── TH07/                                      # Thực hành 07 - Preprocessing & Modeling
    ├── BaocaoTH07.docx                        # Báo cáo chi tiết (Word format)
    ├── TH07.pptx                              # Slide thuyết trình
    ├── data/
    │   └── diabetes.csv                       # Dataset Pima Diabetes
    ├── eda/
    │   └── eda_pima.ipynb                     # Notebook EDA
    ├── preprocessing/
    │   ├── preprocessing_pima.ipynb           # Notebook preprocessing chính
    │   └── preprocessing_pima.html            # Báo cáo HTML
    ├── exps/
    │   ├── fix.py                             # Script sửa lỗi dữ liệu
    │   ├── modeling_experiments.ipynb         # Notebook thực nghiệm mô hình
    │   ├── data/
    │   │   └── idx.npz                        # Chỉ số dữ liệu (NumPy format)
    │   └── feature1/
    │       ├── feat_minmax.npz                # Đặc trưng MinMax scaled
    │       ├── feat_minmax_test.npz           # Dữ liệu test MinMax scaled
    │       ├── feat_standard.npz              # Đặc trưng Standard scaled
    │       ├── feat_standard_test.npz         # Dữ liệu test Standard scaled
    │       ├── feat_standard_bal.npz          # Dữ liệu cân bằng Standard scaled
    │       ├── minmax_scaler.joblib           # MinMax Scaler object
    │       ├── standard_scaler.joblib         # Standard Scaler object
    │       ├── selected_features.npz          # Đặc trưng được lựa chọn
    │       ├── scale_columns.npz              # Thông tin cột scaling
    │       ├── selector_kbest.joblib          # K-Best Feature Selector object
    │       └── X_train_bal.npz                # Dữ liệu training cân bằng
    └── models/                                # Lưu trữ các mô hình đã huấn luyện

```

---

## Chi tiết từng phần

### 1. CSTTNT_DeAnCuoiKy - Phân loại tiền giả (Banknote Classification)

**Mục tiêu**: Xây dựng mô hình phân loại nhị phân để phân biệt tiền giả với tiền thật dựa trên các đặc trưng hình ảnh.

#### Dataset

- **Tên**: Banknote Authentication Dataset
- **Kích thước**: 1372 mẫu (samples)
- **Số đặc trưng**: 4 đặc trưng số thực (float64)
- **Số lớp (Classes)**: 2
  - Class 0 (Genuine - Tiền thật): 762 mẫu (55.5%)
  - Class 1 (Forged - Tiền giả): 610 mẫu (44.5%)
- **Tình trạng dữ liệu**: Không có giá trị thiếu, cân bằng tích cực

#### Các đặc trưng (Features)

| Đặc trưng | Mô tả |
|-----------|-------|
| **Variance** | Phương sai của ảnh chuyển đổi Wavelet |
| **Skewness** | Độ lệch (skewness) của ảnh chuyển đổi Wavelet |
| **Curtosis** | Độ nhọn (kurtosis) của ảnh chuyển đổi Wavelet |
| **Entropy** | Entropy của ảnh chuyển đổi Wavelet |

#### Quy trình thực hiện

**Bước 1: Tải dữ liệu**
- File CSV không có header
- Xác định tên cột tự động

**Bước 2: Phân tích dữ liệu ban đầu**
- Kiểm tra giá trị thiếu: Không có
- Phân phối lớp: Cân bằng hợp lý (55.5% vs 44.5%)
- Thống kê mô tả (mean, std, min, max, quartiles)

**Bước 3: Exploratory Data Analysis (EDA)**
- Biểu đồ phân phối đặc trưng theo lớp (histograms)
- Biểu đồ hộp (boxplots) để phát hiện outliers
- Ma trận tương quan heatmap
- Pairplot để visualize mối quan hệ giữa các đặc trưng

**Bước 4: Tiền xử lý dữ liệu**
- Chia dữ liệu: Train (70%) / Test (30%)
- Stratified split để bảo toàn tỷ lệ lớp
- Chuẩn hóa dữ liệu (StandardScaler): Fit trên train, áp dụng trên test

**Bước 5: Huấn luyện Baseline Models - 10 thuật toán**

| Model | Mô tả |
|-------|-------|
| **kNN** | K-Nearest Neighbors (k=5 mặc định) |
| **Naive Bayes** | Gaussian Naive Bayes |
| **SVM** | Support Vector Machine với RBF kernel |
| **Decision Tree** | Cây quyết định phân loại |
| **Random Forest** | Rừng ngẫu nhiên (100 cây) |
| **AdaBoost** | Adaptive Boosting (50 base estimators) |
| **Gradient Boosting** | Gradient Boosting Classifier |
| **LDA** | Linear Discriminant Analysis |
| **MLP** | Multi-Layer Perceptron (Neural Network) |
| **Logistic Regression** | Hồi quy Logistic |

- Cross-validation: Stratified K-Fold (k=5)
- Metrics đánh giá: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Ghi nhận thời gian huấn luyện

**Bước 6: Phân tích kết quả Baseline**
- Lập bảng so sánh chi tiết các mô hình
- Vẽ biểu đồ cột so sánh các metrics
- Xác định Top 3 mô hình theo F1-Score

**Bước 7: Confusion Matrix và ROC Curve**
- Vẽ confusion matrix cho top 3 mô hình
- Vẽ đường cong ROC (Receiver Operating Characteristic)
- Tính AUC (Area Under Curve) cho từng mô hình

**Bước 8: Hyperparameter Tuning**
- Áp dụng GridSearchCV/RandomizedSearchCV cho các mô hình hàng đầu
- Les mô hình được fine-tune:
  - Random Forest (Tuned)
  - SVM (Tuned)
  - Gradient Boosting (Tuned)
  - kNN (Tuned)
  - MLP (Tuned)
  - Stacking Ensemble (kết hợp nhiều mô hình)

**Bước 9: So sánh kết quả Final**
- Baseline vs Tuned models
- Visualize improvement

#### Kết quả Baseline

```
Model                 CV Acc (mean)  Accuracy  Precision  Recall  F1-Score  AUC
kNN                        0.9969      1.0        1.0      1.0     1.0      1.0
Naive Bayes                0.8396      0.8641     0.8547   0.8361  0.8453   0.9483
SVM                        0.999       1.0        1.0      1.0     1.0      1.0
Decision Tree              0.9844      0.9879     0.9785   0.9945  0.9864   0.9885
Random Forest              0.9896      0.9951     0.9892   1.0     0.9946   0.9999
AdaBoost                   0.9938      0.9976     0.9946   1.0     0.9973   1.0
Gradient Boosting          0.9896      0.9927     0.9839   1.0     0.9919   0.9998
LDA                        0.976       0.9684     0.9337   1.0     0.9657   0.9998
MLP                        1.0         1.0        1.0      1.0     1.0      1.0
Logistic Regression        0.9833      0.9782     0.9531   1.0     0.976    0.9999
```

#### Kết quả sau Fine-tuning

```
Model                Accuracy  Precision  Recall  F1-Score  AUC
RF (Tuned)             0.9951      0.9892   1.0     0.9946    0.9999
SVM (Tuned)            1.0         1.0      1.0     1.0       1.0
GB (Tuned)             0.9976      0.9946   1.0     0.9973    0.9999
kNN (Tuned)            1.0         1.0      1.0     1.0       1.0
MLP (Tuned)            1.0         1.0      1.0     1.0       1.0
Stacking Ensemble      1.0         1.0      1.0     1.0       1.0
```

#### Kết luận

- **Mô hình tốt nhất**: SVM (Tuned), kNN (Tuned), MLP (Tuned), Stacking Ensemble đạt 100% accuracy
- **Mô hình cân bằng nhất**: Gradient Boosting tuned - hiệu suất cao (99.76%) với thời gian huấn luyện hợp lý
- **Kết quả**: Dataset này có độ khó tương đối thấp, các mô hình đều hoạt động rất tốt

---

### 2. lab06_eda - Exploratory Data Analysis

**Mục tiêu**: Thực hành EDA với dataset Pima Diabetes để học các kỹ thuật phân tích và trực quan hóa dữ liệu.

#### Dataset - Pima Indians Diabetes

- **Tên**: Pima Indians Diabetes Database
- **Nguồn**: National Institute of Diabetes and Digestive and Kidney Diseases
- **Kích thước**: 768 bệnh nhân
- **Số đặc trưng**: 8 đặc trưng y tế
- **Target**: Outcome (0 = Không tiểu đường, 1 = Có tiểu đường)

#### Các đặc trưng (Features)

| STT | Đặc trưng | Ý nghĩa | Đơn vị |
|-----|-----------|---------|--------|
| 1 | **Pregnancies** | Số lần mang thai | Lần |
| 2 | **Glucose** | Nồng độ glucose trong máu (2 giờ sau uống glucose) | mg/dL |
| 3 | **BloodPressure** | Huyết áp tâm trương (diastolic blood pressure) | mm Hg |
| 4 | **SkinThickness** | Độ dày da cơ tam đầu (triceps skin fold thickness) | mm |
| 5 | **Insulin** | Insulin huyết thanh 2 giờ sau uống glucose | mu U/ml |
| 6 | **BMI** | Chỉ số khối cơ thể (Body Mass Index) | kg/m² |
| 7 | **DiabetesPedigreeFunction** | Chỉ số tiền sử gia đình tiểu đường | Chỉ số |
| 8 | **Age** | Tuổi | Năm |
| 9 | **Outcome** | **Biến mục tiêu** | 0 hoặc 1 |

#### Nội dung Lab

**Lab3_EDA_Pima_Diabetes.ipynb** (File chính) gồm 7 sections:

1. **Khai báo thư viện và tham số**
   - Import numpy, pandas, matplotlib, seaborn
   - Set up visualizations

2. **Tải dữ liệu**
   - Đọc CSV file
   - Kiểm tra kích thước và kiểu dữ liệu

3. **Kiểm tra giá trị thiếu (Missing Values)**
   - Đếm missing values
   - Chiến lược xử lý

4. **Thống kê mô tả (Descriptive Statistics)**
   - Mean, std, min, max, percentiles
   - Phân tích phân phối

5. **Biểu đồ đơn biến (Univariate Analysis)**
   - Histograms cho từng đặc trưng
   - Kde plots

6. **Biểu đồ đa biến (Multivariate Analysis)**
   - Pairplot
   - Heatmap tương quan
   - Boxplots theo target

7. **Feature Engineering**
   - Tạo đặc trưng mới
   - Biến đổi dữ liệu

#### Các file ví dụ bổ trợ

- **[Tutorial] EDA-Python.ipynb**: Hướng dẫn EDA với dataset Iris
- **ex1_FeatureEngineering.ipynb**: Ví dụ về feature engineering
- **ex2_MissingValues.ipynb**: Kỹ thuật xử lý missing values
- **ex3_eg_plot_3_features.ipynb**: Ví dụ vẽ biểu đồ nâng cao

---

### 3. TH07 - Preprocessing và Modeling Experiments

**Mục tiêu**: Thực hành các kỹ thuật preprocessing dữ liệu và thử nghiệm với nhiều mô hình machine learning.

#### Cấu trúc thư mục TH07

```
TH07/
├── data/                          # Dữ liệu gốc
│   └── diabetes.csv              # Dataset Pima Diabetes
├── eda/                           # Exploratory Data Analysis
│   └── eda_pima.ipynb            # Notebook phân tích dữ liệu
├── preprocessing/                 # Tiền xử lý
│   ├── preprocessing_pima.ipynb   # Notebook preprocessing chính
│   └── preprocessing_pima.html    # Báo cáo HTML
├── exps/                          # Thực nghiệm và mô hình
│   ├── fix.py                     # Script sửa lỗi dữ liệu
│   ├── modeling_experiments.ipynb # Thử nghiệm mô hình
│   ├── data/
│   │   └── idx.npz               # Chỉ số mẫu
│   └── feature1/                  # Đặc trưng sau xử lý
│       ├── feat_minmax.npz        # MinMax normalization
│       ├── feat_standard.npz      # Standardization
│       ├── feat_standard_bal.npz  # Standardized + balanced
│       ├── selected_features.npz  # Selected features
│       ├── X_train_bal.npz        # Balanced training data
│       ├── minmax_scaler.joblib   # Saved scaler
│       ├── standard_scaler.joblib # Saved scaler
│       └── selector_kbest.joblib  # Saved selector
├── models/                        # Lưu trữ mô hình
├── BaocaoTH07.docx               # Báo cáo Word
└── TH07.pptx                      # Slide thuyết trình
```

#### Quy trình Preprocessing

**1. Xử lý giá trị thiếu (Missing Values)**
- Xác định các giá trị 0 bất hợp lý (ví dụ Glucose = 0 không hợp lý)
- Thay thế bằng trung bình hoặc median của nhóm

**2. Chuẩn hóa dữ liệu (Scaling)**
- **StandardScaler**: (X - mean) / std (chuẩn hóa tương đối)
- **MinMaxScaler**: (X - min) / (max - min) (chuẩn hóa về [0, 1])

**3. Lựa chọn đặc trưng (Feature Selection)**
- **SelectKBest**: Chọn K đặc trưng tốt nhất
- **Variance threshold**: Loại bỏ đặc trưng có phương sai thấp
- Correlation analysis: Loại bỏ đặc trưng tương quan cao

**4. Cân bằng dữ liệu (Data Balancing)**
- Kiểm tra imbalance trong target variable
- Sử dụng SMOTE hoặc random oversampling/undersampling

#### Thực nghiệm mô hình (Modeling Experiments)

File **modeling_experiments.ipynb** bao gồm:

- Huấn luyện nhiều mô hình
- So sánh hiệu suất trên dữ liệu preprocessed khác nhau
- Nghiên cứu tác động của:
  - Scaling method (MinMax vs Standard)
  - Feature selection
  - Data balancing

#### Báo cáo

- **BaocaoTH07.docx**: Báo cáo chi tiết về quá trình preprocessing và kết quả
- **TH07.pptx**: Slide thuyết trình các kỹ thuật sử dụng

---

## Công nghệ sử dụng

### Core Libraries

```
Python 3.7+
```

### Data Processing & Analysis

```
pandas              # Xử lý và phân tích dữ liệu
numpy               # Tính toán số học
scipy               # Thuật toán khoa học
```

### Machine Learning

```
scikit-learn        # Các thuật toán ML cơ bản
  - Model selection & evaluation
  - Preprocessing (scaler, encoder)
  - Classification algorithms
  - Feature selection
```

### Visualization

```
matplotlib          # Vẽ biểu đồ cơ bản
seaborn             # Vẽ biểu đồ thống kê nâng cao
  - Heatmaps
  - Pairplots
  - Distribution plots
```

### Utilities

```
joblib              # Lưu/tải mô hình và objects
jupyter             # Jupyter Notebook
ipython             # IPython shell
```

---

## Hướng dẫn sử dụng

### Yêu cầu hệ thống

- Python 3.7 trở lên
- Pip package manager
- Jupyter Notebook hoặc JupyterLab (tùy chọn)

### Cài đặt

1. **Clone repository**
```bash
git clone <repository-url>
cd CSTTNT_Nhom4TL
```

2. **Tạo Python virtual environment (khuyến nghị)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

Hoặc cài đặt thủ công:
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter joblib
```

### Chạy các phần của dự án

#### 1. CSTTNT_DeAnCuoiKy - Banknote Classification

**Chạy script Python chính:**
```bash
cd CSTTNT_DeAnCuoiKy
python banknote_classification.py
```

**Output:**
- Console: In chi tiết từng bước (load, EDA, baseline, tuning)
- CSV files: `baseline_results.csv`, `tuned_results.csv`
- PNG files: 
  - `eda_distributions.png`
  - `eda_boxplot.png`
  - `eda_correlation.png`
  - `eda_pairplot.png`
  - `model_comparison.png`
  - `confusion_matrices.png`
  - `roc_curve.png`
  - `final_comparison_f1.png`

**Hoặc xem kết quả bằng Notebook:**
```bash
jupyter notebook Dashboard_KetQua.ipynb
```

#### 2. lab06_eda - EDA with Pima Diabetes

**Mở notebook chính:**
```bash
cd lab06_eda
jupyter notebook Lab3_EDA_Pima_Diabetes.ipynb
```

**Hoặc xem file hướng dẫn:**
```bash
cat DocTaoNe.md
```

**Chạy các ví dụ (tuần tự):**
```bash
jupyter notebook ex1_FeatureEngineering.ipynb
jupyter notebook ex2_MissingValues.ipynb
jupyter notebook ex3_eg_plot_3_features.ipynb
```

#### 3. TH07 - Preprocessing & Modeling

**Thực hiện preprocessing:**
```bash
cd TH07
jupyter notebook preprocessing/preprocessing_pima.ipynb
```

**Thực nghiệm mô hình:**
```bash
jupyter notebook exps/modeling_experiments.ipynb
```

**Xem báo cáo:**
- Word: `BaocaoTH07.docx` (mở bằng Microsoft Word hoặc LibreOffice)
- PowerPoint: `TH07.pptx` (mở bằng PowerPoint hoặc LibreOffice Impress)
- HTML: `preprocessing/preprocessing_pima.html` (mở bằng trình duyệt web)

---

## Kết quả chính

### 1. Banknote Classification - Điểm nổi bật

**Kết quả Baseline (10 mô hình)**

| Loại mô hình | Mô hình tốt nhất | F1-Score | Độ chính xác |
|--------------|-----------------|----------|------------|
| Neighbor-based | kNN | 1.0000 | 100% |
| Kernel-based | SVM | 1.0000 | 100% |
| Tree-based | Gradient Boosting | 0.9919 | 99.27% |
| Ensemble | AdaBoost | 0.9973 | 99.76% |
| Neural Network | MLP | 1.0000 | 100% |
| Linear | Logistic Regression | 0.9760 | 97.82% |

**Kết quả sau Fine-tuning**

Sau khi áp dụng GridSearchCV và tuning siêu tham số:
- **SVM (Tuned)**: 100% accuracy
- **kNN (Tuned)**: 100% accuracy
- **MLP (Tuned)**: 100% accuracy
- **Stacking Ensemble**: 100% accuracy

**Thời gian huấn luyện (Baseline)**
- Nhanh nhất: Naive Bayes (0.013s)
- Chậm nhất: MLP (2.257s)
- Cân bằng: Random Forest (0.871s)

### 2. Lab06 EDA - Insights từ Pima Diabetes Dataset

**Phân bố target variable**
- Không tiểu đường (Outcome=0): ~65% mẫu
- Có tiểu đường (Outcome=1): ~35% mẫu
- Dataset hơi mất cân bằng

**Đặc trưng quan trọng**
- **Glucose**: Chỉ số mạnh nhất dự đoán tiểu đường
- **BMI**: Tương quan trung bình
- **Age**: Tương quan yếu nhưng đáng chú ý

**Giá trị thiếu**
- Một số cột có giá trị 0 bất hợp lý (Glucose, BloodPressure, BMI)
- Cần thay thế bằng imputation

### 3. TH07 - Preprocessing Impact

**Tác động của Scaling**
- MinMax (0-1): Thích hợp cho neural networks
- Standard (z-score): Thích hợp cho linear models, SVM

**Tác động của Feature Selection**
- Giảm chiều từ 8 xuống 5-6 features tốt nhất
- Cải thiện tốc độ huấn luyện
- Hiệu suất model không giảm đáng kể

**Tác động của Data Balancing**
- Chuẩn hóa F1-score giữa hai lớp
- Recall của lớp thiểu số cải thiện

---

## Tài liệu tham khảo

### Dataset Sources

1. **Banknote Authentication Dataset**
   - URL: https://archive.ics.uci.edu/dataset/267/banknote+authentication
   - Format: CSV, 1372 mẫu, 4 features + 1 label
   - License: Public domain

2. **Pima Indians Diabetes Database**
   - URL: https://archive.ics.uci.edu/dataset/34/diabetes
   - Source: National Institute of Diabetes and Digestive and Kidney Diseases
   - Format: CSV, 768 mẫu, 8 features + 1 target
   - Publication: Smith,J.W., Everhart,J.E., Dickson,W.C., Knowler,W.C., & Johannes,R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus

### Thư viện Python

- **pandas**: https://pandas.pydata.org/
- **scikit-learn**: https://scikit-learn.org/
- **matplotlib**: https://matplotlib.org/
- **seaborn**: https://seaborn.pydata.org/
- **numpy**: https://numpy.org/

### Tài liệu học tập

- **Exploratory Data Analysis**
  - Tukey, John W. (1977). "Exploratory Data Analysis"
  - https://en.wikipedia.org/wiki/Exploratory_data_analysis

- **Feature Engineering**
  - Zheng, A., & Casari, A. (2018). "Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists"

- **Model Evaluation**
  - Scikit-learn Model Evaluation: https://scikit-learn.org/stable/modules/model_evaluation/
  - ROC and AUC: https://en.wikipedia.org/wiki/Receiver_operating_characteristic

- **Hyperparameter Tuning**
  - Grid Search: https://scikit-learn.org/stable/modules/grid_search/
  - Hyperopt: http://hyperopt.github.io/hyperopt/

---

## Ghi chú quan trọng

### 1. Tiền xử lý dữ liệu

- **Luôn fit scaler/encoder trên tập training**, sau đó áp dụng trên test set
- Tránh data leakage (rò rỉ thông tin từ test set)

### 2. Đánh giá mô hình

- **Không chỉ nhìn Accuracy** - sử dụng Precision, Recall, F1-Score
- Với dataset mất cân bằng: Sử dụng **F1-Score** hoặc **ROC-AUC**

### 3. Cross-Validation

- Luôn sử dụng cross-validation để đánh giá ổn định
- Với dataset mất cân bằng: Sử dụng **StratifiedKFold**

### 4. Hyperparameter Tuning

- Tuning trên training set (với cross-validation)
- Đánh giá final trên completely separate test set
- Tránh overfitting khi tuning

---

## Cơ cấu team / Thông tin tác giả

**Dự án**: CSTTNT - Nhóm 4 - Kỳ TL (Tailored Learning / Learning Transfer)

**Các phần của dự án**:
- **CSTTNT_DeAnCuoiKy**: Dự án cuối kỳ
- **lab06_eda**: Bài thực hành 06
- **TH07**: Thực hành 07

**Liên hệ**: [Thêm thông tin liên hệ nếu cần]

---

## License

Dự án này được tạo cho mục đích giáo dục. Datasets được sử dụng từ UCI Machine Learning Repository.

---

## Hướng phát triển trong tương lai

- Thử nghiệm với tuning sâu hơn (bayesian optimization)
- Áp dụng advanced ensemble methods
- Sử dụng automl tools (auto-sklearn, TPOT)
- Deploy mô hình tốt nhất dưới dạng API
- A/B testing với production data

---

**Last Updated**: April 2026
**Version**: 1.0

