# 🤟 Hand Sign Recognition – ASL Gesture Detection với LSTM + MediaPipe

Hệ thống nhận diện ngôn ngữ ký hiệu tay (ASL) theo thời gian thực, sử dụng **MediaPipe Hands** để trích xuất tọa độ xương tay và mô hình **LSTM** để phân loại cử chỉ. Hỗ trợ 26 nhãn ký hiệu.

---

## 📋 Mục lục

- [Tổng quan dự án](#-tổng-quan-dự-án)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Yêu cầu cài đặt](#-yêu-cầu-cài-đặt)
- [Dữ liệu](#-dữ-liệu)
- [⚡ Quick Start – Tải dữ liệu sẵn & chạy ngay](#-quick-start--tải-dữ-liệu-sẵn--chạy-ngay)
- [🔬 Hướng dẫn đầy đủ – Từ bước thu thập dữ liệu](#-hướng-dẫn-đầy-đủ--từ-bước-thu-thập-dữ-liệu)
  - [Bước 1 – Thu thập dữ liệu](#bước-1--thu-thập-dữ-liệu)
  - [Bước 2 – Augmentation từ video](#bước-2--augmentation-từ-video)
  - [Bước 3 – Phân tích dữ liệu (EDA)](#bước-3--phân-tích-dữ-liệu-eda)
  - [Bước 4 – Tiền xử lý & tạo tập train/val/test](#bước-4--tiền-xử-lý--tạo-tập-trainvaltest)
  - [Bước 5 – Huấn luyện mô hình](#bước-5--huấn-luyện-mô-hình)
  - [Bước 6 – Đánh giá mô hình](#bước-6--đánh-giá-mô-hình)
  - [Bước 7 – Kiểm tra trực tiếp (Live Test)](#bước-7--kiểm-tra-trực-tiếp-live-test)
  - [Bước 8 – Chạy Web Demo](#bước-8--chạy-web-demo)
- [Nhãn được hỗ trợ](#-nhãn-được-hỗ-trợ)
- [Kết quả huấn luyện](#-kết-quả-huấn-luyện)

---

## 🧠 Tổng quan dự án

```
Video/Webcam  →  MediaPipe Hands  →  Keypoints (126 chiều)  →  LSTM Model  →  Nhãn ký hiệu
```

- **Input**: Chuỗi 30 frame, mỗi frame là vector 126 chiều `[21 điểm × 3 tọa độ × 2 tay]`
- **Coordinates**: Tọa độ tương đối so với cổ tay (landmark[0]) để bất biến với vị trí
- **Model**: 3 tầng LSTM (64 → 128 → 64) + 2 tầng Dense + Softmax output
- **Output**: 1 trong 26 nhãn ký hiệu

---

## 📁 Cấu trúc thư mục

```
ComputerVision 2/
│
├── 📄 data_collecting.py           # Thu thập dữ liệu trực tiếp từ webcam
├── 📄 collecting_data_fromVideo.py # Augmentation: trích keypoints từ video đã quay
├── 📄 EDA_data.py                  # Phân tích & trực quan hóa dữ liệu
├── 📄 Data_Processing_Final.py     # Tiền xử lý, chuẩn hóa, chia tập dữ liệu
├── 📄 Train_LSTM.py                # Huấn luyện mô hình LSTM
├── 📄 Train_Bi_LSTM.py             # (Thực nghiệm) Huấn luyện Bidirectional LSTM
├── 📄 Evaluate_LSTM.py             # Đánh giá mô hình: Confusion Matrix, Metrics
├── 📄 live_test.py                 # Kiểm tra mô hình trực tiếp qua webcam
│
├── 📁 web_demo/
│   ├── app.py                      # FastAPI backend + WebSocket
│   └── frontend/                   # Giao diện web (HTML/CSS/JS)
│
├── 📁 My_Dataset/                  # ⬇️ Tải từ Drive (xem mục Dữ liệu)
│   └── <Nhãn>/
│       ├── keypoints/              # File .npy keypoints từ webcam
│       └── videos/                 # File .mp4 video gốc
│
├── 📁 My_Dataset_Agumentation/     # ⬇️ Sinh ra sau khi chạy Bước 2
│   └── <Nhãn>/
│       ├── 1.npy                   # Keypoints gốc
│       └── 1_v.npy                 # Keypoints trích từ video
│
├── LSTM_ASL_Model_S9_2.h5          # Model đã huấn luyện (phiên bản tốt nhất)
├── StandardScaler_S9.pkl           # Bộ chuẩn hóa tương ứng
├── X_train_S9.npy / y_train_S9.npy # Tập huấn luyện
├── X_val_S9.npy   / y_val_S9.npy   # Tập validation
└── X_test_S9.npy  / y_test_S9.npy  # Tập kiểm tra
```

---

## ⚙️ Yêu cầu cài đặt

### Python
Yêu cầu **Python 3.9 – 3.11** (khuyên dùng 3.10).

### Cài đặt thư viện

```bash
pip install opencv-python mediapipe tensorflow numpy scikit-learn \
            matplotlib seaborn scipy pandas joblib fastapi uvicorn
```

> **Lưu ý GPU**: Nếu muốn huấn luyện nhanh hơn, cài `tensorflow-gpu` thay vì `tensorflow` và đảm bảo đã cài CUDA + cuDNN phù hợp.

---

## 📦 Dữ liệu

Dữ liệu thô (`My_Dataset/`) được nén và lưu trên Google Drive.

> 🔗 **Link Drive**: [https://drive.google.com/drive/folders/1E_2DPkP9kzg_cdDf02bWF6FU5Noud8yt](https://drive.google.com/drive/folders/1E_2DPkP9kzg_cdDf02bWF6FU5Noud8yt)

### Nội dung thư mục Drive

| Thư mục / File | Mô tả |
|----------------|-------|
| `My_Dataset.zip` | Dữ liệu thô: keypoints + video gốc từ webcam (150 mẫu/nhãn) |
| `My_Dataset_Agumentation.zip` | Dữ liệu sau augmentation (keypoints gốc + trích từ video) |

### Cấu trúc file sau khi giải nén `My_Dataset.zip`

```
My_Dataset/
├── A/
│   ├── keypoints/   →  1.npy, 2.npy, ..., 150.npy
│   └── videos/      →  1.mp4, 2.mp4, ..., 150.mp4
├── B/
│   ├── keypoints/
│   └── videos/
...
```

Mỗi file `.npy` có shape `(30, 126)` – tương ứng 30 frame, mỗi frame là 126 giá trị tọa độ keypoints.

**Sau khi tải về**, giải nén vào thư mục gốc của dự án:
```
ComputerVision 2/
├── My_Dataset/           ← giải nén vào đây
├── My_Dataset_Agumentation/  ← giải nén vào đây (nếu muốn bỏ qua Bước 2)
└── ...
```

---

## ⚡ Quick Start – Tải dữ liệu sẵn & chạy ngay

> Dành cho người **không muốn tự thu thập dữ liệu** từ đầu. Tải dataset có sẵn rồi chạy thẳng từ EDA → Tiền xử lý → Huấn luyện.

### Bước QS.1 – Tải & giải nén dữ liệu

1. Truy cập **Google Drive**:  
   🔗 [https://drive.google.com/drive/folders/1E_2DPkP9kzg_cdDf02bWF6FU5Noud8yt](https://drive.google.com/drive/folders/1E_2DPkP9kzg_cdDf02bWF6FU5Noud8yt)

2. Tải xuống **`My_Dataset.zip`** (dùng cho EDA) và **`My_Dataset_Agumentation.zip`** (dùng cho tiền xử lý).

3. Giải nén cả hai vào thư mục gốc của dự án:

```
ComputerVision 2/
├── My_Dataset/              ← sau khi giải nén My_Dataset.zip
│   ├── A/
│   │   ├── keypoints/
│   │   └── videos/
│   ├── B/
│   └── ...
│
├── My_Dataset_Agumentation/ ← sau khi giải nén My_Dataset_Agumentation.zip
│   ├── A/
│   │   ├── 1.npy
│   │   ├── 1_v.npy
│   │   └── ...
│   └── ...
└── ...
```

### Bước QS.2 – Phân tích dữ liệu (EDA) *(tùy chọn)*

```bash
python EDA_data.py
```

Kiểm tra phân phối mẫu, trực quan hóa xương tay 3D và quỹ đạo chuyển động.  
*(Yêu cầu `My_Dataset/` đã được giải nén)*

### Bước QS.3 – Tiền xử lý & tạo tập train/val/test

```bash
python Data_Processing_Final.py
```

Script sẽ đọc `My_Dataset_Agumentation/`, xử lý tín hiệu, chuẩn hóa và tạo ra các file:

```
X_train_S9.npy  y_train_S9.npy
X_val_S9.npy    y_val_S9.npy
X_test_S9.npy   y_test_S9.npy
StandardScaler_S9.pkl
```

### Bước QS.4 – Huấn luyện mô hình

```bash
python Train_LSTM.py
```

Sau khi chạy xong sẽ tạo ra `LSTM_ASL_Model_S9_2.h5` và `train_history_S9_2.pkl`.

### Bước QS.5 – Đánh giá & chạy Demo

```bash
# Đánh giá trên tập test
python Evaluate_LSTM.py

# Chạy Web Demo
cd web_demo
uvicorn app:app --reload --port 8000
```

Mở trình duyệt tại **http://localhost:8000** để dùng thử.

---

## 🔬 Hướng dẫn đầy đủ – Từ bước thu thập dữ liệu

> Dành cho người muốn **tự thu thập dữ liệu mới** hoặc hiểu toàn bộ pipeline từ đầu đến cuối.

---

### Bước 1 – Thu thập dữ liệu

> **Mục đích**: Quay video trực tiếp từ webcam và lưu keypoints + video cho mỗi ký hiệu.

```bash
python data_collecting.py
```

**Cấu hình trước khi chạy** (chỉnh trong file):

```python
actions         = np.array(['Talk'])  # Tên nhãn muốn thu thập
START_SEQ       = 131                 # Index bắt đầu của chuỗi
END_SEQ         = 150                 # Index kết thúc
sequence_length = 30                  # Số frame mỗi chuỗi
```

**Cách hoạt động**:
- Mỗi chuỗi có countdown **2 giây** để chuẩn bị tư thế
- Ghi **30 frame** khi phát hiện tay
- Lưu file `.npy` vào `My_Dataset/<Nhãn>/keypoints/`
- Lưu video `.mp4` vào `My_Dataset/<Nhãn>/videos/`

**Phím điều khiển**: `Q` để thoát sớm.

> 💡 **Mẹo**: Đảm bảo ánh sáng tốt, tay nằm hoàn toàn trong khung hình. Nên thu thập **150 mẫu/nhãn** để đồng đều với dataset gốc.

---

### Bước 2 – Augmentation từ video

> **Mục đích**: Trích xuất lại keypoints từ video đã quay để tăng gấp đôi lượng dữ liệu train/val.

```bash
python collecting_data_fromVideo.py
```

**Cấu hình** (chỉnh trong file):

```python
INPUT_DATA_PATH  = 'My_Dataset'              # Thư mục nguồn
OUTPUT_DATA_PATH = 'My_Dataset_Agumentation' # Thư mục đích
SEQUENCE_LENGTH  = 30
```

**Kết quả tạo ra trong `My_Dataset_Agumentation/`**:
- `<Nhãn>/1.npy`   → Copy keypoints gốc (từ bước thu thập)
- `<Nhãn>/1_v.npy` → Keypoints trích xuất lại từ video `1.mp4`

> Script hỗ trợ **resume** – nếu file đích đã tồn tại thì tự động bỏ qua.

---

### Bước 3 – Phân tích dữ liệu (EDA)

> **Mục đích**: Kiểm tra phân phối, tính toàn vẹn và trực quan hóa dữ liệu trước khi huấn luyện.

```bash
python EDA_data.py
```

**Output**:
1. **Bar chart** – Phân phối số lượng mẫu mỗi nhãn
2. Kiểm tra shape và missing values của 1 sample ngẫu nhiên
3. **3D visualization** – Khung xương tay tại frame 15
4. **Line chart** – Quỹ đạo chuyển động ngón trỏ qua 30 frames

> Yêu cầu `My_Dataset/` đã có dữ liệu trước khi chạy.

---

### Bước 4 – Tiền xử lý & tạo tập train/val/test

> **Mục đích**: Chuẩn hóa dữ liệu, chia tập, lưu file `.npy` và `StandardScaler`.

```bash
python Data_Processing_Final.py
```

**Pipeline xử lý**:
1. Đọc toàn bộ `.npy` từ `My_Dataset_Agumentation/`
2. **Nội suy tuyến tính** cho các frame thiếu tay (giá trị 0)
3. **Savitzky-Golay filter** làm mượt tín hiệu
4. **StandardScaler** chuẩn hóa toàn bộ features
5. **One-hot encoding** cho nhãn
6. Chiến lược chia tập:
   - Mẫu có index ≤ 100 → **Train + Val** (tỉ lệ 70:30, cả file gốc lẫn `_v`)
   - Mẫu có index > 100 → **Test** (chỉ dùng file gốc, không dùng `_v`)

**Files được tạo ra**:

| File | Mô tả |
|------|-------|
| `X_train_S9.npy` | Dữ liệu huấn luyện |
| `y_train_S9.npy` | Nhãn huấn luyện (one-hot) |
| `X_val_S9.npy` | Dữ liệu validation |
| `y_val_S9.npy` | Nhãn validation |
| `X_test_S9.npy` | Dữ liệu kiểm tra |
| `y_test_S9.npy` | Nhãn kiểm tra |
| `StandardScaler_S9.pkl` | Bộ chuẩn hóa để dùng lại khi inference |

---

### Bước 5 – Huấn luyện mô hình

> **Mục đích**: Huấn luyện mạng LSTM trên tập dữ liệu đã tiền xử lý.

```bash
python Train_LSTM.py
```

**Kiến trúc mô hình**:

```
Input: (30, 126)
  ↓ LSTM(64, return_sequences=True)
  ↓ LSTM(128, return_sequences=True)
  ↓ LSTM(64)
  ↓ Dense(64, relu)
  ↓ Dense(32, relu)
  ↓ Dense(26, softmax)   ← Output
```

**Cấu hình huấn luyện**:
- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Epochs: tối đa 500 (EarlyStopping với `patience=20`)
- Batch size: `48`

**Files được tạo ra**:

| File | Mô tả |
|------|-------|
| `LSTM_ASL_Model_S9_2.h5` | Model đã huấn luyện |
| `train_history_S9_2.pkl` | Lịch sử training để vẽ biểu đồ |

> **Lưu ý**: Quá trình huấn luyện có thể mất **30–120 phút** tùy phần cứng.

---

### Bước 6 – Đánh giá mô hình

> **Mục đích**: Tính Accuracy, Precision, Recall, F1 trên tập test và vẽ Confusion Matrix.

```bash
python Evaluate_LSTM.py
```

**Yêu cầu**: Phải có sẵn các file:
- `X_test_S9.npy`, `y_test_S9.npy`
- `LSTM_ASL_Model_S9_2.h5`
- `train_history_S9_2.pkl` *(tùy chọn – để vẽ line graph)*

**Output**:
1. **Line graph** – Accuracy & Loss qua các epochs → lưu `Training_Validation_Metrics_S9.png`
2. **Classification Report** – Precision, Recall, F1 từng nhãn
3. **Confusion Matrix** heatmap → lưu `Confusion_Matrix_S9_2.png`

---

### Bước 7 – Kiểm tra trực tiếp (Live Test)

> **Mục đích**: Test mô hình realtime với webcam, không cần trình duyệt.

```bash
python live_test.py
```

**Yêu cầu**: Phải có sẵn model và scaler. File hiện tại đang trỏ model `S5`, để dùng model mới nhất hãy chỉnh 2 dòng sau trong `live_test.py`:

```python
model  = load_model('LSTM_ASL_Model_S9_2.h5')
scaler = joblib.load('StandardScaler_S9.pkl')
```

**Phím điều khiển**:

| Phím | Hành động |
|------|-----------|
| `S` | Bắt đầu đếm ngược 2 giây rồi nhận diện |
| `Q` | Thoát chương trình |

---

### Bước 8 – Chạy Web Demo

> **Mục đích**: Chạy giao diện web nhận diện ký hiệu qua trình duyệt sử dụng camera.

**Yêu cầu**: Phải có sẵn `LSTM_ASL_Model_S9.h5` và `StandardScaler_S9.pkl` ở thư mục gốc.

```bash
cd web_demo
uvicorn app:app --reload --port 8000
```

Sau đó mở trình duyệt tại: **http://localhost:8000**

**Cơ chế hoạt động**:
- Trình duyệt dùng MediaPipe JS để trích keypoints từ camera
- Gửi keypoints qua **WebSocket** lên server
- Server nhận đủ 15 frame → pad lên 30 frame → predict → trả kết quả
- Giao diện hiển thị nhãn + confidence score realtime

---

## 🏷️ Nhãn được hỗ trợ

| # | Nhãn | # | Nhãn | # | Nhãn |
|---|------|---|------|---|------|
| 1 | A | 10 | Hi | 19 | O |
| 2 | B | 11 | I | 20 | Ok |
| 3 | Bye | 12 | I love you | 21 | Q |
| 4 | C | 13 | L | 22 | Sorry |
| 5 | D | 14 | Like | 23 | Take Photo |
| 6 | Everything | 15 | Love | 24 | Talk |
| 7 | G | 16 | M | 25 | U |
| 8 | Heart | 17 | N | 26 | Y |
| 9 | Help | 18 | No | | |

---

## 📊 Kết quả huấn luyện

| Phiên bản | Kiến trúc | Test Accuracy | Ghi chú |
|-----------|-----------|:-------------:|---------|
| S8 | 3 LSTM + 2 Dense | ~88.8% | Baseline augmentation |
| S9 | 3 LSTM + 2 Dense | > 90% | Cải thiện augmentation pipeline |

---

## 📝 Ghi chú

- Tất cả tọa độ keypoints được tính **tương đối so với cổ tay** (landmark[0]) để loại bỏ ảnh hưởng của vị trí tay trên màn hình.
- Suffix `_S8`, `_S9` trên tên file chỉ **phiên bản thực nghiệm** (Session), không phải version phần mềm.
- Khi thu thập dữ liệu mới, luôn đảm bảo điều kiện ánh sáng tốt và tay nằm hoàn toàn trong khung hình.
