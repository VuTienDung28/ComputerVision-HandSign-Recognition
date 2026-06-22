import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# ==========================================
# 1. LOAD DỮ LIỆU TEST VÀ MODEL
# ==========================================
print("Đang tải dữ liệu test, mô hình và lịch sử huấn luyện...")

try:
    X_test = np.load('X_test_S9.npy')
    y_test = np.load('y_test_S9.npy')
except FileNotFoundError:
    print("Không tìm thấy các file dữ liệu test ('X_test_S9.npy', 'y_test_S9.npy'). Vui lòng kiểm tra lại đường dẫn.")
    exit()

try:
    model = load_model('LSTM_ASL_Model_S9_2.h5')
except FileNotFoundError:
    print("Không tìm thấy file mô hình 'LSTM_ASL_Model_S9_2.h5'. Vui lòng chạy huấn luyện trước.")
    exit()

# Tải lịch sử huấn luyện
try:
    with open('train_history_S9_2.pkl', 'rb') as f:
        history = pickle.load(f)
except FileNotFoundError:
    print("Không tìm thấy file lịch sử 'train_history_S9_2.pkl'. Sẽ bỏ qua phần vẽ biểu đồ Line Graph.")
    history = None

actions = ['A', 'B', 'Bye', 'C', 'D', 'Everything', 'G', 'Heart', 'Help', 'Hi', 'I',
            'I love you', 'L', 'Like', 'Love', 'M', 'N', 'No', 'O', 'Ok', 'Q', 'Sorry',
            'Take Photo', 'Talk' , 'U', 'Y']

# ==========================================
# 2. VẼ BIỂU ĐỒ LINE GRAPH ACCURACY VÀ LOSS
# ==========================================
if history is not None:
    plt.figure(figsize=(14, 5))

    # Biểu đồ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True)

    # Biểu đồ Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('Training_Validation_Metrics_S9.png', dpi=300)
    print(" Đã lưu biểu đồ Accuracy và Loss thành 'Training_Validation_Metrics_S9.png'")
    plt.show()

# ==========================================
# 3. ĐÁNH GIÁ (EVALUATION) VÀ XUẤT CHỈ SỐ METRICS TRÊN TẬP TEST
# ==========================================
print("\n==================================================")
print("--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ---")
print("==================================================")

# Dự đoán trên tập Test
y_pred = model.predict(X_test)

# Chuyển đổi One-hot vector về số nguyên (index)
y_true_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# In báo cáo Metrics (Accuracy, Precision, Recall, F1-Score)
print("\n1. BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT):")
print(classification_report(y_true_labels, y_pred_labels, target_names=actions, digits=4))

# ==========================================
# 4. VẼ CONFUSION MATRIX (MA TRẬN NHẦM LẪN)
# ==========================================
cm = confusion_matrix(y_true_labels, y_pred_labels)

# Thiết lập kích thước khung hình
plt.figure(figsize=(12, 10))

# Dùng thư viện Seaborn vẽ Heatmap tông màu xanh lam ('Blues') tương tự màu của bài báo
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=actions, yticklabels=actions,
            annot_kws={"size": 12}) # Kích thước chữ bên trong ô

plt.title('Confusion Matrix - ASL Gesture Recognition', fontsize=16)
plt.ylabel('Thực tế (True Label)', fontsize=14)
plt.xlabel('Dự đoán (Predicted Label)', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Lưu lại ảnh ma trận để chèn vào báo cáo Word
plt.savefig('Confusion_Matrix_S9_2.png', dpi=300)
print("\n Đã lưu ảnh ma trận nhầm lẫn thành 'Confusion_Matrix_S9_2.png'")

# Hiển thị lên màn hình
plt.show()
