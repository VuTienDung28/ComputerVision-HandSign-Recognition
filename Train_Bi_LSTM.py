import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout ,Bidirectional
from tensorflow.keras.callbacks import EarlyStopping ,ReduceLROnPlateau

# ==========================================
# 1. LOAD DỮ LIỆU ĐÃ TIỀN XỬ LÝ
# ==========================================
print("Đang tải dữ liệu...")
X_train = np.load('X_train_S9.npy')
y_train = np.load('y_train_S9.npy')
X_val = np.load('X_val_S9.npy')
y_val = np.load('y_val_S9.npy')
# Xác định số lượng class
NUM_CLASSES = y_train.shape[1]

# danh sách tên các nhãn theo đúng thứ tự Alphabet/Thư mục trong dataset (theo đúng thứ tự đã được one-hot encoding)

actions = ['A', 'B', 'Bye', 'C', 'D', 'Everything', 'G', 'Heart', 'Help', 'Hi', 'I',
            'I love you', 'L', 'Like', 'Love', 'M', 'N', 'No', 'O', 'Ok', 'Q', 'Sorry',
            'Take Photo', 'Talk' , 'U', 'Y']

# ==========================================
# 2. XÂY DỰNG KIẾN TRÚC LSTM (3 LAYER LSTM + 2 LAYER DENSE)
# ==========================================
model = Sequential()
# Chỉ dùng 2 lớp BiLSTM, giảm số Unit xuống để mạng không bị "thừa chất"
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(30, 126)))
model.add(Dropout(0.3)) # Tăng dropout lên một chút

model.add(Bidirectional(LSTM(32, return_sequences=False))) 
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASSES, activation='softmax'))
# ==========================================
# 3. COMPILE & HIỂN THỊ BẢNG PARAMETERS CỦA MÔ HÌNH (MODEL SUMMARY)
# ==========================================
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- BẢNG THỐNG KÊ THAM SỐ MÔ HÌNH (MODEL SUMMARY) ---")
model.summary()

# ==========================================
# 4. THIẾT LẬP EARLY STOPPING VÀ TRAIN
# ==========================================
# Dừng sớm nếu tập Validation không cải thiện sau 20 epcoh 

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20, 
    restore_best_weights=True
)

print("\n--- BẮT ĐẦU HUẤN LUYỆN (TRAINING) ---")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500, 
    batch_size=48,
    callbacks=[early_stopping]
)

# ==========================================
# 5. LƯU MÔ HÌNH VÀ LỊCH SỬ HUẤN LUYỆN
# ==========================================
# Lưu toàn bộ mô hình (để sau này đem đi Live Test với Webcam)
model.save('LSTM_ASL_Model_S9.h5')
print("\n Đã lưu mô hình huấn luyện thành 'LSTM_ASL_Model_S9.h5'")

# Lưu lịch sử huấn luyện (history) để file riêng vẽ biểu đồ
import pickle
with open('train_history_S9.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print(" Đã lưu lịch sử huấn luyện thành 'train_history_S9.pkl'")