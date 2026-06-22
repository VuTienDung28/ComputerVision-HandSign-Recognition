import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import joblib

# Fix encoding cho Windows console
sys.stdout.reconfigure(encoding='utf-8')

# 1. CAU HINH THONG SO & DUONG DAN

DATA_PATH = 'My_Dataset_Agumentation'
SEQUENCE_LENGTH = 30
FEATURES = 126

print("=" * 60)
print("  DATA PREPROCESSING S8 - SU DUNG DU LIEU AUGMENTATION")
print("=" * 60)

print("\n--- BUOC 1: QUET NHAN VA THU THAP DU LIEU ---")
actions = np.array(sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]))
print(f"Cac nhan dang xu ly ({len(actions)} nhan): {actions}")

# 2. CAC HAM XU LY TIN HIEU (giu nguyen tu S7)
def interpolate_missing_frames(sequence):
    df = pd.DataFrame(sequence)
    df = df.replace(0.0, np.nan)
    df = df.interpolate(method='linear', limit_direction='both')
    return df.fillna(0).to_numpy()

def apply_savgol_filter(sequence, window_length=5, polyorder=2):
    smoothed_seq = np.zeros_like(sequence)
    for feature_idx in range(FEATURES):
        feature_data = sequence[:, feature_idx]
        if not np.all(feature_data == 0):
            smoothed_seq[:, feature_idx] = savgol_filter(feature_data, window_length, polyorder)
    return smoothed_seq

# 3. DOC VA TACH TAP DU LIEU
# ====================================
X_train_val_raw, y_train_val_raw = [], []
X_test_raw, y_test_raw = [], []

train_val_count = 0
test_count = 0

for action in actions:
    action_dir = os.path.join(DATA_PATH, action)
    
    # Lay tat ca file .npy trong thu muc
    all_files = [f for f in os.listdir(action_dir) if f.endswith('.npy')]
    
    # Phan loai file: lay so thu tu goc tu ten file
    train_val_files = []  
    test_files = []       
    
    for file_name in all_files:
        # Trich xuat so thu tu
        base_name = file_name.replace('.npy', '')  # "1" hoac "1_v"
        is_video = '_v' in base_name
        seq_id = int(base_name.replace('_v', ''))  # Lay so thu tu goc
        
        if seq_id <= 100:
            # Train/Val: lay CA file goc VA file _v
            train_val_files.append(file_name)
        else:
            # Test: chi lay file goc (khong lay _v de dam bao test tren du lieu thuc)
            if not is_video:
                test_files.append(file_name)
    
    # Doc va tien xu ly cac file Train/Val
    for file_name in train_val_files:
        file_path = os.path.join(action_dir, file_name)
        sequence = np.load(file_path)  # Shape: (30, 126)
        
        # Tien xu ly
        seq_interpolated = interpolate_missing_frames(sequence)
        seq_smoothed = apply_savgol_filter(seq_interpolated)
        
        X_train_val_raw.append(seq_smoothed)
        y_train_val_raw.append(action)
        train_val_count += 1
    
    # Doc va tien xu ly cac file Test
    for file_name in test_files:
        file_path = os.path.join(action_dir, file_name)
        sequence = np.load(file_path)  # Shape: (30, 126)
        
        seq_interpolated = interpolate_missing_frames(sequence)
        seq_smoothed = apply_savgol_filter(seq_interpolated)
        
        X_test_raw.append(seq_smoothed)
        y_test_raw.append(action)
        test_count += 1
    
    print(f"  [{action:15s}] Train/Val: {len(train_val_files):3d} mau (goc + video) | Test: {len(test_files):2d} mau (chi goc)")

# Chuyen sang Numpy Array
X_train_val_raw = np.array(X_train_val_raw)
y_train_val_raw = np.array(y_train_val_raw)
X_test_raw = np.array(X_test_raw)
y_test_raw = np.array(y_test_raw)

print(f"\n  Tong so mau Train+Val: {X_train_val_raw.shape[0]}")
print(f"  Tong so mau Test:      {X_test_raw.shape[0]} ")

# 4. CHUAN HOA (STANDARD SCALER)
print("\n--- BUOC 2: CHUAN HOA ---")
scaler = StandardScaler()

if len(X_train_val_raw) > 0:
    X_train_val_reshaped = X_train_val_raw.reshape(-1, FEATURES)
    X_train_val_scaled = scaler.fit_transform(X_train_val_reshaped)
    X_train_val_final = X_train_val_scaled.reshape(X_train_val_raw.shape)
else:
    X_train_val_final = np.array([])

if len(X_test_raw) > 0:
    X_test_reshaped = X_test_raw.reshape(-1, FEATURES)
    X_test_scaled = scaler.transform(X_test_reshaped)
    X_test_final = X_test_scaled.reshape(X_test_raw.shape)
else:
    X_test_final = np.array([])

joblib.dump(scaler, 'StandardScaler_S9.pkl')
print("Da luu bo chuan hoa vao 'StandardScaler_S9.pkl'")

# 5. MA HOA NHAN (ONE-HOT ENCODING)
print("\n--- BUOC 3: MA HOA NHAN ---")
label_map = {label:num for num, label in enumerate(actions)}

if len(y_train_val_raw) > 0:
    y_tv_mapped = np.array([label_map[label] for label in y_train_val_raw])
    y_train_val_final = to_categorical(y_tv_mapped, num_classes=len(actions))
else:
    y_train_val_final = np.array([])

if len(y_test_raw) > 0:
    y_t_mapped = np.array([label_map[label] for label in y_test_raw])
    y_test_final = to_categorical(y_t_mapped, num_classes=len(actions))
else:
    y_test_final = np.array([])

# 6. CHIA TAP TRAIN - VAL (TI LE 7:3)
print("\n--- BUOC 4: CHIA TAP TRAIN VA VAL (7/3) ---")
if len(X_train_val_final) > 0:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val_final, y_train_val_final, test_size=0.30, random_state=42, stratify=y_train_val_final
    )
    print(f"Kich thuoc tap Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Kich thuoc tap Val:   X={X_val.shape}, y={y_val.shape}")
else:
    print("Khong du du lieu de chia tap Train/Val.")
    
print(f"Kich thuoc tap Test:  X={X_test_final.shape}, y={y_test_final.shape}")

# 7. LUU KET QUA CUOI CUNG
print("\n--- BUOC 5: LUU FILE ---")
if len(X_train_val_final) > 0:
    np.save('X_train_S9.npy', X_train)
    np.save('y_train_S9.npy', y_train)
    np.save('X_val_S9.npy', X_val)
    np.save('y_val_S9.npy', y_val)
    
if len(X_test_final) > 0:
    np.save('X_test_S9.npy', X_test_final)
    np.save('y_test_S9.npy', y_test_final)

print("\nDa xuat thanh cong cac file du lieu dang _S9.npy!")