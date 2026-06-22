import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- CẤU HÌNH ---
DATA_PATH = 'My_Dataset'
# quét các thư mục con (A, B, C, Heart, Hi...)
actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
actions.sort()

print(f"Các nhãn phát hiện được: {actions}\n")

# PHẦN 1: THỐNG KÊ SỐ LƯỢNG MẪU (CLASS DISTRIBUTION)

print("--- 1. PHÂN PHỐI DỮ LIỆU ---")
counts = {}
total_samples = 0

for action in actions:
    kp_dir = os.path.join(DATA_PATH, action, 'keypoints')
    if os.path.exists(kp_dir):
        num_files = len([f for f in os.listdir(kp_dir) if f.endswith('.npy')])
        counts[action] = num_files
        total_samples += num_files

# Vẽ biểu đồ Bar Chart
plt.figure(figsize=(10, 5))
bars = plt.bar(counts.keys(), counts.values(), color='#4C72B0')
plt.title(f'Phân phối số lượng mẫu mỗi nhãn (Tổng: {total_samples} samples)')
plt.ylabel('Số lượng chuỗi (30 frames)')
plt.xlabel('Nhãn (Hành động)')

# Thêm số lượng trên đỉnh mỗi cột
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')

plt.show()


# PHẦN 2: KIỂM TRA TÍNH TOÀN VẸN CỦA 1 SAMPLE BẤT KỲ

sample_action = actions[10]
sample_file = os.listdir(os.path.join(DATA_PATH, sample_action, 'keypoints'))[115]
sample_path = os.path.join(DATA_PATH, sample_action, 'keypoints', sample_file)

data = np.load(sample_path) # Shape: (30, 126)

print("\n--- 2. KIỂM TRA TÍNH TOÀN VẸN (SHAPE & MISSING VALUES) ---")
print(f"Kiểm tra file: {sample_action}/keypoints/{sample_file}")
print(f"Kích thước mảng: {data.shape} -> (frames, features)")

zero_frames = np.sum(np.all(data == 0, axis=1))
print(f"Số khung hình không bắt được tay (toàn số 0): {zero_frames} / 30")

# PHẦN 3: TRỰC QUAN HÓA 3D (1 FRAME CỤ THỂ)
# Các cặp điểm nối khớp xương theo chuẩn MediaPipe
connections = [(0,1), (1,2), (2,3), (3,4),        # Ngón cái
               (0,5), (5,6), (6,7), (7,8),        # Ngón trỏ
               (5,9), (9,10), (10,11), (11,12),   # Ngón giữa
               (9,13), (13,14), (14,15), (15,16), # Ngón áp út
               (13,17), (0,17), (17,18), (18,19), (19,20)] # Ngón út

def plot_3d_hand(frame_data, title="Trực quan hóa xương tay 3D"):
    """Hàm vẽ khung xương tay trái và phải trong không gian 3D"""
    lh = frame_data[:63].reshape(21, 3) # Tay trái
    rh = frame_data[63:].reshape(21, 3) # Tay phải

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    if np.any(lh):
        ax.scatter(lh[:, 0], lh[:, 1], lh[:, 2], c='blue', s=20, label='Tay Trái (Left)')
        for start, end in connections:
            ax.plot([lh[start,0], lh[end,0]], [lh[start,1], lh[end,1]], [lh[start,2], lh[end,2]], 'blue')

    if np.any(rh):
        ax.scatter(rh[:, 0], rh[:, 1], rh[:, 2], c='darkorange', s=20, label='Tay Phải (Right)')
        for start, end in connections:
            ax.plot([rh[start,0], rh[end,0]], [rh[start,1], rh[end,1]], [rh[start,2], rh[end,2]], 'darkorange')

    ax.set_title(title)
    ax.set_xlabel('X') ; ax.set_ylabel('Y') ; ax.set_zlabel('Z')
    
    ax.invert_yaxis() 
    ax.view_init(elev=20, azim=-60)
    plt.legend()
    plt.show()

print("\n--- 3. TRỰC QUAN HÓA 3D BÀN TAY ---")
plot_3d_hand(data[15], title=f"Frame 15 của nhãn '{sample_action}'")


# PHẦN 4: TRỰC QUAN HÓA QUỸ ĐẠO THỜI GIAN (TEMPORAL MOVEMENT)
print("\n--- 4. TRỰC QUAN HÓA CHUYỂN ĐỘNG QUA 30 FRAMES ---")

rh_index_finger_x = data[:, 63 + 24] # Trục X
rh_index_finger_y = data[:, 63 + 25] # Trục Y
rh_index_finger_z = data[:, 63 + 26] # Trục Z

plt.figure(figsize=(10, 4))
plt.plot(rh_index_finger_x, label='X (Ngang)', marker='o', markersize=4)
plt.plot(rh_index_finger_y, label='Y (Dọc)', marker='s', markersize=4)
plt.plot(rh_index_finger_z, label='Z (Sâu)', marker='^', markersize=4)

plt.title('Biến thiên tọa độ Đầu ngón trỏ phải (Index 8) trong 30 frames')
plt.xlabel('Frame Number')
plt.ylabel('Tọa độ (so với cổ tay)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()