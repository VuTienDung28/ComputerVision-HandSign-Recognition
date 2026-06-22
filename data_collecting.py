import cv2
import numpy as np
import os
import mediapipe as mp
import time

# 1. KHỞI TẠO MEDIAPIPE HANDS
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 2. THIẾT LẬP THÔNG SỐ DATASET
DATA_PATH = os.path.join('My_Dataset') 
actions = np.array(['Talk']) 

# --- THAY ĐỔI TẠI ĐÂY: Nhập khoảng video bạn muốn quay ---
START_SEQ = 131  
END_SEQ =  150
sequence_length = 30 

# 3. TỰ ĐỘNG TẠO CẤU TRÚC THƯ MỤC
for action in actions: 
    os.makedirs(os.path.join(DATA_PATH, action, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, action, 'keypoints'), exist_ok=True)

# Hàm trích xuất tọa độ
def extract_keypoints(results):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            ref_x = hand_landmarks.landmark[0].x
            ref_y = hand_landmarks.landmark[0].y
            ref_z = hand_landmarks.landmark[0].z
            
            keypoints = np.array([[res.x - ref_x, res.y - ref_y, res.z - ref_z] 
                                  for res in hand_landmarks.landmark]).flatten()
            
            label = handedness.classification[0].label
            if label == 'Left': 
                lh = keypoints
            else: 
                rh = keypoints
                
    return np.concatenate([lh, rh])

# 4. BẮT ĐẦU QUAY PHIM
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30 

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    for action in actions:
        for sequence in range(START_SEQ, END_SEQ + 1):
            
            # --- CHUYỂN KHỞI TẠO VIDEO LÊN TRƯỚC ĐẾM NGƯỢC ---
            video_path = os.path.join(DATA_PATH, action, 'videos', f'{sequence}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # --- VÒNG LẶP ĐẾM NGƯỢC 2 GIÂY ---
            start_time = time.time()
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time >= 2.0:
                    break # Hết 2 giây đếm ngược thì thoát vòng lặp
                
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                
                # BẮT ĐẦU QUAY RAW VIDEO TRƯỚC 0.75s (Tức là khi đã trôi qua 1.25s)
                if elapsed_time >= 1.25:
                    out.write(frame) # Ghi frame sạch (không dính text)
                
                # Vẽ text lên bản copy để hiển thị
                display_frame = frame.copy()
                remain = int(2 - elapsed_time) + 1
                cv2.putText(display_frame, f'STARTING IN {remain}s...', (150, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
                
                # Hiển thị trạng thái Record sớm (Nhấp nháy đỏ)
                if elapsed_time >= 1.25:
                    cv2.putText(display_frame, 'PRE-RECORDING...', (15, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.putText(display_frame, f'Next: {action} #{sequence}', (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', display_frame)
                cv2.waitKey(1) 

            # --- BẮT ĐẦU QUAY 30 FRAME CHÍNH THỨC (CÓ LẤY KEYPOINTS) ---
            sequence_data = [] 
            frame_num = 0
            
            while frame_num < sequence_length:
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False 
                results = hands.process(image)
                image.flags.writeable = True
                
                keypoints = extract_keypoints(results)
                display_frame = frame.copy()
                
                if np.any(keypoints): 
                    out.write(frame) # Tiếp tục ghi vào video
                    sequence_data.append(keypoints) # Bắt đầu lưu tọa độ
                    
                    frame_num += 1
                    color = (0, 255, 0)
                    msg = f'Recording {action} #{sequence}: Frame {frame_num}/30'
                else:
                    color = (0, 0, 255)
                    msg = "KEEP HANDS IN VIEW!"

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.putText(display_frame, msg, (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', display_frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    exit()
                    
            out.release() 
            
            npy_path = os.path.join(DATA_PATH, action, 'keypoints', str(sequence)) 
            np.save(npy_path, np.array(sequence_data))

cap.release()
cv2.destroyAllWindows()
print("Hoàn tất thu thập và lưu trữ!")