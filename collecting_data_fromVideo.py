import cv2
import numpy as np
import os
import shutil
import mediapipe as mp

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_DATA_PATH  = 'My_Dataset'              # Source: contains keypoints/ and videos/
OUTPUT_DATA_PATH = 'My_Dataset_Agumentation' # Dest: flat structure per sign label
SEQUENCE_LENGTH  = 30

# ==============================================================================
# INIT MEDIAPIPE HANDS
# ==============================================================================
mp_hands = mp.solutions.hands


def extract_keypoints(results):
    """
    Extract relative keypoints from MediaPipe results.
    Returns a 126-dim vector = [left_hand (63,), right_hand (63,)].
    Coordinates are normalized relative to the wrist landmark (landmark[0]).
    """
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            ref_x = hand_landmarks.landmark[0].x
            ref_y = hand_landmarks.landmark[0].y
            ref_z = hand_landmarks.landmark[0].z

            keypoints = np.array([
                [res.x - ref_x, res.y - ref_y, res.z - ref_z]
                for res in hand_landmarks.landmark
            ]).flatten()

            # Videos were already flip(1) during recording, so do NOT flip here.
            # MediaPipe will correctly identify Left/Right as in live recording.
            label = handedness.classification[0].label
            if label == 'Left':
                lh = keypoints
            else:
                rh = keypoints

    return np.concatenate([lh, rh])


# ==============================================================================
# STEP 1 — CHECK SOURCE DIRECTORY
# ==============================================================================
if not os.path.exists(INPUT_DATA_PATH):
    print(f"[ERROR] Source directory not found: '{INPUT_DATA_PATH}'")
    exit()

actions = [
    d for d in os.listdir(INPUT_DATA_PATH)
    if os.path.isdir(os.path.join(INPUT_DATA_PATH, d))
]
actions.sort()

print(f"Found {len(actions)} sign(s): {actions}")
print("=" * 60)

# ==============================================================================
# STEP 2 — CREATE OUTPUT STRUCTURE & PROCESS EACH SIGN
# ==============================================================================
total_copied    = 0
total_extracted = 0
total_skipped   = 0

with mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
) as hands:

    for action in actions:
        src_keypoints_dir = os.path.join(INPUT_DATA_PATH, action, 'keypoints')
        src_videos_dir    = os.path.join(INPUT_DATA_PATH, action, 'videos')
        dst_dir           = os.path.join(OUTPUT_DATA_PATH, action)

        os.makedirs(dst_dir, exist_ok=True)

        print(f"\n[{action}]")

        # ------------------------------------------------------------------
        # STEP 2A — COPY ORIGINAL KEYPOINTS: 1.npy -> My_Dataset_Agumentation/A/1.npy
        # ------------------------------------------------------------------
        copied_count = 0
        if os.path.exists(src_keypoints_dir):
            for npy_file in os.listdir(src_keypoints_dir):
                if not npy_file.endswith('.npy'):
                    continue
                src_path = os.path.join(src_keypoints_dir, npy_file)
                dst_path = os.path.join(dst_dir, npy_file)
                # Skip if already copied (resume support)
                if os.path.exists(dst_path):
                    continue
                shutil.copy2(src_path, dst_path)
                copied_count += 1

            total_copied += copied_count
            print(f"  [OK] Copied {copied_count} original keypoints (*.npy)")
        else:
            print(f"  [WARN] keypoints directory not found, skipping copy.")

        # ------------------------------------------------------------------
        # STEP 2B — EXTRACT FROM VIDEO: 1.mp4 -> My_Dataset_Agumentation/A/1_v.npy
        # ------------------------------------------------------------------
        if not os.path.exists(src_videos_dir):
            print(f"  [WARN] videos directory not found, skipping extraction.")
            continue

        video_files = sorted(
            [f for f in os.listdir(src_videos_dir) if f.endswith('.mp4')],
            key=lambda x: int(x.split('.')[0])  # sort numerically
        )

        extracted_count = 0
        skipped_count   = 0

        for video_name in video_files:
            seq_name    = video_name.split('.')[0]   # "1" from "1.mp4"
            output_name = f"{seq_name}_v.npy"         # "1_v.npy"
            video_path  = os.path.join(src_videos_dir, video_name)
            dst_path    = os.path.join(dst_dir, output_name)

            # Skip if already extracted
            if os.path.exists(dst_path):
                skipped_count += 1
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"    [!] Cannot open video: {video_name}")
                skipped_count += 1
                continue

            sequence_data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # BGR -> RGB for MediaPipe
                image   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                keypoints = extract_keypoints(results)

                # Only keep frames where hands are detected
                if np.any(keypoints):
                    sequence_data.append(keypoints)

            cap.release()

            # ------ SEQUENCE LENGTH NORMALIZATION ------
            if len(sequence_data) >= SEQUENCE_LENGTH:
                # Take the LAST 30 frames (skip the "warm-up" frames at the start)
                sequence_data = sequence_data[-SEQUENCE_LENGTH:]
            elif len(sequence_data) > 0:
                # Pad with zero vectors if not enough frames
                valid_frames  = len(sequence_data)
                padding_count = SEQUENCE_LENGTH - valid_frames
                padding       = [np.zeros(126) for _ in range(padding_count)]
                sequence_data.extend(padding)
                print(f"    [WARN] {video_name}: only {valid_frames} valid frames, "
                      f"padded {padding_count} zero frames.")
            else:
                # No hands detected in the entire video
                print(f"    [SKIP] {video_name}: no hands detected.")
                skipped_count += 1
                continue

            # Save _v.npy  ->  shape: (30, 126)
            np.save(dst_path, np.array(sequence_data))
            extracted_count += 1

        total_extracted += extracted_count
        total_skipped   += skipped_count
        print(f"  [OK] Extracted {extracted_count} _v.npy files  |  Skipped: {skipped_count}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("DONE!")
print(f"  Original keypoints copied : {total_copied}")
print(f"  Video keypoints extracted : {total_extracted}")
print(f"  Videos skipped/errors     : {total_skipped}")
print(f"  Output directory          : {OUTPUT_DATA_PATH}/")
print("=" * 60)