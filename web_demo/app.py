import os
import sys
import json
import numpy as np
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
MODEL_PATH  = os.path.join(BASE_DIR, "LSTM_ASL_Model_S9.h5")
SCALER_PATH = os.path.join(BASE_DIR, "StandardScaler_S9.pkl")

# ── Labels ─────────────────────────────────────────────────────────────────────
ACTIONS = ['A', 'B', 'Bye', 'C', 'D', 'Everything', 'G', 'Heart', 'Help', 'Hi', 'I',
            'I love you', 'L', 'Like', 'Love', 'M', 'N', 'No', 'O', 'Ok', 'Q', 'Sorry',
            'Take Photo', 'Talk', 'U', 'Y']
SEQ_LEN = 30
TARGET_FRAMES = 15
FEATURE_DIM = 126   # 21 landmarks × 3 coords × 2 hands

# ── Load model once at startup ─────────────────────────────────────────────────
print("Loading model and scaler...")
from tensorflow.keras.models import load_model   # noqa: E402 (import after print)
model  = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("[OK] Model loaded!")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Hand Sign Recognition API")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/labels")
async def get_labels():
    return {"labels": ACTIONS}

# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sequence: list[np.ndarray] = []
    window_count = 0   # total prediction windows this session

    try:
        while True:
            raw  = await websocket.receive_text()
            msg  = json.loads(raw)
            kind = msg.get("type")

            if kind == "keypoints":
                kp = np.array(msg["keypoints"], dtype=np.float32)  # (126,)
                sequence.append(kp)

                # ── Progress update ─────────────────────────────────────────
                await websocket.send_text(json.dumps({
                    "type":       "progress",
                    "frameCount": len(sequence),
                    "total":      TARGET_FRAMES,
                }))

                # ── Predict when sequence is full ───────────────────────────
                if len(sequence) == TARGET_FRAMES:
                    # Pad the remaining frames by repeating the last frame
                    padded_seq = sequence + [sequence[-1]] * (SEQ_LEN - TARGET_FRAMES)
                    
                    input_data   = np.array(padded_seq).reshape(-1, FEATURE_DIM)
                    input_scaled = scaler.transform(input_data).reshape(1, SEQ_LEN, FEATURE_DIM)
                    res          = model.predict(input_scaled, verbose=0)[0]
                    pred_idx     = int(np.argmax(res))
                    window_count += 1

                    all_scores = {ACTIONS[i]: round(float(res[i]), 4) for i in range(len(ACTIONS))}

                    await websocket.send_text(json.dumps({
                        "type":        "prediction",
                        "label":       ACTIONS[pred_idx],
                        "confidence":  round(float(res[pred_idx]), 4),
                        "allScores":   all_scores,
                        "windowIndex": window_count,   # cumulative count for this session
                    }))
                    sequence = []  # reset → client decides to continue or stop

            elif kind == "reset":
                sequence    = []
                window_count = 0
                await websocket.send_text(json.dumps({"type": "reset_ack"}))

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
