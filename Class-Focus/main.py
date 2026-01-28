import time
import base64
from collections import deque, Counter
from math import hypot
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
import mediapipe as mp

app = Flask(__name__)

# ---------------------- Settings ----------------------
FRAME_W, FRAME_H = 640, 480
SMOOTH_WINDOW = 7
EYE_CLOSED_EAR = 0.20
SLEEP_SECS = 20.0
NOFACE_NOT_ATTENTIVE_AFTER = 1.0

# Intensity thresholds
LOW_TH = 0.25
MED_TH = 0.45
HIGH_TH = 0.70
# ------------------------------------------------------

mp_face_mesh = mp.solutions.face_mesh
mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

"""cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)"""
cap = None  # lazily initialized when /video_feed is requested

last_labels = deque(maxlen=SMOOTH_WINDOW)
last_face_time = 0.0
sleep_start = None
sleeping = False

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_UP, MOUTH_DOWN = 13, 14
LEFT_EYE_CENTER, RIGHT_EYE_CENTER = 33, 263


def dist(p, q):
    return hypot(p[0] - q[0], p[1] - q[1])


def eye_aspect_ratio(pts, lm):
    p1, p2, p3, p4, p5, p6 = [lm[i] for i in pts]
    A = dist(p2, p6)
    B = dist(p3, p5)
    C = dist(p1, p4)
    return (A + B) / (2.0 * C + 1e-6)


def classify_emotion(lm):
    L = lm[LEFT_EYE_CENTER]
    R = lm[RIGHT_EYE_CENTER]
    face_scale = dist(L, R) + 1e-6

    # Smile score
    mouth_L, mouth_R = lm[MOUTH_LEFT], lm[MOUTH_RIGHT]
    smile_w = dist(mouth_L, mouth_R) / face_scale
    mouth_U, mouth_D = lm[MOUTH_UP], lm[MOUTH_DOWN]
    mouth_open = dist(mouth_U, mouth_D) / face_scale
    smile_score = smile_w + 0.5 * mouth_open

    # Sad score
    mouth_center_y = (mouth_U[1] + mouth_D[1]) / 2.0
    corner_avg_y = (mouth_L[1] + mouth_R[1]) / 2.0
    corner_drop = (corner_avg_y - mouth_center_y) / (face_scale + 1e-6)
    sad_score = corner_drop * 1.3 + (0.4 - mouth_open)
    # sad_score = corner_drop
    # sad_score = corner_drop * 1.8  


    # Neutral zone → stricter
    if smile_score < 0.50 and sad_score < 0.50:
        return "Neutral", None

    # Happy
    # if smile_score > sad_score and smile_score >= 0.30: #10 perfect
    #     if smile_score > HIGH_TH:
    #         return "Happy", "High"
    #     elif smile_score > MED_TH:
    #         return "Happy", "Medium"
    #     else:
    #         return "Happy", "Low"
    if smile_score > sad_score and smile_score >= 0.10 and corner_drop < -0.03:
        if smile_score > HIGH_TH:
            return "Happy", "High"
        elif smile_score > MED_TH:
            return "Happy", "Medium"
        else:
            return "Happy", "Low"
    # Sad
    # if sad_score >= 0.35:
    #     if sad_score > HIGH_TH:
    #         return "Sad", "High"
    #     elif sad_score > MED_TH:
    #         return "Sad", "Medium"
    #     else:
    #         return "Sad", "Low"
    # Sad (tuned for sensitivity)
    if sad_score >= 0.35 :
        if sad_score > HIGH_TH:
            return "Sad", "High"
        elif sad_score > MED_TH:
            return "Sad", "Medium"
        else:
            return "Sad", "Low"


    # Fallback
    return "Neutral", None


def draw_face_box(image, lm):
    xs = [int(p[0]) for p in lm.values()]
    ys = [int(p[1]) for p in lm.values()]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.shape[1] - 1, x2 + pad)
    y2 = min(image.shape[0] - 1, y2 + pad)
    cv2.rectangle(image, (x1, y1), (x2, y2), (160, 160, 160), 2)  # grey box


def process_frame(frame):
    global last_face_time, sleep_start, sleeping

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    now = time.time()

    if not res.multi_face_landmarks:
        if now - last_face_time > NOFACE_NOT_ATTENTIVE_AFTER:
            label = ("Neutral", None)
            status = "Not Attentive (No face)"
        else:
            label = ("Neutral", None)
            status = "Attentive (Face lost briefly)"
        return frame, label, status, False, None

    last_face_time = now
    lms = res.multi_face_landmarks[0]
    h, w = frame.shape[:2]
    lm = {i: (lms.landmark[i].x * w, lms.landmark[i].y * h) for i in range(len(lms.landmark))}

    # === Face bounding box ===
    xs = [int(p[0]) for p in lm.values()]
    ys = [int(p[1]) for p in lm.values()]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    pad = 15
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 💚 green box

    # === Eye + sleep detection ===
    ear_left = eye_aspect_ratio(LEFT_EYE, lm)
    ear_right = eye_aspect_ratio(RIGHT_EYE, lm)
    ear = (ear_left + ear_right) / 2.0
    eyes_closed = ear < EYE_CLOSED_EAR

    if eyes_closed:
        if sleep_start is None:
            sleep_start = now
        sleeping = (now - sleep_start) >= SLEEP_SECS
    else:
        sleep_start = None
        sleeping = False

    # === Emotion classify ===
    emotion, intensity = classify_emotion(lm)

    if sleeping:
        status = "Not Attentive (Sleeping)"
    else:
        if emotion == "Neutral":
            status = "Attentive (Neutral)"
        elif intensity in ("Low", "Medium"):
            status = f"Attentive ({emotion} {intensity})"
        elif intensity == "High":
            status = f"Not Attentive ({emotion} High)"
        else:
            status = "Attentive"

    hud = f"{emotion}{' | ' + intensity if intensity else ''} | {'Eyes Closed' if eyes_closed else 'Eyes Open'} | {status}"
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)

    return frame, (emotion, intensity), status, eyes_closed, (x1, y1, x2-x1, y2-y1) if 'x1' in locals() else None


def gen_frames():
    global cap
    if cap is None:  # open camera only when /video_feed is requested
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame, label, status, _, face_box = process_frame(frame)


        last_labels.append((label[0], label[1], status))
        majority = Counter([x[2] for x in last_labels]).most_common(1)[0][0]
        cv2.putText(frame, f"Smoothed: {majority}", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue
        jpg = buffer.tobytes()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'


@app.route('/')
def index():
    return render_template('index.html') if (app.jinja_loader and app.jinja_loader.searchpath) else "ClassFocus running"


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        img_b64 = data["image"].split(",")[1]
        nparr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        frame, (emotion, intensity), status, eyes_closed, face_box = process_frame(img)

        probability = 0.9 if face_box is not None else 0.0

        return jsonify({
            "dominant": emotion,
            "intensity": intensity if intensity else "",
            "attentive": status.startswith("Attentive"),
            "sleeping": "Sleeping" in status,
            "probability": probability,
            "face_box": list(face_box) if face_box is not None else None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/status')
def status():
    if last_labels:
        emo, inten, stat = last_labels[-1]
        return jsonify({
            "emotion": emo,
            "intensity": inten if inten else "",
            "status": stat,
            "attentive": stat.startswith("Attentive")
        })
    else:
        return jsonify({
            "emotion": "Neutral",
            "intensity": "",
            "status": "Booting",
            "attentive": True
        })


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        if cap is not None:
            cap.release()
