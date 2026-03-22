import cv2
import os
import sys
import time
import sqlite3
import json
from datetime import datetime
from ultralytics import YOLO
import threading
import queue
from collections import deque
import event_manager
import state_manager
from flask import Flask, Response, jsonify, send_from_directory, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable explicit frontend isolation
latest_frame = None
latest_stats = {
    "safe_count": 0,
    "warning_count": 0,
    "danger_count": 0,
    "final_state": "SAFE",
    "fps": 0.0
}
fps_buffer = deque(maxlen=20)


def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            # 3. CREATE VIDEO STREAM ENDPOINT
            _, buffer = cv2.imencode('.jpg', latest_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01) # Avoid tight-loop cpu spikes


@app.route('/')
def serve_dashboard():
    # Root endpoint seamlessly serves the HTML Javascript Dashboard
    return send_from_directory(os.path.join(os.getcwd(), 'src'), 'dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats_feed():
    return jsonify(latest_stats)

@app.route('/status')
def status_feed():
    return jsonify({
        "current_zone": latest_stats["final_state"],
        "fps": round(latest_stats.get("fps", 0), 1),
        "system_status": "ONLINE",
        "camera_status": "ON",
        "model_status": "RUNNING"
    })

@app.route('/logs')
def logs_feed():
    if not os.path.exists('intrusion_log.json'):
        return jsonify([])
    try:
        with open('intrusion_log.json', 'r') as f:
            return jsonify(json.load(f))
    except:
        return jsonify([])

@app.route('/evidence')
def evidence_list():
    files = []
    if os.path.exists('evidence'):
        for f in sorted(os.listdir('evidence'), reverse=True):
            if f.endswith('.jpg') or f.endswith('.mp4'):
                files.append({
                    "filename": f,
                    "url": f"/evidence_file/{f}",
                    "type": "image" if f.endswith('.jpg') else "video",
                    "timestamp": f.replace("intrusion_", "").replace(".jpg", "").replace(".mp4", "")
                })
    return jsonify(files)

@app.route('/evidence_file/<path:filename>')
def serve_evidence(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'evidence'), filename)

@app.route('/delete_evidence', methods=['POST'])
def delete_evidence():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({"success": False, "error": "No filename"}), 400
    
    path = os.path.join('evidence', filename)
    if os.path.exists(path):
        try:
            os.remove(path)
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": False, "error": "File not found"}), 404

@app.route('/delete_log', methods=['POST'])
def delete_log():
    data = request.json
    timestamp = data.get('timestamp')
    if not timestamp:
        return jsonify({"success": False, "error": "No timestamp"}), 400
    
    # Update JSON
    try:
        if os.path.exists("intrusion_log.json"):
            with open("intrusion_log.json", "r") as f:
                logs = json.load(f)
            new_logs = [l for l in logs if l.get('timestamp') != timestamp]
            with open("intrusion_log.json", "w") as f:
                json.dump(new_logs, f, indent=4)
    except:
        pass

    # Update SQLite
    try:
        c = sqlite3.connect("intrusion_logs.db")
        curr = c.cursor()
        curr.execute("DELETE FROM logs WHERE timestamp = ?", (timestamp,))
        c.commit()
        c.close()
    except:
        pass

    return jsonify({"success": True})

def run_flask():
    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)

threading.Thread(target=run_flask, daemon=True).start()

# ---------- File Setup ----------
if not os.path.exists("intrusion_log.json"):
    with open("intrusion_log.json", "w") as f:
        json.dump([], f)



# ---------- DB Setup ----------
DB_FILE = "intrusion_logs.db"

conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    event TEXT,
    zone TEXT,
    image_path TEXT,
    video_path TEXT
)
""")
conn.commit()

# Add zone column if it does not exist (for older DBs)
cur.execute("PRAGMA table_info(logs)")
columns = [row[1] for row in cur.fetchall()]
if "zone" not in columns:
    cur.execute("ALTER TABLE logs ADD COLUMN zone TEXT")
    conn.commit()

# ---------- Model ----------
import torch
device = 0 if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
model = YOLO("yolov8n.pt")
model.to(device)

# ---------- Camera / Video Source ----------
source = 0
if len(sys.argv) > 1:
    source = sys.argv[1]
    if isinstance(source, str) and source.isdigit():
        source = int(source)

if isinstance(source, int):
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("❌ Camera not opened. Allow camera permission or try index 1/2.")
    exit()

# ---------- Evidence Folder ----------
os.makedirs("evidence", exist_ok=True)

print("✅ Press 'q' to quit.")
print(f"📹 Source: {source}")

recording = False
video_writer = None
record_end_time = 0
cooldown_end_time = 0
last_warning_log_time = 0
last_critical_log_time = 0

frame_count = 0
last_detections = []
pre_intrusion_buffer = deque(maxlen=100)  # 5 seconds at 20fps
zone_state_manager = state_manager.ZoneStateManager()

while True:
    loop_start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break


    frame_count += 1

    # Resize input frame to 512x512 for optimization
    frame = cv2.resize(frame, (640, 480))
    raw_frame = frame
    h, w, _ = raw_frame.shape
    zone1_end = w // 3
    zone2_end = (w * 2) // 3

    # Process every 2nd frame, cache otherwise
    if frame_count % 2 != 0:
        # YOLO tracking (for smooth object tracking)
        results = model.track(raw_frame, persist=True, verbose=False, device=device)
        intrusion = False
        warning_detected = False
        detections = []

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    track_id = int(box.id[0]) if box.id is not None else 0

                    if cls in [0, 1, 2, 3, 7] and conf > 0.5:  # person, bicycle, car, motorcycle, truck
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        if cx < zone1_end:
                            zone_label = "SAFE ZONE"
                            zone_color = (0, 255, 0)
                        elif cx < zone2_end:
                            zone_label = f"WARNING [ID:{track_id}]"
                            zone_color = (0, 255, 255)
                            warning_detected = True
                        else:
                            zone_label = f"CRITICAL [ID:{track_id}]"
                            zone_color = (0, 0, 255)
                            intrusion = True

                        detections.append((x1, y1, x2, y2, cx, cy, conf, zone_label, zone_color, track_id))
        
        last_detections = detections
    else:
        # Use cached detections
        detections = last_detections

    # Update global stats for dashboard
    safe_v = 0
    warn_v = 0
    dang_v = 0
    for d in detections:
        l = d[7]
        if "SAFE" in l: safe_v += 1
        elif "WARNING" in l: warn_v += 1
        elif "CRITICAL" in l: dang_v += 1
        
    latest_stats["safe_count"] = safe_v
    latest_stats["warning_count"] = warn_v
    latest_stats["danger_count"] = dang_v

    # Status text (Reduced logging overhead: log every 3 seconds)
    now = time.time()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if warning_detected and (now - last_warning_log_time) > 3.0:
        print(f"{timestamp_str} - Object tracked in WARNING ZONE")
        last_warning_log_time = now

    if intrusion and (now - last_critical_log_time) > 3.0:
        print(f"{timestamp_str} - Object tracked in CRITICAL ZONE – INTRUSION DETECTED")
        last_critical_log_time = now

    # Draw zones overlay (vertical lines for Left/Middle/Right mapping)
    frame = raw_frame.copy()
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (zone1_end, h), (0, 255, 0), -1)
    cv2.rectangle(overlay, (zone1_end, 0), (zone2_end, h), (0, 255, 255), -1)
    cv2.rectangle(overlay, (zone2_end, 0), (w, h), (0, 0, 255), -1)
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    cv2.line(frame, (zone1_end, 0), (zone1_end, h), (0, 255, 0), 2)
    cv2.line(frame, (zone2_end, 0), (zone2_end, h), (0, 255, 255), 2)
    cv2.putText(frame, "SAFE", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "WARNING", (zone1_end + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "CRITICAL", (zone2_end + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for (x1, y1, x2, y2, cx, cy, conf, zone_label, zone_color, track_id) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"Target {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, zone_label, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
                    
    raw_state = "SAFE"
    if intrusion:
        raw_state = "INTRUSION"
    elif warning_detected:
        raw_state = "WARNING"
        
    final_state = zone_state_manager.update(raw_state)

    if final_state == "INTRUSION":
        cv2.putText(frame, "🚨 INTRUSION DETECTED!", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    elif final_state == "WARNING":
        cv2.putText(frame, "WARNING ZONE ACTIVITY", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    else:
        cv2.putText(frame, "SAFE", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    
    latest_stats["final_state"] = final_state

    # Stream frame to Flask endpoint globally
    global_frame = frame.copy()

    # Start evidence capture only when INTRUSION happens AND cooldown passed
    if final_state == "INTRUSION" and (now > cooldown_end_time):
        event_manager.handle_intrusion_event(frame, detections, w, h, list(pre_intrusion_buffer))
        recording = True
        record_end_time = now + 10
        cooldown_end_time = now + 15

        # Update Database Log
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_path = f"evidence/intrusion_{timestamp}.jpg"
        video_path = f"evidence/intrusion_{timestamp}.mp4"
        
        cur.execute(
            "INSERT INTO logs (timestamp, event, zone, image_path, video_path) VALUES (?, ?, ?, ?, ?)",
            (timestamp, "INTRUSION", "CRITICAL", img_path, video_path)
        )
        conn.commit()
        
        # Explicit JSON Appending for UI API Fetch
        try:
            with open("intrusion_log.json", "r") as f:
                json_data = json.load(f)
        except:
            json_data = []
            
        json_data.insert(0, {
            "timestamp": timestamp,
            "event": "INTRUSION DETECTED",
            "zone": "CRITICAL",
            "image_path": img_path,
            "video_path": video_path,
            "confidence": "HIGH"
        })
        
        with open("intrusion_log.json", "w") as f:
            json.dump(json_data, f, indent=4)
            
        print("🗃️ Log saved to database & JSON (Zone: CRITICAL).")

    # Write video frames
    if recording:
        event_manager.video_queue.put({"type": "FRAME", "frame": frame.copy()})
        if now >= record_end_time:
            recording = False
    else:
        pre_intrusion_buffer.append(frame.copy())

    # 2. STORE FRAME FOR DASHBOARD ONLY
    latest_frame = frame.copy()

    loop_end_time = time.time()
    fps_buffer.append(1.0 / max((loop_end_time - loop_start_time), 0.001))
    latest_stats["fps"] = sum(fps_buffer) / len(fps_buffer)

cap.release()
conn.close()

