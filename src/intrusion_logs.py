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
    if os.path.exists('evidence_archive'):
        for f in sorted(os.listdir('evidence_archive'), reverse=True):
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
    return send_from_directory(os.path.join(os.getcwd(), 'evidence_archive'), filename)

@app.route('/delete_evidence', methods=['POST'])
def delete_evidence():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({"success": False, "error": "No filename"}), 400
    
    path = os.path.join('evidence_archive', filename)
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

# ---------- System Logs Setup ----------
os.makedirs("system_logs", exist_ok=True)
def write_system_log(event, confidence, obj_type, zone, timestamp=None):
    # Non-blocking log structure per Requirement 4
    if timestamp is None: timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = "system_logs/log.txt"
    try:
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] EVENT: {event} | ZONE: {zone} | CONFIDENCE: {confidence:.2f} | TYPE: {obj_type}\n")
    except Exception as e:
        print(f"Log writing failed: {e}")

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
print("🔧 [DEBUG] Loading Torch...")
import torch
device = 0 if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
print(f"🔧 [DEBUG] Torch Device: {device}")

print("🔧 [DEBUG] Loading YOLO...")
model = YOLO("src/yolov8n.pt" if os.path.exists("src/yolov8n.pt") else "yolov8n.pt")
model.to(device)
print("🔧 [DEBUG] YOLO Loaded.")

# Dictionary of class ID to name, mostly 0=person, 2=car etc in COCO
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 7: "truck"}

# ---------- Camera / Video Source ----------
source = 0
if len(sys.argv) > 1:
    source = sys.argv[1]
    if isinstance(source, str) and source.isdigit():
        source = int(source)

def init_camera(src):
    # Mac compatibility cv2.CAP_AVFOUNDATION logic
    return cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION) if isinstance(src, int) else cv2.VideoCapture(src)

print(f"🔧 [DEBUG] Opening VideoCapture with source: {source}")
cap = init_camera(source)

# Fallback handling for Mac indexes
if not cap.isOpened() and isinstance(source, int):
    print("⚠️ Camera index 0 failed. Failing back to index 1...")
    cap = init_camera(1)

if not cap.isOpened():
    print("❌ Camera not opened. Please check macOS Camera privacy permissions.")
    sys.exit(1)

# ---------- Evidence Folder ----------
# Auto-creates folder if it does not exist (Requirement 3)
os.makedirs("evidence_archive", exist_ok=True)

print("✅ Press 'q' to quit.")
print(f"📹 Source: {source}")

recording = False
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
    
    # Try-catch camera loop recovery for Stability Fix
    try:
        ret, frame = cap.read()
    except Exception as e:
        print(f"❌ Read Crash: {e}. Restarting camera...")
        cap.release()
        time.sleep(1.0)
        cap = init_camera(source)
        continue

    # Graceful handling for missing frames
    if not ret or frame is None or frame.size == 0:
        print("⚠️ Camera lost connection or empty frame. Restarting stream...")
        cap.release()
        time.sleep(1.0)
        cap = init_camera(source)
        continue
        
    frame_count += 1
    
    try:
        # Resize frame before YOLO explicitly specified to optimize processing
        frame = cv2.resize(frame, (640, 480))
    except Exception as resize_err:
        print(f"⚠️ Frame resize error: {resize_err}")
        continue

    raw_frame = frame
    h, w, _ = raw_frame.shape
    zone1_end = w // 3
    zone2_end = (w * 2) // 3

    # Process every 2nd frame, cache otherwise (Skip frames condition implemented to maintain 30-40fps)
    if frame_count % 2 != 0:
        try:
            results = model.track(raw_frame, persist=True, verbose=False, device=device)
        except Exception:
            results = []
            
        intrusion = False
        warning_detected = False
        detections = []
        highest_conf = 0.0
        primary_cls_id = None

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    track_id = int(box.id[0]) if box.id is not None else 0

                    if cls in [0, 1, 2, 3, 7] and conf >= 0.5:  # Supported objects + threshold>=0.5 applied
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Only trigger logic if CENTROID belongs in zone parameters
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
                            if conf > highest_conf:
                                highest_conf = conf
                                primary_cls_id = cls

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

    # Draw zones overlay (vertical lines for Left/Middle/Right mapping)
    frame = raw_frame.copy()

    # Create distinct overlays to render on camera output
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (zone1_end, h), (0, 255, 0), -1)
    cv2.rectangle(overlay, (zone1_end, 0), (zone2_end, h), (0, 255, 255), -1)
    cv2.rectangle(overlay, (zone2_end, 0), (w, h), (0, 0, 255), -1)
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    
    cv2.line(frame, (zone1_end, 0), (zone1_end, h), (0, 255, 0), 2)
    cv2.line(frame, (zone2_end, 0), (zone2_end, h), (0, 255, 255), 2)
    cv2.putText(frame, "SAFE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "WARNING", (zone1_end + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "CRITICAL", (zone2_end + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Frame With Bounding Boxes + Zone overlays
    for (x1, y1, x2, y2, cx, cy, conf, zone_label, zone_color, track_id) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"Target {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, zone_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
                    
    raw_state = "SAFE"
    if intrusion:
        raw_state = "INTRUSION"
    elif warning_detected:
        raw_state = "WARNING"
        
    final_state = zone_state_manager.update(raw_state)

    now = time.time()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Only log explicitly when state changes effectively via ZoneStateManager Debounce Logic (Issue 6)
    if final_state == "WARNING" and (now - last_warning_log_time) > 3.0:
        write_system_log("WARNING", detections[0][6] if len(detections)>0 else 0.5, "person", "WARNING_ZONE", timestamp_str)
        last_warning_log_time = now

    if final_state == "INTRUSION" and (now > cooldown_end_time):
        event_manager.handle_intrusion_event(frame, detections, w, h, list(pre_intrusion_buffer))
        recording = True
        record_end_time = now + 10
        cooldown_end_time = now + 15

        c_val = highest_conf if 'highest_conf' in locals() else 1.0
        cls_v = CLASS_NAMES.get(primary_cls_id, "unknown") if 'primary_cls_id' in locals() else "unknown"

        write_system_log("INTRUSION", c_val, cls_v, "CRITICAL", timestamp_str)

        # Update Database Log with evidence_archive path
        timestamp_slug = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_path = f"evidence_archive/intrusion_{timestamp_slug}.jpg"
        video_path = f"evidence_archive/intrusion_{timestamp_slug}.mp4"
        
        try:
            cur.execute(
                "INSERT INTO logs (timestamp, event, zone, image_path, video_path) VALUES (?, ?, ?, ?, ?)",
                (timestamp_slug, "INTRUSION", "CRITICAL", img_path, video_path)
            )
            conn.commit()
        except Exception:
            pass
        
        # Explicit JSON Appending for UI API Fetch
        try:
            with open("intrusion_log.json", "r") as f:
                json_data = json.load(f)
        except:
            json_data = []
            
        json_data.insert(0, {
            "timestamp": timestamp_slug,
            "event": f"INTRUSION DETECTED ({cls_v.upper()})",
            "zone": "CRITICAL",
            "image_path": img_path,
            "video_path": video_path,
            "confidence": f"{c_val:.2f}"
        })
        
        try:
            with open("intrusion_log.json", "w") as f:
                json.dump(json_data, f, indent=4)
        except:
            pass
            

    if final_state == "INTRUSION":
        cv2.putText(frame, "🚨 INTRUSION DETECTED!", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    elif final_state == "WARNING":
        cv2.putText(frame, "WARNING ZONE ACTIVITY", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    else:
        cv2.putText(frame, "SAFE", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    
    latest_stats["final_state"] = final_state

    if recording:
        event_manager.video_queue.put({"type": "FRAME", "frame": frame.copy()})
        if now >= record_end_time:
            recording = False
    else:
        pre_intrusion_buffer.append(frame.copy())

    # Update frame variable for Flask
    latest_frame = frame.copy()

    loop_end_time = time.time()
    fps_buffer.append(1.0 / max((loop_end_time - loop_start_time), 0.001))
    latest_stats["fps"] = sum(fps_buffer) / len(fps_buffer)

    # Required for MacOS AVFoundation camera event pumping, ensures no freezing bugs!
    cv2.waitKey(1)

cap.release()
conn.close()
