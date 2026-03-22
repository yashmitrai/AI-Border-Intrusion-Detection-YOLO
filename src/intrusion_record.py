import cv2
import os
import sys
import time
from datetime import datetime
from ultralytics import YOLO
import threading
import queue
from collections import deque
import event_manager
import state_manager

# Load YOLOv8 model
import torch
device = 0 if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
model = YOLO("yolov8n.pt")
model.to(device)

# Open webcam or video file
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
    print("❌ Camera not opened. Try changing index 0 -> 1 or 2.")
    exit()

# Create evidence folder
os.makedirs("evidence", exist_ok=True)

print("✅ Press 'q' to quit.")
print(f"📹 Source: {source}")

# Recording settings
recording = False
video_writer = None
record_end_time = 0
cooldown_end_time = 0  # prevents saving again immediately
last_warning_log_time = 0
last_critical_log_time = 0

frame_count = 0
last_detections = []
pre_intrusion_buffer = deque(maxlen=100)  # 5 seconds at 20fps
zone_state_manager = state_manager.ZoneStateManager()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break
        
    frame_count += 1

    # Resize input frame to 512x512 for optimization
    frame = cv2.resize(frame, (512, 512))
    raw_frame = frame
    h, w, _ = raw_frame.shape
    zone1_end = h // 3
    zone2_end = (h * 2) // 3

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

                        if cy < zone1_end:
                            zone_label = "SAFE ZONE"
                            zone_color = (0, 255, 0)
                        elif cy < zone2_end:
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

    # Status text (Reduced logging overhead: log every 3 seconds)
    now = time.time()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if warning_detected and (now - last_warning_log_time) > 3.0:
        print(f"{timestamp_str} - Object tracked in WARNING ZONE")
        last_warning_log_time = now

    if intrusion and (now - last_critical_log_time) > 3.0:
        print(f"{timestamp_str} - Object tracked in CRITICAL ZONE – INTRUSION DETECTED")
        last_critical_log_time = now

    # Draw zones overlay (after detection so YOLO sees clean frames)
    frame = raw_frame.copy()
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, zone1_end), (0, 255, 0), -1)
    cv2.rectangle(overlay, (0, zone1_end), (w, zone2_end), (0, 255, 255), -1)
    cv2.rectangle(overlay, (0, zone2_end), (w, h), (0, 0, 255), -1)
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    cv2.line(frame, (0, zone1_end), (w, zone1_end), (0, 255, 0), 2)
    cv2.line(frame, (0, zone2_end), (w, zone2_end), (0, 255, 255), 2)
    cv2.putText(frame, "SAFE ZONE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "WARNING ZONE", (10, zone1_end + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "CRITICAL ZONE", (10, zone2_end + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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

    # Start evidence capture only when INTRUSION happens AND cooldown passed
    if final_state == "INTRUSION" and (now > cooldown_end_time):
        event_manager.handle_intrusion_event(frame, detections, w, h, list(pre_intrusion_buffer))
        recording = True
        record_end_time = now + 10
        cooldown_end_time = now + 15

    # If recording, write frames
    if recording:
        event_manager.video_queue.put({"type": "FRAME", "frame": frame.copy()})
        if now >= record_end_time:
            recording = False
    else:
        pre_intrusion_buffer.append(frame.copy())

    # Show frame
    cv2.imshow("AI Border Intrusion Detection (Recording)", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
