import os
import cv2
import time
import queue
import threading
import requests
from datetime import datetime

TELEGRAM_BOT_TOKEN = "8386179312:AAFTl9sUROIxS6823O74S61dU3GhtXLRRWA".strip()
TELEGRAM_CHAT_ID = "5719400553".strip()

# Validate and mask tokens
def mask_string(s):
    if len(s) > 8:
        return s[:4] + "*" * (len(s)-8) + s[-4:]
    return "***"

print(f"🔧 [DEBUG] INIT TELEGRAM: BOT_TOKEN={mask_string(TELEGRAM_BOT_TOKEN)}, CHAT_ID={mask_string(TELEGRAM_CHAT_ID)}")

# Startup test removed to avoid user confusion


image_queue = queue.Queue()
telegram_queue = queue.Queue()
video_queue = queue.Queue()

def image_worker():
    while True:
        try:
            task = image_queue.get()
            if task is None: continue
            frame, path, timestamp = task
            os.makedirs(os.path.dirname(path), exist_ok=True)
            success = cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            if success:
                print(f"✅ IMAGE SAVED SUCCESS: {path}")
                print("🔧 [DEBUG] TASK ADDED TO TELEGRAM QUEUE")
                telegram_queue.put((path, timestamp))
            else:
                print(f"❌ IMAGE SAVED FAILED: {path}")
        except Exception as e:
            print(f"❌ IMAGE SAVED ERROR: {e}")

def telegram_worker():
    while True:
        try:
            task = telegram_queue.get()
            if task is None: continue
            image_path, timestamp = task
            
            print("🔧 [DEBUG] TELEGRAM WORKER EXECUTING")
            
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                print("⚠️ Telegram Config missing! TELEGRAM SENT FAILED.")
                continue
                
            caption = f"🚨 Intrusion Detected\nTime: {timestamp}\nTargets Tracked By System"
            photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            
            photo_sent = False
            print("🔧 [DEBUG] SENDING TELEGRAM PHOTO...")
            try:
                if os.path.exists(image_path):
                    with open(image_path, "rb") as f:
                        fname = os.path.basename(image_path)
                        files_payload = {"photo": (fname, f, "image/jpeg")}
                        photo_response = requests.post(
                            photo_url, 
                            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, 
                            files=files_payload, 
                            timeout=30 # Increased timeout for slow image uploads
                        )
                    
                    print("🔧 [DEBUG] RESPONSE RECEIVED")
                    print(f"🔧 [DEBUG] Status Code: {photo_response.status_code}")
                    
                    if photo_response.status_code == 200:
                        print("✅ TELEGRAM PHOTO SENT SUCCESS")
                        photo_sent = True
                    else:
                        print(f"❌ TELEGRAM PHOTO SENT FAILED: {photo_response.text}")
                else:
                    print(f"❌ TELEGRAM PHOTO FAILED: Image file not found {image_path}")
            except Exception as pe:
                print(f"❌ TELEGRAM PHOTO ERROR: {str(pe)}")
                
            # Fallback to Text Message if Photo failed
            if not photo_sent:
                print("🔧 [DEBUG] FALLBACK: SENDING TELEGRAM TEXT ALERT...")
                text_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                text_payload = {"chat_id": TELEGRAM_CHAT_ID, "text": f"⚠️ IMAGE FAILED TO UPLOAD.\n{caption}"}
                try:
                    text_response = requests.post(text_url, data=text_payload, timeout=10)
                    if text_response.status_code == 200:
                        print("✅ TELEGRAM TEXT SENT SUCCESS")
                    else:
                        print(f"❌ TELEGRAM TEXT FAILED: {text_response.text}")
                except Exception as text_e:
                    print(f"❌ TELEGRAM TEXT ERROR: {str(text_e)}")

        except Exception as e:
            print(f"❌ TELEGRAM ERROR: {str(e)}")

def video_worker():
    writer = None
    target_stop_time = 0
    while True:
        try:
            task = video_queue.get(timeout=0.1)
            msg_type = task.get("type")
            if msg_type == "START":
                path = task.get("path")
                fps = task.get("fps", 20.0)
                w, h = task.get("size")
                buffer = task.get("buffer", [])
                
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                if writer is not None:
                    writer.release()
                writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
                if writer.isOpened():
                    for bf in buffer:
                        writer.write(bf)
                    target_stop_time = time.time() + 10.0 # Extra 10s recording
                else:
                    print(f"❌ VIDEO SAVED FAILED: could not open {path}")
                    writer = None
            elif msg_type == "FRAME":
                if writer is not None:
                    writer.write(task.get("frame"))
        except queue.Empty:
            if writer is not None and time.time() > target_stop_time:
                writer.release()
                writer = None
                print("✅ VIDEO SAVED SUCCESS")
        except Exception as e:
            print(f"❌ VIDEO SAVED ERROR: {str(e)}")
            if writer is not None:
                writer.release()
                writer = None

threading.Thread(target=image_worker, daemon=True).start()
threading.Thread(target=telegram_worker, daemon=True).start()
threading.Thread(target=video_worker, daemon=True).start()

def handle_intrusion_event(frame, detection_data, w, h, pre_intrusion_buffer):
    print("🚨 INTRUSION EVENT TRIGGERED")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    img_path = f"evidence/intrusion_{timestamp}.jpg"
    video_path = f"evidence/intrusion_{timestamp}.mp4"
    
    image_queue.put((frame.copy(), img_path, timestamp))
    video_queue.put({
        "type": "START",
        "path": video_path,
        "fps": 20.0,
        "size": (w, h),
        "buffer": pre_intrusion_buffer
    })
