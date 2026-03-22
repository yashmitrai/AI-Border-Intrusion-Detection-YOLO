# AI Border Intrusion Detection (YOLOv8 + Streamlit)

A real-time border intrusion detection system built using YOLOv8 for object detection and a Streamlit dashboard for monitoring and logging intrusion events.

---

## Features
- Real-time intrusion detection using YOLOv8n
- Evidence capture for detected intrusions
- Intrusion logging system to store detection records
- Streamlit dashboard for viewing logs and monitoring activity

---

## Tech Stack
- Python
- Ultralytics YOLOv8
- OpenCV
- Streamlit

---

## Project Structure
```txt
src/
  dashboard.py
  intrusion_logs.py
  intrusion_record.py
  views_log.py
weights/
  yolov8n.pt
evidence/
assets/