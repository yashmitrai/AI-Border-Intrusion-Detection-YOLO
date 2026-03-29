# 🛡️ AI BORDER INTRUSION DETECTION COMMAND CENTRE (V3.0)

A high-performance, real-time border surveillance and intrusion detection system. This installation utilizes **YOLOv8** for tracking and human detection, coupled with a hybrid **Flask/Streamlit** architecture to provide a seamless MJPEG tactical feed and evidence management.

---

## 🎖️ CORE ARCHITECTURE

This system is built as a microservice-ready installation:
*   **Tactical Backend (`intrusion_logs.py`)**: A multi-threaded Flask server that handles raw camera capture, YOLO detection logic, vertical sector mapping, MJPEG frame generation, and RESTful data endpoints.
*   **Command Centre (`dashboard.py`)**: A Streamlit-wrapped HTML5 bridge that auto-boots the backend hardware and displays the tactical radar feed, encrypted logs, and visual evidence locker.

---

## ⚡ FEATURES

*   **Vertical Tactical Sectors**: Real-time left/middle/right sector mapping for Safe, Warning, and Critical zones.
*   **Embedded Tactical Feed**: High-speed MJPEG stream embedded directly into the browser dashboard.
*   **Automated Evidence Locker**: Instant capture of high-resolution stills and surveillance video recordings during intrusion.
*   **Encrypted Log Registry**: Local persistence of all engagements in JSON and SQLite for historical auditing.
*   **Remote Telegram Alerts**: Instant high-resolution photo alerts with verified timestamps sent directly to HQ Telegram Bot.
*   **Data Management**: Integrated "Delete" functionality to cleanse evidence folders and log registries directly from HQ.

---

## 🔧 INSTALLATION & LAUNCH

### 1. Configure HQ Hardware
Ensure your environment variables for the Telegram link are correctly mapped in `src/event_manager.py`:
*   `TELEGRAM_BOT_TOKEN`
*   `TELEGRAM_CHAT_ID`

### 2. Deployment
Execute the following from the root directory:
```bash
# Activate Surveillance Environment
source venv/bin/activate

# Launch Unified Command Centre
streamlit run src/dashboard.py
```
*Port 8501: Dashboard UI*
*Port 5050: Tactical Backend (Auto-boots)*

---

## 📡 TACTICAL NAVIGATION

1.  **DASHBOARD**: Real-time radar monitoring and live metrics.
2.  **SYSTEM LOGS**: Tactical historical record of detected engagements.
3.  **EVIDENCE LOCKER**: Gallery of captured visual data for post-engagement review.

---

## 🛠️ TECH STACK

*   **Logic**: Python 3.x
*   **Vision**: Ultralytics YOLOv8, OpenCV
*   **API**: Flask Core with CORS
*   **Interface**: Streamlit (Container), HTML5 / CSS3 (Military Framework)
*   **Database**: SQLite3 + JSON Persistence