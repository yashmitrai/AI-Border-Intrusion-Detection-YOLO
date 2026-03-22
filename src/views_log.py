import sqlite3

conn = sqlite3.connect("intrusion_logs.db")
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

cur.execute("PRAGMA table_info(logs)")
columns = [row[1] for row in cur.fetchall()]
if "zone" not in columns:
    cur.execute("ALTER TABLE logs ADD COLUMN zone TEXT")
    conn.commit()

rows = cur.execute(
    "SELECT id, timestamp, event, zone, image_path, video_path FROM logs ORDER BY id DESC"
).fetchall()

print("\n📌 Intrusion History:\n")
if not rows:
    print("No logs found yet. Run intrusion_logs.py once and trigger intrusion.")
else:
    for r in rows:
        print(r)

conn.close()
